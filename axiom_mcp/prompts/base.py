"""Base classes and types for prompt management."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from mcp.types import EmbeddedResource, ImageContent
from mcp.types import TextContent as MCPTextContent
from pydantic import BaseModel, ConfigDict, Field, validate_call
from typing_extensions import runtime_checkable

from axiom_mcp.exceptions import (
    InvalidMessageRoleError,
    LambdaNameError,
    MissingArgumentsError,
    PromptRenderError,
)

# Define ContentType using type
type ContentType = (
    str | dict[str, Any] | MCPTextContent | ImageContent | EmbeddedResource
)


class Message(BaseModel):
    """Base class for all prompt messages."""

    role: Literal["user", "assistant", "system"]
    content: ContentType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, content: ContentType, **kwargs: Any) -> None:
        if isinstance(content, str):
            content = MCPTextContent(type="text", text=content)
        elif isinstance(content, dict):
            if content.get("type") == "text":
                content = MCPTextContent(**content)
            elif content.get("type") == "image":
                content = ImageContent(**content)
            elif content.get("type") == "resource":
                content = EmbeddedResource(**content)
        super().__init__(content=content, **kwargs)

    def __str__(self) -> str:
        if isinstance(self.content, MCPTextContent):
            return self.content.text
        if isinstance(self.content, ImageContent):
            return f"[Image: {getattr(self.content, 'description', '')}]"
        if isinstance(self.content, EmbeddedResource):
            return f"[Resource: {getattr(self.content, 'uri', '')}]"
        return str(self.content)


class UserMessage(Message):
    """A message from the user."""

    role: Literal["user", "assistant", "system"] = Field("user", frozen=True)


class AssistantMessage(Message):
    """A message from the assistant."""

    role: Literal["user", "assistant", "system"] = Field("assistant", frozen=True)


class SystemMessage(Message):
    """A system message providing context or instructions."""

    role: Literal["user", "assistant", "system"] = Field("system", frozen=True)


@runtime_checkable
class PromptResponse(Protocol):
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> (
        str
        | Message
        | dict[str, Any]
        | Sequence[str | Message | dict[str, Any]]
        | Awaitable[Any]
    ): ...

    __name__: str


class PromptArgument(BaseModel):
    name: str = Field(..., description="Name of the argument")
    description: str | None = Field(
        None, description="Description of what the argument does"
    )
    required: bool = Field(
        default=False, description="Whether the argument is required"
    )
    type_hint: str = Field(..., description="Type hint for the argument")
    default: Any | None = Field(None, description="Default value if any")

    model_config = ConfigDict(frozen=False)


class Prompt(BaseModel):
    """A prompt template that can be rendered with parameters."""

    name: str = Field(..., description="Name of the prompt")
    description: str | None = Field(
        None, description="Description of what the prompt does"
    )
    version: str = Field(default="1.0.0", description="Version of the prompt")
    arguments: list[PromptArgument] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    fn: Callable[..., Any] = Field(..., exclude=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_function(
        cls,
        fn: PromptResponse,
        name: str | None = None,
        description: str | None = None,
        version: str = "1.0.0",
        tags: list[str] | None = None,
    ) -> Prompt:
        func_name = name or fn.__name__
        if func_name == "<lambda>":
            raise LambdaNameError()

        sig = inspect.signature(fn)
        type_hints = inspect.get_annotations(fn)

        arguments = []
        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, Any).__name__
            required = param.default == param.empty
            default = None if param.default == param.empty else param.default

            arguments.append(
                PromptArgument(
                    name=param_name,
                    description=None,
                    required=required,
                    type_hint=param_type,
                    default=default,
                )
            )

        validated_fn = validate_call(fn)

        return cls(
            name=func_name,
            description=description or fn.__doc__ or "",
            version=version,
            arguments=arguments,
            tags=tags or [],
            fn=validated_fn,
        )

    def _create_message(self, msg: Any) -> Message:
        """Create a Message instance from various input types."""
        if isinstance(msg, Message):
            return msg
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            if role == "user":
                return Message(content=msg, role="user")
            if role == "assistant":
                return AssistantMessage(**msg)
            if role == "system":
                return SystemMessage(**msg)
            raise InvalidMessageRoleError(role)
        if isinstance(msg, str):
            return Message(content=msg, role="user")
        return Message(content=str(msg), role="user")

    async def render(self, arguments: dict[str, Any] | None = None) -> list[Message]:
        arguments = arguments or {}

        missing = [
            arg.name
            for arg in self.arguments
            if arg.required and arg.name not in arguments
        ]
        if missing:
            raise MissingArgumentsError(missing)

        try:
            result = self.fn(**arguments)
            if inspect.iscoroutine(result):
                result = await result

            if not isinstance(result, list | tuple):
                result = [result]

            return [self._create_message(msg) for msg in result]

        except Exception as e:
            raise PromptRenderError(self.name, str(e)) from e
