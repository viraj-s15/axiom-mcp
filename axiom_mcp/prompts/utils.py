"""Utility functions and classes for prompt management."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Generic, ParamSpec, TypeVar, cast, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, Field

from axiom_mcp.exceptions import PromptRenderError

from .base import Message, Prompt

T = TypeVar("T")
P = ParamSpec("P")


class FunctionInfo(BaseModel):
    """Information about an executable function."""

    name: str
    description: str | None = None
    source: str | None = None
    return_type: str
    parameters: dict[str, tuple[str, Any]] = Field(default_factory=dict)
    is_async: bool = False
    is_generator: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ExecutableFunction(Generic[P, T]):
    """A wrapper for executable functions with metadata."""

    def __init__(self, fn: Callable[P, T], name: str | None = None) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.info = self._create_function_info()

    def _create_function_info(self) -> FunctionInfo:
        """Create function info from inspection."""
        sig = inspect.signature(self.fn)
        type_hints = get_type_hints(self.fn)
        return_type = type_hints.get("return", Any).__name__
        parameters = {
            name: (param.annotation.__name__, param.default)
            for name, param in sig.parameters.items()
        }

        return FunctionInfo(
            name=self.name,
            description=self.fn.__doc__,
            source=inspect.getsource(self.fn),
            return_type=return_type,
            parameters=parameters,
            is_async=asyncio.iscoroutinefunction(self.fn),
            is_generator=inspect.isgeneratorfunction(self.fn),
        )

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute the function with given arguments."""
        if self.info.is_async:
            return await cast(Callable[P, Awaitable[T]], self.fn)(*args, **kwargs)
        return cast(Callable[P, T], self.fn)(*args, **kwargs)


class FunctionRegistry:
    """Registry for executable functions with metadata."""

    def __init__(self) -> None:
        self._functions: dict[str, ExecutableFunction[Any, Any]] = {}
        self._prompts: dict[str, Prompt] = {}

    @overload
    def register(
        self, fn: None = None, *, name: str | None = None
    ) -> Callable[[Callable[P, T]], ExecutableFunction[P, T]]: ...

    @overload
    def register(
        self, fn: Callable[P, T], *, name: str | None = None
    ) -> ExecutableFunction[P, T]: ...

    def register(
        self, fn: Callable[P, T] | None = None, *, name: str | None = None
    ) -> (
        Callable[[Callable[P, T]], ExecutableFunction[P, T]] | ExecutableFunction[P, T]
    ):
        """Register a function or return a decorator."""
        if fn is None:
            return lambda f: self.register(f, name=name)

        func = ExecutableFunction(fn, name)
        self._functions[func.name] = func
        return func

    def unregister(self, name: str) -> None:
        """Remove a function from the registry."""
        self._functions.pop(name, None)

    def get(self, name: str) -> ExecutableFunction[Any, Any] | None:
        """Get a registered function by name."""
        return self._functions.get(name)

    def list_functions(self) -> dict[str, FunctionInfo]:
        """List all registered functions."""
        return {name: func.info for name, func in self._functions.items()}

    async def execute(
        self, name: str, args: dict[str, Any] | None = None
    ) -> list[Message]:
        """Execute a registered function with given arguments."""
        func = self.get(name)
        if not func:
            raise PromptRenderError(name, f"Function {name} not found")

        args = args or {}
        result = await func(**args)

        if not isinstance(result, list | tuple):
            result = [result]

        return [
            msg if isinstance(msg, Message) else Message(content=msg, role="assistant")
            for msg in result
        ]

    def add_prompt(self, prompt: Prompt) -> None:
        """Add a prompt to the registry."""
        self._prompts[prompt.name] = prompt

    def get_prompt(self, name: str) -> Prompt | None:
        """Get a prompt by name."""
        return self._prompts.get(name)


def prompt(
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    registry: FunctionRegistry | None = None,
) -> Callable[
    [Callable[..., str | Message | Sequence[str | Message]]],
    Callable[..., str | Message | Sequence[str | Message]],
]:
    """Decorator to create a prompt from a function.

    Args:
        name: Optional name for the prompt (defaults to function name)
        description: Optional description (defaults to function docstring)
        tags: Optional list of tags for categorizing prompts
        registry: Optional registry to use (defaults to global registry)
    """

    def decorator(
        fn: Callable[..., str | Message | Sequence[str | Message]],
    ) -> Callable[..., str | Message | Sequence[str | Message]]:
        @wraps(fn)
        def wrapper(
            *args: Any, **kwargs: Any
        ) -> str | Message | Sequence[str | Message]:
            return fn(*args, **kwargs)

        # Create the prompt
        prompt = Prompt.from_function(
            fn=wrapper,
            name=name or fn.__name__,
            description=description or fn.__doc__,
            tags=tags or [],
        )

        target_registry = registry or globals()["registry"]

        target_registry._functions[prompt.name] = ExecutableFunction(
            wrapper, prompt.name
        )
        target_registry.add_prompt(prompt)

        return wrapper

    return decorator


def batch_register(*prompts: Prompt) -> None:
    """Register multiple prompts at once."""
    for p in prompts:
        registry.add_prompt(p)


def combine_prompts(*prompt_funcs: Callable[..., Any]) -> Prompt:
    """Combine multiple prompt functions into a single prompt."""

    def combined(**kwargs: Any) -> Sequence[Message]:
        messages: list[Message] = []
        for fn in prompt_funcs:
            result = fn(**kwargs)
            if isinstance(result, list | tuple):
                messages.extend(result)
            else:
                messages.append(Message(content=str(result), role="assistant"))
        return messages

    first_func = prompt_funcs[0]
    names = ", ".join(f.__name__ for f in prompt_funcs)
    return Prompt.from_function(
        fn=combined,
        name=f"combined_{first_func.__name__}",
        description=f"Combined prompt from {names}",
        tags=["combined"],
    )


# Global registry instance
registry = FunctionRegistry()
