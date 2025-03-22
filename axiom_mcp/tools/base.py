"""Base classes for tool system."""

import abc
import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from ..exceptions import ToolError


class ToolValidationSchema(BaseModel):
    """Schema for tool input/output validation."""

    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


class ToolDependency(BaseModel):
    """Dependency information for a tool."""

    tool_name: str
    required: bool = True
    condition: str | None = None


class ToolMetadata(BaseModel):
    """Tool metadata and configuration."""

    name: str
    version: str = "0.1.0"
    description: str | None = None
    author: str | None = None
    dependencies: list[ToolDependency] = Field(default_factory=list)
    validation: ToolValidationSchema | None = None
    supports_dry_run: bool = False
    supports_streaming: bool = False
    cacheable: bool = False
    cache_ttl: int = 3600  # seconds

    def model_validate_dependencies(
        self, value: list[dict[str, Any]]
    ) -> list[ToolDependency]:
        """Convert dictionary dependencies to ToolDependency objects."""
        return [
            ToolDependency(**dep) if isinstance(dep, dict) else dep for dep in value
        ]


class ToolContext(BaseModel):
    """Execution context for tools."""

    dry_run: bool = False
    cache_enabled: bool = True
    timeout: float | None = None
    validation_enabled: bool = True
    state: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel, abc.ABC):
    """Base class for all tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: ClassVar[ToolMetadata]
    context: ToolContext = Field(default_factory=ToolContext)

    async def validate_input(self, args: dict[str, Any]) -> None:
        """Validate tool input arguments."""
        if not self.context.validation_enabled:
            return

        if not self.metadata.validation or not self.metadata.validation.input_schema:
            return

        try:
            schema = self.metadata.validation.input_schema
            validation_model = create_model(f"{self.metadata.name}Input", **schema)
            validation_model.model_validate(args)
        except ValidationError as e:
            raise ToolError(
                message="Invalid tool input",
                tool_name=self.metadata.name,
                details={"validation_errors": e.errors()},
            )

    async def validate_output(self, result: Any) -> None:
        """Validate tool output."""
        if not self.context.validation_enabled:
            return

        if not self.metadata.validation or not self.metadata.validation.output_schema:
            return

        try:
            schema = self.metadata.validation.output_schema
            validation_model = create_model(f"{self.metadata.name}Output", **schema)
            validation_model.model_validate(result)
        except ValidationError as e:
            raise ToolError(
                message="Invalid tool output",
                tool_name=self.metadata.name,
                details={"validation_errors": e.errors()},
            )

    async def check_dependencies(self) -> None:
        """Check if all required dependencies are available."""
        for dep in self.metadata.dependencies:
            # In a real implementation, this would check the dependency
            # using a dependency manager/registry
            pass

    async def simulate(self, args: dict[str, Any]) -> Any:
        """Simulate tool execution for dry runs."""
        return {
            "simulated": True,
            "tool": self.metadata.name,
            "args": args,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @abc.abstractmethod
    async def execute(self, args: dict[str, Any]) -> Any:
        """Execute the tool."""
        pass

    async def stream(self, args: dict[str, Any]) -> AsyncGenerator[Any, None]:
        """Stream tool execution results."""
        if not self.metadata.supports_streaming:
            raise NotImplementedError("Streaming not supported by this tool")
        result = await self.execute(args)
        yield result

    async def __call__(self, args: dict[str, Any]) -> Any:
        """Execute the tool with validation and context handling."""
        await self.check_dependencies()
        await self.validate_input(args)

        if self.context.dry_run and self.metadata.supports_dry_run:
            return await self.simulate(args)

        if self.context.timeout:
            async with asyncio.timeout(self.context.timeout):
                result = await self.execute(args)
        else:
            result = await self.execute(args)

        await self.validate_output(result)
        return result
