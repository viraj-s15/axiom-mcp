"""Base classes and types for tool management."""

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field, ValidationError

from ..exceptions import ToolError

logger = logging.getLogger(__name__)


class ToolDependency(BaseModel):
    """Dependency configuration for a tool."""

    name: str = Field(..., description="Name of the dependency")
    version: str | None = Field(None, description="Version constraint")
    optional: bool = Field(default=False, description="Whether dependency is optional")
    source: str = Field(default="pip", description="Source of the dependency")
    command: str | None = Field(
        None, description="Custom install command if not using standard source"
    )

    @property
    def tool_name(self) -> str:
        """Alias for name field to maintain compatibility."""
        return self.name


class ToolValidation(BaseModel):
    """Validation configuration for tool inputs and outputs."""

    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for validating tool inputs",
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for validating tool outputs",
    )
    strict: bool = Field(
        default=True,
        description="Whether to enforce strict schema validation",
    )


class ToolValidationSchema(ToolValidation):
    """Schema-based validation configuration for tool inputs and outputs.

    This class extends ToolValidation to support Python type hints as schema definitions.
    It converts Python types to JSON Schema compatible formats.
    """

    def __init__(
        self,
        *,
        input_schema: dict[str, tuple[type, ...]],
        output_schema: dict[str, tuple[type, ...]],
    ) -> None:
        """Initialize with Python type-based schemas.

        Args:
            input_schema: Dict mapping field names to (type, ...) tuples where ... indicates required
            output_schema: Dict mapping field names to (type, ...) tuples where ... indicates required
        """
        converted_input = self._convert_type_schema(input_schema)
        converted_output = self._convert_type_schema(output_schema)
        super().__init__(input_schema=converted_input, output_schema=converted_output)

    def _convert_type_schema(
        self, schema: dict[str, tuple[type, ...]]
    ) -> dict[str, Any]:
        """Convert Python type schema to JSON Schema format."""
        properties = {}
        required = []

        for field_name, type_info in schema.items():
            field_type = type_info[0]
            is_required = len(type_info) > 1 and type_info[1] is ...

            if is_required:
                required.append(field_name)

            properties[field_name] = self._type_to_json_schema(field_type)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _type_to_json_schema(self, type_hint: type) -> dict[str, Any]:
        """Convert Python type to JSON Schema type definition."""
        match type_hint:
            case type() if type_hint is str:
                return {"type": "string"}
            case type() if type_hint is int:
                return {"type": "integer"}
            case type() if type_hint is float:
                return {"type": "number"}
            case type() if type_hint is bool:
                return {"type": "boolean"}
            case type() if type_hint is list:
                return {"type": "array"}
            case type() if type_hint is dict:
                return {"type": "object"}
            case _:
                return {}  # Default to any type if unknown


class ToolMetadata(BaseModel):
    """Metadata for a tool."""

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "validate_default": True,
        "populate_by_name": True,
    }

    name: str = Field(..., description="Name of the tool")
    version: str = Field(default="1.0.0", description="Tool version")
    description: str = Field(
        default="", description="Description of what the tool does"
    )
    author: Optional[str] = Field(default=None, description="Tool author")
    tags: list[str] = Field(
        default_factory=list, description="Tool tags for categorization"
    )
    dependencies: list[ToolDependency] = Field(
        default_factory=list,
        description="Tool dependencies",
    )
    validation: ToolValidation | None = Field(
        default=None,
        description="Schema validation configuration",
    )
    supports_streaming: bool = Field(
        default=False,
        description="Whether tool supports streaming output",
    )


class ToolContext(BaseModel):
    """Execution context for tools."""

    model_config = {"arbitrary_types_allowed": True}

    dry_run: bool = False
    cache_enabled: bool = True
    timeout: float | None = None
    validation_enabled: bool = True
    state: dict[str, Any] = Field(default_factory=dict)
    logger: logging.Logger = Field(default_factory=lambda: logging.getLogger(__name__))

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)


class Tool:
    """Base class for all tools.

    Tools are the primary way to extend MCP functionality. Each tool should:
    1. Inherit from this base class
    2. Define metadata using the ToolMetadata class
    3. Implement the execute method

    Example:
        class MyTool(Tool):
            metadata = ToolMetadata(
                name="my_tool",
                description="Does something useful",
                version="1.0.0",
                dependencies=[
                    ToolDependency(name="requests", version=">=2.0.0")
                ],
                validation=ToolValidation(
                    input_schema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "format": "uri"}
                        },
                        "required": ["url"]
                    }
                )
            )

            async def execute(self, args: dict[str, Any]) -> Any:
                url = args["url"]
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        return await response.text()
    """

    metadata: ClassVar[ToolMetadata]
    _dependencies_checked: bool = False
    _validation_cache: dict[str, Any] = {}

    def __init__(self, *, context: ToolContext | None = None) -> None:
        """Initialize tool with optional context."""
        self.context = context or ToolContext()
        self._ensure_metadata()

    def _ensure_metadata(self) -> None:
        """Ensure tool has required metadata."""
        if not hasattr(self, "metadata"):
            raise ToolError("Tool missing metadata")

        if not isinstance(self.metadata, ToolMetadata):
            raise ToolError("Tool metadata must be instance of ToolMetadata")

    async def check_dependencies(self) -> None:
        """Check if all required dependencies are available."""
        if self._dependencies_checked:
            return

        missing = []
        for dep in self.metadata.dependencies:
            if not await self._check_dependency(dep) and not dep.optional:
                missing.append(f"{dep.name}{dep.version or ''}")

        if missing:
            raise ToolError(
                f"Missing required dependencies: {', '.join(missing)}",
                tool_name=self.metadata.name,
            )

        self._dependencies_checked = True

    async def _check_dependency(self, dep: ToolDependency) -> bool:
        """Check if a specific dependency is available."""
        try:
            if dep.source == "pip":
                # Try importing the module
                __import__(dep.name)
            else:
                # For non-pip dependencies, we assume they're available
                # Could be extended to check binaries, etc.
                pass
            return True
        except ImportError:
            return False

    def validate_input(self, args: dict[str, Any]) -> None:
        """Validate tool input arguments against schema."""
        if not self.metadata.validation or not self.metadata.validation.input_schema:
            return

        try:
            # Cache the validation model for performance
            model = self._get_validation_model(
                "input", self.metadata.validation.input_schema
            )
            model(**args)
        except ValidationError as e:
            if self.metadata.validation.strict:
                raise ToolError(
                    f"Invalid input for tool {self.metadata.name}: {str(e)}",
                    tool_name=self.metadata.name,
                ) from e
            else:
                # Log warning but continue if not strict
                self.context.warning(f"Input validation warning: {str(e)}")

    def validate_output(self, result: Any) -> None:
        """Validate tool output against schema."""
        if not self.metadata.validation or not self.metadata.validation.output_schema:
            return

        try:
            model = self._get_validation_model(
                "output", self.metadata.validation.output_schema
            )
            if isinstance(result, (list, tuple)):
                for item in result:
                    model(**item if isinstance(item, dict) else {"value": item})
            else:
                model(**result if isinstance(result, dict) else {"value": result})
        except ValidationError as e:
            if self.metadata.validation.strict:
                raise ToolError(
                    f"Invalid output from tool {self.metadata.name}: {str(e)}",
                    tool_name=self.metadata.name,
                ) from e
            else:
                self.context.warning(f"Output validation warning: {str(e)}")

    def _get_validation_model(
        self, key: str, schema: dict[str, Any]
    ) -> type[BaseModel]:
        """Get or create a Pydantic model for validation."""
        cache_key = f"{self.metadata.name}:{key}"
        if cache_key not in self._validation_cache:
            self._validation_cache[cache_key] = type(
                f"{self.metadata.name.title()}{key.title()}Model",
                (BaseModel,),
                {"__annotations__": schema.get("properties", {})},
            )
        return self._validation_cache[cache_key]

    async def execute(self, args: dict[str, Any]) -> Any:
        """Execute the tool with the given arguments.

        This method should be implemented by subclasses.

        Args:
            args: Tool arguments matching the input schema

        Returns:
            Tool execution result
        """
        raise NotImplementedError("Tool must implement execute method")

    async def stream(self, args: dict[str, Any]) -> AsyncGenerator[Any, None]:
        """Stream tool results.

        Override this method to support streaming output.

        Args:
            args: Tool arguments matching the input schema

        Yields:
            Tool execution results as they become available
        """
        if not self.metadata.supports_streaming:
            raise ToolError(
                f"Tool {self.metadata.name} does not support streaming",
                tool_name=self.metadata.name,
            )
        result = await self.execute(args)
        yield result

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        version: str = "1.0.0",
        dependencies: list[ToolDependency] | None = None,
        validation: ToolValidation | None = None,
    ) -> type["Tool"]:
        """Create a Tool class from a function.

        This is a convenience method for creating tools from simple functions.
        It will:
        1. Create a new Tool subclass
        2. Set up metadata from the function's signature and docstring
        3. Create an execute method that calls the function

        Args:
            fn: Function to convert to a tool
            name: Optional name (defaults to function name)
            description: Optional description (defaults to function docstring)
            version: Tool version
            dependencies: Optional list of dependencies
            validation: Optional validation configuration

        Returns:
            A new Tool subclass
        """
        if name is None:
            name = fn.__name__

        if description is None:
            description = fn.__doc__ or ""

        # Create input schema from function signature
        sig = inspect.signature(fn)
        type_hints = inspect.get_annotations(fn, eval_str=True)

        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            param_type = type_hints.get(param_name, Any)
            type_info = cls._type_to_schema(param_type)

            properties[param_name] = type_info
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        input_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        # Create output schema from return annotation
        return_type = type_hints.get("return", Any)
        output_schema = cls._type_to_schema(return_type)

        # Create validation if not provided
        if validation is None:
            validation = ToolValidation(
                input_schema=input_schema,
                output_schema=output_schema,
            )

        # Create the new tool class
        class FunctionTool(cls):
            metadata = ToolMetadata(
                name=name,
                description=description,
                version=version,
                dependencies=dependencies or [],
                validation=validation,
                author=fn.__module__,
            )

            async def execute(self, args: dict[str, Any]) -> Any:
                # Convert args to match function signature
                kwargs = {name: args[name] for name in sig.parameters if name in args}

                # Call function and handle async/sync
                if asyncio.iscoroutinefunction(fn):
                    return await fn(**kwargs)
                return fn(**kwargs)

        return FunctionTool

    @staticmethod
    def _type_to_schema(type_hint: Any) -> dict[str, Any]:
        """Convert a type hint to JSON Schema."""
        # Handle basic types
        if isinstance(type_hint, str):
            return {"type": "string"}
        if isinstance(type_hint, int):
            return {"type": "integer"}
        if isinstance(type_hint, float):
            return {"type": "number"}
        if isinstance(type_hint, bool):
            return {"type": "boolean"}
        if isinstance(type_hint, list):
            return {"type": "array"}
        if isinstance(type_hint, dict):
            return {"type": "object"}

        # Handle optional types
        origin = getattr(type_hint, "__origin__", None)
        if origin is not None:
            if isinstance(origin, list):
                item_type = type_hint.__args__[0]
                return {
                    "type": "array",
                    "items": Tool._type_to_schema(item_type),
                }
            if isinstance(origin, dict):
                key_type, value_type = type_hint.__args__
                if isinstance(key_type, str):
                    return {
                        "type": "object",
                        "additionalProperties": Tool._type_to_schema(value_type),
                    }

        # Default to any
        return {}
