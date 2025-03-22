"""FastMCP - A more ergonomic interface for MCP servers."""

from axiom_mcp.prompts.base import Prompt, Message
from axiom_mcp.resources.base import Resource
from axiom_mcp.tools.base import Tool, ToolContext, ToolMetadata
from mcp.types import (
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    TextContent,
    Tool as MCPBaseTool,
    Resource as MCPBaseResource,
    Prompt as MCPBasePrompt,
)
from mcp.shared.context import RequestContext
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.server import Server as MCPServer
import uvicorn
from pydantic import BaseModel, Field
import pydantic_core
from typing import Any, Callable, Dict, Literal, Sequence, TypeVar, ParamSpec, TYPE_CHECKING, Protocol, cast, Awaitable
from itertools import chain
import re
import json
import inspect
import functools
import asyncio
from pydantic.networks import AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from axiom_mcp.exceptions import ResourceError
from axiom_mcp.prompts import Prompt, PromptManager, PromptResponse
from axiom_mcp.resources import FunctionResource, Resource, ResourceManager
from axiom_mcp.tools import ToolManager
from axiom_mcp.utilities.logging import configure_logging, get_logger
from axiom_mcp.utilities.types import Image
from pydantic import BaseModel
from typing import cast, Any, Callable, Dict, Literal, Sequence, TypeVar, ParamSpec, TYPE_CHECKING, Protocol, Awaitable
from typing_extensions import TypeAlias

# Define Unknown type correctly
T = TypeVar('T')
Unknown = T


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str
    description: str | None = None
    inputSchema: dict[str, Any] | None = None


class MCPResource(BaseModel):
    """MCP resource definition."""
    uri: str
    name: str = ""
    description: str | None = None
    mimeType: str | None = None


class MCPResourceTemplate(BaseModel):
    """MCP resource template definition."""
    uriTemplate: str
    name: str = ""
    description: str | None = None


class MCPPromptArgument(BaseModel):
    """MCP prompt argument definition."""
    name: str
    description: str | None = None
    required: bool = False


class MCPPrompt(BaseModel):
    """MCP prompt definition."""
    name: str
    description: str | None = None
    arguments: list[MCPPromptArgument] | None = None


class ServerSession(Protocol):
    """Protocol for server session."""

    async def send_progress_notification(
        self, progress_token: str, progress: float, total: float | None = None
    ) -> None: ...

    def send_log_message(
        self, level: str, data: str, logger: str | None = None
    ) -> None: ...


if TYPE_CHECKING:
    from axiom_mcp.axiom_mcp import AxiomMCP

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
R_PromptResponse = TypeVar("R_PromptResponse", bound=PromptResponse)


class Settings(BaseSettings):
    """FastMCP server settings."""

    # Server settings
    debug: bool = False
    log_level: Literal["DEBUG", "INFO",
                       "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # Manager settings
    warn_on_duplicate_tools: bool = True
    warn_on_duplicate_resources: bool = True
    warn_on_duplicate_prompts: bool = True

    # Dependencies
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of dependencies to install in the server environment",
    )

    model_config = SettingsConfigDict(
        env_prefix="FASTMCP_",
        env_file=".env",
        extra="ignore",
    )


class AxiomMCP:
    def __init__(self, name: str | None = None, **settings: Any):
        self.settings = Settings(**settings)
        self._mcp_server = MCPServer(name=name or "FastMCP")
        self._tool_manager = ToolManager(
            cache_size=1000,
            default_timeout=30.0,
            enable_metrics=True
        )
        self._resource_manager = ResourceManager(
            warn_on_duplicate_resources=self.settings.warn_on_duplicate_resources
        )
        self._prompt_manager = PromptManager()

        self.dependencies = self.settings.dependencies

        # Set up MCP protocol handlers
        self._setup_handlers()

        # Configure logging
        configure_logging(self.settings.log_level)

    @property
    def name(self) -> str:
        return self._mcp_server.name

    def run(self, transport: Literal["stdio", "sse"] = "stdio") -> None:
        """Run the FastMCP server. Note this is a synchronous function.

        Args:
            transport: Transport protocol to use ("stdio" or "sse")
        """
        TRANSPORTS = Literal["stdio", "sse"]
        if transport not in TRANSPORTS.__args__:  # type: ignore
            raise ValueError(f"Unknown transport: {transport}")

        if transport == "stdio":
            asyncio.run(self.run_stdio_async())
        else:  # transport == "sse"
            asyncio.run(self.run_sse_async())

    def _setup_handlers(self) -> None:
        """Set up core MCP protocol handlers."""
        # Use type casting to handle type compatibility
        self._mcp_server.list_tools()(
            cast(Callable[[], Awaitable[list[MCPBaseTool]]], self.list_tools)
        )
        self._mcp_server.call_tool()(self.call_tool)
        self._mcp_server.list_resources()(
            cast(Callable[[], Awaitable[list[MCPBaseResource]]],
                 self.list_resources)
        )
        self._mcp_server.read_resource()(self.read_resource)
        self._mcp_server.list_prompts()(
            cast(Callable[[], Awaitable[list[MCPBasePrompt]]],
                 self.list_prompts)
        )
        self._mcp_server.get_prompt()(self.get_prompt)

    async def list_tools(self) -> list[MCPTool]:
        """List all available tools."""
        tools = self._tool_manager._tools.values()  # Access registered tools directly
        return [
            MCPTool(
                name=tool_cls.metadata.name,
                description=tool_cls.metadata.description,
                inputSchema=tool_cls.metadata.validation.input_schema if tool_cls.metadata.validation else None,
            )
            for tool_cls in tools
        ]

    def get_context(self) -> "Context":
        """
        Returns a Context object. Note that the context will only be valid
        during a request; outside a request, most methods will error.
        """
        try:
            request_context = self._mcp_server.request_context
        except LookupError:
            request_context = None

        return Context(
            request_context=cast(RequestContext, request_context),
            resource_manager=self._resource_manager,
        )

    async def call_tool(
        self, name: str, arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Call a tool by name with arguments."""
        context = self.get_context()
        tool_context = ToolContext(
            dry_run=False,
            cache_enabled=True,
            validation_enabled=True
        )
        result = await self._tool_manager.execute_tool(name, arguments, context=tool_context)
        converted_result = _convert_to_content(result)
        return converted_result

    async def list_resources(self) -> list[MCPResource]:
        """List all available resources."""

        resources = self._resource_manager.list_resources()
        return [
            MCPResource(
                uri=str(resource.uri),  # Convert AnyUrl to str
                name=resource.name or "",
                description=resource.description,
                mimeType=resource.mime_type,
            )
            for resource in resources
        ]

    async def list_resource_templates(self) -> list[MCPResourceTemplate]:
        """List all available resource templates."""
        templates = self._resource_manager.list_templates()
        return [
            MCPResourceTemplate(
                # Convert AnyUrl to str for template URI
                uriTemplate=str(template.uri),
                name=template.name or "",
                description=template.description or "",
            )
            for template in templates
        ]

    async def read_resource(self, uri: AnyUrl | str) -> str | bytes:
        """Read a resource by URI."""
        resource = await self._resource_manager.get_resource(str(uri))  # Convert AnyUrl to str
        if not resource:
            raise ResourceError(f"Unknown resource: {uri}")

        try:
            return await resource.read()
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            raise ResourceError(str(e))

    def add_tool(
        self,
        fn: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Add a tool to the server."""
        # Create a Tool class from the function
        tool_name = name or fn.__name__

        class FunctionTool(Tool):
            metadata = ToolMetadata(
                name=tool_name,
                description=description or fn.__doc__ or "",
                version="1.0.0"
            )

            async def execute(self, args: dict[str, Any]) -> Any:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(**args)
                return fn(**args)

        self._tool_manager.register_tool(FunctionTool)

    def tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to register a tool.

        Tools can optionally request a Context object by adding a parameter with the Context type annotation.
        The context provides access to MCP capabilities like logging, progress reporting, and resource access.

        Args:
            name: Optional name for the tool (defaults to function name)
            description: Optional description of what the tool does

        Example:
            @server.tool()
            def my_tool(x: int) -> str:
                return str(x)

            @server.tool()
            def tool_with_context(x: int, ctx: Context) -> str:
                ctx.info(f"Processing {x}")
                return str(x)

            @server.tool()
            async def async_tool(x: int, context: Context) -> str:
                await context.report_progress(50, 100)
                return str(x)
        """
        # Check if user passed function directly instead of calling decorator
        if callable(name):
            raise TypeError(
                "The @tool decorator was used incorrectly. "
                "Did you forget to call it? Use @tool() instead of @tool"
            )

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            self.add_tool(fn, name=name, description=description)
            return fn

        return decorator

    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the server.

        Args:
            resource: A Resource instance to add
        """
        asyncio.create_task(self._resource_manager.add_resource(
            resource))  # Make it async

    def resource(
        self,
        uri: str,
        *,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to register a function as a resource."""
        if callable(uri):
            raise TypeError(
                "The @resource decorator was used incorrectly. "
                "Did you forget to call it? Use @resource() instead of @resource"
            )

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            # Keep track of the original function
            orig_fn = fn

            # Wrap the function to handle async/sync compatibility
            @functools.wraps(fn)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                return fn(*args, **kwargs)

            # Check if URI contains parameters
            uri_has_params = "{" in uri and "}" in uri
            func_params = list(inspect.signature(fn).parameters.keys())

            if uri_has_params or func_params:
                logger.debug(
                    "Registering template resource with parameters: %s "
                    f"and function parameters {func_params}",
                    uri,
                )

                # Register as template - create task to handle the async call
                asyncio.create_task(self._resource_manager.add_template(
                    wrapper,
                    uri_template=uri,
                    name=name,
                    description=description,
                    mime_type=mime_type or "text/plain",
                ))
            else:
                # Register as regular resource
                resource = FunctionResource(
                    uri=AnyUrl(uri),
                    name=name,
                    description=description,
                    mime_type=mime_type or "text/plain",
                    fn=wrapper,
                )
                self.add_resource(resource)

            # Return the original function
            return orig_fn

        return decorator

    def add_prompt(self, prompt: Prompt) -> None:
        """Add a prompt to the server.

        Args:
            prompt: A Prompt instance to add
        """
        self._prompt_manager.add_prompt(prompt)

    def prompt(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[P, R_PromptResponse]], Callable[P, R_PromptResponse]]:
        """Decorator to register a prompt.

        Args:
            name: Optional name for the prompt (defaults to function name)
            description: Optional description of what the prompt does

        Example:
            @server.prompt()
            def analyze_table(table_name: str) -> list[Message]:
                schema = read_table_schema(table_name)
                return [
                    {
                        "role": "user",
                        "content": f"Analyze this schema:\n{schema}"
                    }
                ]

            @server.prompt()
            async def analyze_file(path: str) -> list[Message]:
                content = await read_file(path)
                return [
                    {
                        "role": "user",
                        "content": {
                            "type": "resource",
                            "resource": {
                                "uri": f"file://{path}",
                                "text": content
                            }
                        }
                    }
                ]
        """
        # Check if user passed function directly instead of calling decorator
        if callable(name):
            raise TypeError(
                "The @prompt decorator was used incorrectly. "
                "Did you forget to call it? Use @prompt() instead of @prompt"
            )

        def decorator(func: Callable[P, R_PromptResponse]) -> Callable[P, R_PromptResponse]:
            # Wrap the function to handle async/sync compatibility
            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R_PromptResponse:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)

            prompt = Prompt.from_function(
                wrapper, name=name, description=description)
            self.add_prompt(prompt)
            return func

        return decorator

    async def run_stdio_async(self) -> None:
        """Run the server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self._mcp_server.run(
                read_stream,
                write_stream,
                self._mcp_server.create_initialization_options(),
            )

    async def run_sse_async(self) -> None:
        """Run the server using SSE transport."""
        from starlette.applications import Starlette
        from starlette.routing import Route

        sse = SseServerTransport("/messages")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self._mcp_server.run(
                    streams[0],
                    streams[1],
                    self._mcp_server.create_initialization_options(),
                )

        async def handle_messages(request):
            await sse.handle_post_message(request.scope, request.receive, request._send)

        starlette_app = Starlette(
            debug=self.settings.debug,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Route("/messages", endpoint=handle_messages, methods=["POST"]),
            ],
        )

        config = uvicorn.Config(
            starlette_app,
            host=self.settings.host,
            port=self.settings.port,
            log_level=self.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def list_prompts(self) -> list[MCPPrompt]:
        """List all available prompts."""
        prompts = self._prompt_manager.list_prompts()
        return [
            MCPPrompt(
                name=prompt.name,
                description=prompt.description,
                arguments=[
                    MCPPromptArgument(
                        name=arg.name,
                        description=arg.description,
                        required=arg.required,
                    )
                    for arg in (prompt.arguments or [])
                ],
            )
            for prompt in prompts
        ]

    async def get_prompt(
        self, name: str, arguments: Dict[str, Any] | None = None
    ) -> GetPromptResult:
        """Get a prompt by name with arguments."""
        try:
            messages = await self._prompt_manager.render_prompt(name, arguments)

            return GetPromptResult(messages=pydantic_core.to_jsonable_python(messages))
        except Exception as e:
            logger.error(f"Error getting prompt {name}: {e}")
            raise ValueError(str(e))


def _convert_to_content(
    result: Any,
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Convert a result to a sequence of content objects."""
    if result is None:
        return []

    if isinstance(result, (TextContent, ImageContent, EmbeddedResource)):
        return [result]

    if isinstance(result, Image):
        return [result.to_image_content()]

    if isinstance(result, (list, tuple)):
        return list(chain.from_iterable(_convert_to_content(item) for item in result))

    if not isinstance(result, str):
        try:
            result = json.dumps(pydantic_core.to_jsonable_python(result))
        except Exception:
            result = str(result)

    return [TextContent(type="text", text=result)]


class Context(BaseModel):
    """Context object providing access to MCP capabilities.

    This provides a cleaner interface to MCP's RequestContext functionality.
    It gets injected into tool and resource functions that request it via type hints.

    To use context in a tool function, add a parameter with the Context type annotation:

    ```python
    @server.tool()
    def my_tool(x: int, ctx: Context) -> str:
        # Log messages to the client
        ctx.info(f"Processing {x}")
        ctx.debug("Debug info")
        ctx.warning("Warning message")
        ctx.error("Error message")

        # Report progress
        ctx.report_progress(50, 100)

        # Access resources
        data = ctx.read_resource("resource://data")

        # Get request info
        request_id = ctx.request_id
        client_id = ctx.client_id

        return str(x)
    ```

    The context parameter name can be anything as long as it's annotated with Context.
    The context is optional - tools that don't need it can omit the parameter.
    """

    _request_context: RequestContext | None
    _resource_manager: ResourceManager

    def __init__(
        self,
        *,
        request_context: RequestContext | None = None,
        resource_manager: ResourceManager,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._request_context = request_context
        self._resource_manager = resource_manager

    @property
    def request_context(self) -> RequestContext:
        """Access to the underlying request context."""
        if self._request_context is None:
            raise ValueError("Context is not available outside of a request")
        return self._request_context

    async def report_progress(
        self, progress: float, total: float | None = None
    ) -> None:
        """Report progress for the current operation.

        Args:
            progress: Current progress value e.g. 24
            total: Optional total value e.g. 100
        """

        progress_token = (
            self.request_context.meta.progressToken
            if self.request_context.meta
            else None
        )

        if not progress_token:
            return

        await self.request_context.session.send_progress_notification(
            progress_token=progress_token, progress=progress, total=total
        )

    async def read_resource(self, uri: str) -> str | bytes:
        """Read a resource by URI.

        Args:
            uri: Resource URI to read

        Returns:
            The resource content as either text or bytes
        """
        return await self._resource_manager.read_resource(uri)

    def log(
        self,
        level: Literal["debug", "info", "warning", "error"],
        message: str,
        *,
        logger_name: str | None = None,
    ) -> None:
        """Send a log message to the client.

        Args:
            level: Log level (debug, info, warning, error)
            message: Log message
            logger_name: Optional logger name
            **extra: Additional structured data to include
        """
        self.request_context.session.send_log_message(
            level=level, data=message, logger=logger_name
        )

    @property
    def client_id(self) -> str | None:
        """Get the client ID if available."""
        return (
            getattr(self.request_context.meta, "client_id", None)
            if self.request_context.meta
            else None
        )

    @property
    def request_id(self) -> str:
        """Get the unique ID for this request."""
        return str(self.request_context.request_id)

    @property
    def session(self):
        """Access to the underlying session for advanced usage."""
        return self.request_context.session

    # Convenience methods for common log levels
    def debug(self, message: str, **extra: Any) -> None:
        """Send a debug log message."""
        self.log("debug", message, **extra)

    def info(self, message: str, **extra: Any) -> None:
        """Send an info log message."""
        self.log("info", message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        """Send a warning log message."""
        self.log("warning", message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        """Send an error log message."""
        self.log("error", message, **extra)
