"""FastMCP - A more ergonomic interface for MCP servers."""

from axiom_mcp.resources.advanced_types import TemplateResource
from mcp.types import (
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    TextContent,
    Tool as MCPTool,
    Resource as MCPResource,
    Prompt as MCPPrompt,
    PromptArgument as MCPPromptArgument,
    ResourceTemplate as MCPResourceTemplate,
)
from mcp.shared.context import RequestContext
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.server import Server as MCPServer
import uvicorn
from pydantic import BaseModel, Field
import pydantic_core
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Sequence,
    TypeVar,
    ParamSpec,
    TYPE_CHECKING,
    Protocol,
    cast,
    Awaitable,
)
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
from axiom_mcp.resources.base import ResourceType
from axiom_mcp.tools import ToolManager
from axiom_mcp.tools.base import Tool, ToolContext, ToolMetadata
from axiom_mcp.utilities.logging import configure_logging, get_logger
from axiom_mcp.utilities.types import Image
from starlette.middleware.cors import CORSMiddleware  # Add CORS middleware import

# Define Unknown type correctly
T = TypeVar("T")
Unknown = T


class ServerSession(Protocol):
    """Protocol for server session."""

    async def send_progress_notification(
        self, progress_token: str, progress: float, total: float | None = None
    ) -> None: ...

    def send_log_message(
        self, level: str, data: str, logger: str | None = None
    ) -> None: ...


if TYPE_CHECKING:
    from axiom_mcp.server import AxiomMCP

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
R_PromptResponse = TypeVar("R_PromptResponse", bound=PromptResponse)


class Settings(BaseSettings):
    """FastMCP server settings."""

    # Server settings
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # Production settings
    production_mode: bool = False
    optimize_memory: bool = False
    gc_interval: int = 300  # Garbage collection interval in seconds
    max_cached_items: int = 1000
    request_timeout: int = 30

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
    # Track instances by name to ensure singletons
    _instances: dict[str, "AxiomMCP"] = {}

    def __new__(cls, name: str | None = None, **settings: Any):
        """Create or retrieve an existing instance."""
        init_name = name or "AxiomMCP"
        if init_name in cls._instances:
            # Return existing instance if it exists
            return cls._instances[init_name]

        # Create new instance if it doesn't exist
        instance = super().__new__(cls)
        cls._instances[init_name] = instance
        return instance

    def __init__(self, name: str | None = None, **settings: Any):
        init_name = name or "AxiomMCP"

        # Skip initialization if this instance is already initialized
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.settings = Settings(**settings)
        # Create server with name first before trying to access it
        self._mcp_server = MCPServer(name=init_name)
        self._tool_manager = ToolManager(
            cache_size=1000, default_timeout=30.0, enable_metrics=True
        )
        self._resource_manager = ResourceManager(
            warn_on_duplicate_resources=self.settings.warn_on_duplicate_resources
        )
        self._prompt_manager = PromptManager(
            warn_on_duplicate=self.settings.warn_on_duplicate_prompts
        )

        self.dependencies = self.settings.dependencies

        self._pending_tools = []  # Store pending tools
        self._pending_resources = []
        self._pending_prompts = []

        # Set up MCP protocol handlers
        self._setup_handlers()

        # Configure logging
        configure_logging(self.settings.log_level)

        self._initialized = True

        # Add signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        import signal

        def handle_shutdown(signum: int, frame: Any) -> None:
            """Handle shutdown signals."""
            logger.info("Received shutdown signal, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())

        # Register handlers for common shutdown signals
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

    async def shutdown(self) -> None:
        """Perform graceful shutdown of all components."""
        logger.info("Initiating graceful shutdown...")

        try:
            # Clean up managers in parallel
            self._tool_manager.clear_cache()

            # Wait for any pending operations to complete
            await asyncio.sleep(0.5)  # Short grace period

            logger.info("Shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise
        finally:
            # Force exit if something is hanging
            import sys

            sys.exit(0)

    @property
    def name(self) -> str:
        """Get the server name."""
        return self._mcp_server.name

    def add_tool(
        self,
        fn: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Add a tool to the server."""
        # Store the tool for later registration
        self._pending_tools.append((fn, name, description))

    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the server."""
        # Store the resource for later registration
        self._pending_resources.append(resource)

    def add_prompt(self, prompt: Prompt) -> None:
        """Add a prompt to the server."""
        # Store the prompt for later registration
        self._pending_prompts.append(prompt)

    def tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to register a tool."""
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
            func_params = inspect.signature(fn).parameters

            if uri_has_params:
                # Validate that URI params match function params
                uri_params = set(re.findall(r"{(\w+)}", uri))
                param_names = set(func_params.keys())

                if uri_params != param_names:
                    raise ValueError(
                        f"Mismatch between URI parameters {uri_params} "
                        f"and function parameters {param_names}"
                    )

                # Register as template - add to pending resources for async registration
                self._pending_resources.append(
                    (wrapper, uri, name, description, mime_type)
                )
            else:
                # Register as regular resource - create a FunctionResource instance
                resource = FunctionResource(
                    uri=AnyUrl(uri),
                    name=name,
                    description=description,
                    mime_type=mime_type or "text/plain",
                    resource_type=ResourceType.FUNCTION,
                    fn=wrapper,
                )
                # Add to pending resources for async registration
                self._pending_resources.append(resource)

            # Return the original function
            return orig_fn

        return decorator

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

        def decorator(
            func: Callable[P, R_PromptResponse],
        ) -> Callable[P, R_PromptResponse]:
            # Wrap the function to handle async/sync compatibility
            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R_PromptResponse:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)

            prompt = Prompt.from_function(wrapper, name=name, description=description)
            self.add_prompt(prompt)
            return func

        return decorator

    def _setup_handlers(self) -> None:
        """Set up core MCP protocol handlers."""
        # Use type casting to handle type compatibility
        self._mcp_server.list_tools()(
            cast(Callable[[], Awaitable[list[MCPTool]]], self.list_tools)
        )
        self._mcp_server.call_tool()(self.call_tool)
        self._mcp_server.list_resources()(
            cast(Callable[[], Awaitable[list[MCPResource]]], self.list_resources)
        )
        self._mcp_server.list_resource_templates()(
            cast(
                Callable[[], Awaitable[list[MCPResourceTemplate]]],
                self.list_resource_templates,
            )
        )
        self._mcp_server.read_resource()(self.read_resource)
        self._mcp_server.list_prompts()(
            cast(Callable[[], Awaitable[list[MCPPrompt]]], self.list_prompts)
        )
        self._mcp_server.get_prompt()(self.get_prompt)

    def _process_pending_tools(self) -> None:
        """Process and register pending tools."""
        for fn, name, description in self._pending_tools:
            tool_name = name or fn.__name__

            class FunctionTool(Tool):
                metadata = ToolMetadata(
                    name=tool_name,
                    description=description or fn.__doc__ or "",
                    version="1.0.0",
                    author=None,  # Optional parameter
                    tags=[],  # Empty list by default
                    dependencies=[],  # Empty list by default
                    validation=None,  # Optional parameter
                )

                async def execute(self, args: dict[str, Any]) -> Any:
                    if asyncio.iscoroutinefunction(fn):
                        return await fn(**args)
                    return fn(**args)

            self._tool_manager.register_tool(FunctionTool)

    async def _process_pending_resources(self) -> None:
        """Process and register pending resources."""
        for item in self._pending_resources:
            if isinstance(item, Resource):
                await self._resource_manager.add_resource(item)
            else:
                # Unpack the tuple for template resources
                wrapper, uri, name, description, mime_type = item

                # Check if this is a template resource
                if "{" in uri and "}" in uri:
                    # Create template resource
                    template = TemplateResource.from_function(
                        fn=wrapper,
                        uri_template=uri,
                        name=name,
                        description=description,
                        mime_type=mime_type or "text/plain",
                    )
                    await self._resource_manager.add_resource(template)
                else:
                    # Regular function resource
                    resource = FunctionResource(
                        uri=AnyUrl(uri),
                        name=name,
                        description=description,
                        mime_type=mime_type or "text/plain",
                        resource_type=ResourceType.FUNCTION,
                        fn=wrapper,
                    )
                    await self._resource_manager.add_resource(resource)

        # Clear the pending resources after processing
        self._pending_resources.clear()

    def _process_pending_prompts(self) -> None:
        """Process and register pending prompts."""
        for prompt in self._pending_prompts:
            self._prompt_manager.add_prompt(prompt)

    async def run(self, transport: Literal["stdio", "sse"] = "sse") -> None:
        """Run the AxiomMCP server with specified transport."""
        try:
            # Process all pending items before starting
            self._process_pending_tools()
            await self._process_pending_resources()
            self._process_pending_prompts()

            if transport == "stdio":
                await self.run_stdio_async()
            else:
                await self.run_sse_async()
        except Exception:
            logger.exception("Error running server")
            await self.shutdown()
            raise
        finally:
            # Ensure cleanup happens even on unexpected exits
            await self.shutdown()

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
        from starlette.routing import Route, Mount

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self._mcp_server.run(
                    streams[0],
                    streams[1],
                    self._mcp_server.create_initialization_options(),
                )

        app = Starlette(
            debug=self.settings.debug,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        config = uvicorn.Config(
            app,
            host=self.settings.host,
            port=self.settings.port,
            log_level=self.settings.log_level.lower(),
            workers=1,
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def list_tools(self) -> list[MCPTool]:
        """List all available tools."""
        # Process any pending tools first
        self._process_pending_tools()

        tools = []
        for tool_cls in self._tool_manager._tools.values():
            tool = MCPTool(
                name=tool_cls.metadata.name,
                description=tool_cls.metadata.description,
                inputSchema=(
                    tool_cls.metadata.validation.input_schema
                    if tool_cls.metadata.validation
                    else {}
                ),
            )
            tools.append(tool)  # Actually append the tool to the list

        return tools  # Return the populated list

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
        tool_context = ToolContext(
            dry_run=False, cache_enabled=True, validation_enabled=True
        )
        result = await self._tool_manager.execute_tool(
            name, arguments, context=tool_context
        )

        # If result is already in the correct MCP content type format, return as sequence
        if isinstance(result, (TextContent, ImageContent, EmbeddedResource)):
            return [result]

        # If result is a dict with "type" and "content", convert to proper MCP content type
        if isinstance(result, dict) and "type" in result:
            if result["type"] == "text":
                content = result.get("content")
                if isinstance(content, str):
                    return [TextContent(type="text", text=content)]
                else:
                    # For structured content, convert to JSON string
                    return [TextContent(type="text", text=json.dumps(content))]
            elif result["type"] == "image":
                return [ImageContent(**result)]
            elif result["type"] == "resource":
                return [EmbeddedResource(**result)]

        return _convert_to_content(result)

    async def list_resources(self) -> list[MCPResource]:
        """List all available resources."""
        # Process any pending resources first
        await self._process_pending_resources()

        resources = []
        for resource in self._resource_manager.list_resources():
            mcp_resource = MCPResource(
                uri=AnyUrl(str(resource.uri)),
                name=resource.name or "",
                description=resource.description,
                mimeType=resource.mime_type,
            )
            resources.append(mcp_resource)
        return resources

    async def list_resource_templates(self) -> list[MCPResourceTemplate]:
        """List all available resource templates."""
        # Process any pending resources first
        await self._process_pending_resources()

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
        resource = await self._resource_manager.get_resource(
            str(uri)
        )  # Convert AnyUrl to str
        if not resource:
            raise ResourceError(f"Unknown resource: {uri}")

        try:
            return await resource.read()
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            raise ResourceError(str(e))

    async def list_prompts(self) -> list[MCPPrompt]:
        """List all available prompts."""
        # Process any pending prompts first
        self._process_pending_prompts()

        prompts = []
        for prompt in self._prompt_manager.list_prompts():
            mcp_prompt = MCPPrompt(
                name=prompt.name,
                description=prompt.description,
                arguments=(
                    [
                        MCPPromptArgument(
                            name=arg.name,
                            description=arg.description,
                            required=arg.required,
                        )
                        for arg in prompt.arguments
                    ]
                    if prompt.arguments
                    else None
                ),
            )
            prompts.append(mcp_prompt)
        return prompts

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
