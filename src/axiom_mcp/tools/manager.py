"""Tool management and execution."""

import asyncio
import cachetools
import hashlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from jsonschema import validate as validate_schema

from ..exceptions import ToolError
from .base import Tool, ToolContext

logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger("axiom_mcp.tools.metrics")
metrics_logger.setLevel(logging.INFO)


class ToolCacheEntry(BaseModel):
    """Cache entry for tool results."""

    result: Any
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_version: str
    context_hash: str


class ToolMetrics(BaseModel):
    """Metrics for tool usage."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_execution_time: float = 0.0
    last_used: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolManager:
    """Manager for tool registration, execution and caching."""

    def __init__(
        self,
        cache_size: int = 1000,
        default_timeout: float = 30.0,
        enable_metrics: bool = True,
        warn_on_duplicate_tools: bool = True,
        metrics_dir: str | Path | None = None,
    ):
        """Initialize the tool manager.

        Args:
            cache_size: Maximum number of results to cache
            default_timeout: Default timeout for tool execution in seconds
            enable_metrics: Whether to collect execution metrics
            warn_on_duplicate_tools: Whether to warn on duplicate tool registration
            metrics_dir: Directory to store metrics logs. If None, uses 'logs' directory relative to current working directory
        """
        self._tools: dict[str, type[Tool]] = {}
        self._metrics: dict[str, ToolMetrics] = defaultdict(ToolMetrics)
        self._cache = cachetools.LRUCache(maxsize=cache_size)
        self.default_timeout = default_timeout
        self.enable_metrics = enable_metrics
        self.warn_on_duplicate = warn_on_duplicate_tools
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()

        # Configure metrics logging
        if enable_metrics:
            metrics_path = Path(metrics_dir) if metrics_dir else Path.cwd() / "logs"
            metrics_path.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_path / "tools.log"

            # Remove any existing handlers
            for handler in metrics_logger.handlers[:]:
                metrics_logger.removeHandler(handler)

            # Add new file handler
            metrics_handler = logging.FileHandler(metrics_file)
            metrics_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            metrics_logger.addHandler(metrics_handler)

    async def initialize(self) -> None:
        """Initialize the tool manager."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            # Initialize caches and metrics
            self._cache.clear()
            self._metrics.clear()
            self._initialized = True

    async def _ensure_initialized(self) -> None:
        """Ensure manager is initialized."""
        if not self._initialized:
            await self.initialize()

    def register_tool(self, tool: type[Tool], replace: bool = False) -> None:
        """Register a tool with validation and metrics setup."""
        tool_name = tool.metadata.name

        if tool_name in self._tools and not replace:
            if self.warn_on_duplicate:
                logger.warning(f"Tool {tool_name} already registered")
            return

        # Validate tool implementation
        if not hasattr(tool, "execute"):
            raise ToolError(f"Tool {tool_name} missing execute method")

        # Initialize metrics
        if self.enable_metrics:
            self._metrics[tool_name] = ToolMetrics(total_calls=0)

        self._tools[tool_name] = tool

    async def execute_tool(
        self, name: str, arguments: dict[str, Any], context: ToolContext | None = None
    ) -> Any:
        """Execute a tool with proper error handling and metrics."""
        await self._ensure_initialized()
        tool_cls = self._tools.get(name)
        if not tool_cls:
            raise ToolError(f"Unknown tool: {name}")

        # Use context or create default
        ctx = context or ToolContext()
        timeout = ctx.timeout or self.default_timeout

        # Create tool instance with context
        tool = tool_cls(context=ctx)
        start_time = time.monotonic()

        # Initialize metrics
        metrics = self._metrics[name] if self.enable_metrics else ToolMetrics()

        try:
            if self.enable_metrics:
                metrics.total_calls += 1
                metrics.last_used = datetime.now(UTC)

            # Check cache first
            cache_key = self._get_cache_key(name, arguments)
            if ctx.cache_enabled and cache_key in self._cache:
                if self.enable_metrics:
                    metrics.successful_calls += 1
                return self._cache[cache_key]

            # First try to validate the input if enabled
            if ctx.validation_enabled and tool.metadata.validation:
                try:
                    validate_schema(arguments, tool.metadata.validation.input_schema)
                except Exception as e:
                    if self.enable_metrics:
                        self._log_metrics(name, "validation_error", arguments, 
                                        error=str(e), validation_type="input")
                        metrics.failed_calls += 1
                    raise ToolError(
                        "Invalid tool input: " + str(e),
                        tool_name=name,
                    ) from e  # Add from clause to preserve stack trace

            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    tool.execute(arguments), timeout=timeout
                )

                # Validate output if enabled
                if (
                    ctx.validation_enabled
                    and tool.metadata.validation
                    and tool.metadata.validation.output_schema
                ):
                    try:
                        validate_schema(result, tool.metadata.validation.output_schema)
                    except Exception as e:
                        if self.enable_metrics:
                            self._log_metrics(name, "validation_error", arguments, 
                                            error=str(e), validation_type="output")
                            metrics.failed_calls += 1
                        raise ToolError(
                            "Invalid tool output: " + str(e),
                            tool_name=name,
                        ) from e  # Add from clause to preserve stack trace

                # Cache result if enabled
                if ctx.cache_enabled:
                    self._cache[cache_key] = result

                # Update metrics for success
                if self.enable_metrics:
                    metrics.successful_calls += 1
                    execution_time = time.monotonic() - start_time
                    if metrics.total_calls > 0:
                        metrics.average_execution_time = (
                            metrics.average_execution_time * (metrics.total_calls - 1)
                            + execution_time
                        ) / metrics.total_calls
                    self._log_metrics(
                        name,
                        "success",
                        arguments,
                        execution_time=execution_time,
                    )

                return result

            except asyncio.TimeoutError:
                if self.enable_metrics:
                    self._log_metrics(name, "timeout", arguments, timeout=timeout)
                    metrics.failed_calls += 1
                raise
            except Exception as e:
                if self.enable_metrics:
                    metrics.failed_calls += 1
                    self._log_metrics(name, "error", arguments, error=str(e))
                raise ToolError(
                    f"Tool {name} failed: {str(e)}", tool_name=name, cause=e
                ) from e  # Add from clause to preserve stack trace

        except ToolError:
            # Don't increment failed_calls here as it's already been incremented in the specific error handlers
            raise

    def _log_metrics(
        self, tool_name: str, status: str, arguments: dict, **extra: Any
    ) -> None:
        """Log metrics to file for analysis."""
        if not self.enable_metrics:
            return

        # Ensure arguments are properly serialized for logging
        safe_args = {}
        try:
            # Attempt to create a serializable copy of arguments
            safe_args = json.loads(json.dumps(arguments))
        except (TypeError, ValueError, json.JSONDecodeError):
            # If serialization fails, log what we can
            safe_args = {"error": "Arguments could not be serialized", 
                        "arg_keys": list(arguments.keys() if arguments else [])}

        metrics_data = {
            "tool": tool_name,
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
            "arguments": safe_args,
            **extra,
        }

        # Log metrics in JSON format
        metrics_logger.info(json.dumps(metrics_data))

    def get_metrics_summary(self) -> dict[str, dict]:
        """Get a summary of all tool metrics."""
        if not self.enable_metrics:
            raise ToolError("Metrics are disabled")

        summary = {}
        for tool_name, metrics in self._metrics.items():
            summary[tool_name] = {
                "total_calls": metrics.total_calls,
                "success_rate": (
                    (metrics.successful_calls / metrics.total_calls * 100)
                    if metrics.total_calls > 0
                    else 0
                ),
                "average_execution_time": metrics.average_execution_time,
                "last_used": (
                    metrics.last_used.isoformat() if metrics.last_used else None
                ),
            }
        return summary

    async def _update_metrics(
        self, tool_name: str, start_time: float, success: bool
    ) -> None:
        """Update metrics for a tool execution."""
        if not self.enable_metrics:
            return

        metrics = self._metrics[tool_name]
        if success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1

        metrics.last_used = datetime.now(UTC)
        total_time = time.monotonic() - start_time
        metrics.average_execution_time = (
            metrics.average_execution_time * (metrics.total_calls - 1) + total_time
        ) / metrics.total_calls

    def _get_cache_key(self, name: str, arguments: dict[str, Any]) -> str:
        """Generate a cache key for tool execution."""
        args_str = json.dumps(arguments, sort_keys=True)
        key = f"{name}:{args_str}"
        return hashlib.sha256(key.encode()).hexdigest()

    def get_metrics(self, name: str) -> ToolMetrics:
        """Get metrics for a specific tool."""
        if not self.enable_metrics:
            raise ToolError("Metrics are disabled")

        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")

        return self._metrics[name]

    def list_tools(self) -> list[type[Tool]]:
        """List all registered tools."""
        return list(self._tools.values())

    def clear_cache(self, name: str | None = None) -> None:
        """Clear tool execution cache."""
        if name:
            if name not in self._tools:
                raise ToolError(f"Unknown tool: {name}")
            # Remove specific tool entries
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{name}:")]
            for k in keys_to_remove:
                self._cache.pop(k, None)
        else:
            self._cache.clear()

    async def stream_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: ToolContext | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream tool results."""
        if tool_name not in self._tools:
            raise ToolError(f"Tool not found: {tool_name}")

        tool_cls = self._tools[tool_name]
        tool = tool_cls(context=context or ToolContext())

        if not tool.metadata.supports_streaming:
            raise ToolError(f"Tool does not support streaming: {tool_name}")

        start_time = time.monotonic()

        # Initialize metrics
        metrics = self._metrics[tool_name] if self.enable_metrics else ToolMetrics()

        try:
            # Update metrics at start of execution attempt
            if self.enable_metrics:
                metrics.total_calls += 1
                metrics.last_used = datetime.now(UTC)

            async with self._execution_lock:
                async for chunk in tool.stream(args):
                    yield chunk

            # Update metrics for success
            if self.enable_metrics:
                metrics.successful_calls += 1
                execution_time = time.monotonic() - start_time
                if metrics.total_calls > 0:
                    metrics.average_execution_time = (
                        metrics.average_execution_time * (metrics.total_calls - 1)
                        + execution_time
                    ) / metrics.total_calls

        except Exception as e:
            if self.enable_metrics:
                metrics.failed_calls += 1
            raise ToolError(
                message=str(e),
                tool_name=tool_name,
                tool_version=tool.metadata.version,
                execution_context=tool.context.model_dump(),
            ) from e
