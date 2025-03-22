"""Tool management and execution."""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import cachetools
from pydantic import BaseModel, Field

from ..exceptions import ToolError
from .base import Tool, ToolContext, ToolDependency

logger = logging.getLogger(__name__)


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
        default_timeout: float | None = 30.0,
        enable_metrics: bool = True,
    ):
        self._tools: dict[str, type[Tool]] = {}
        self._metrics: dict[str, ToolMetrics] = {}
        self._cache = cachetools.TTLCache(maxsize=cache_size, ttl=3600)
        self._dependency_graph: dict[str, set[str]] = defaultdict(set)
        self._default_timeout = default_timeout
        self._enable_metrics = enable_metrics
        self._execution_lock = asyncio.Lock()

    def register_tool(self, tool_cls: type[Tool]) -> None:
        """Register a tool with dependency tracking."""
        name = tool_cls.metadata.name
        self._tools[name] = tool_cls

        # Update dependency graph
        for dep in tool_cls.metadata.dependencies:
            if isinstance(dep, dict):
                dep = ToolDependency(**dep)
            self._dependency_graph[name].add(dep.tool_name)

        # Initialize metrics
        if self._enable_metrics:
            self._metrics[name] = ToolMetrics()

        logger.info(f"Registered tool: {name} v{tool_cls.metadata.version}")

    def _calculate_cache_key(self, tool: Tool, args: dict[str, Any]) -> str:
        """Calculate cache key based on tool name, version, args and context."""
        context_dict = tool.context.model_dump()
        context_hash = hashlib.sha256(
            json.dumps(context_dict, sort_keys=True).encode()
        ).hexdigest()

        key_parts = [
            tool.metadata.name,
            tool.metadata.version,
            json.dumps(args, sort_keys=True),
            context_hash,
        ]
        return hashlib.sha256(":".join(key_parts).encode()).hexdigest()

    def _update_metrics(
        self, tool_name: str, start_time: datetime, success: bool
    ) -> None:
        """Update metrics for tool execution."""
        if not self._enable_metrics:
            return

        metrics = self._metrics[tool_name]
        now = datetime.now(UTC)
        execution_time = (now - start_time).total_seconds()

        metrics.total_calls += 1
        if success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1

        # Update average execution time
        if metrics.total_calls == 1:
            metrics.average_execution_time = execution_time
        else:
            metrics.average_execution_time = (
                metrics.average_execution_time * (metrics.total_calls - 1)
                + execution_time
            ) / metrics.total_calls

        metrics.last_used = now

    def get_metrics(self, tool_name: str) -> ToolMetrics | None:
        """Get metrics for a specific tool."""
        return self._metrics.get(tool_name)

    async def execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: ToolContext | None = None,
    ) -> Any:
        """Execute a tool with caching and metrics."""
        if tool_name not in self._tools:
            raise ToolError(f"Tool not found: {tool_name}")

        tool_cls = self._tools[tool_name]
        tool = tool_cls(context=context or ToolContext())

        # Set default timeout if not specified
        if tool.context.timeout is None:
            tool.context.timeout = self._default_timeout

        # Check cache if enabled
        if tool.metadata.cacheable and tool.context.cache_enabled:
            cache_key = self._calculate_cache_key(tool, args)
            if cache_entry := self._cache.get(cache_key):
                if (
                    cache_entry.tool_version == tool.metadata.version
                    and cache_entry.context_hash
                    == hashlib.sha256(
                        json.dumps(tool.context.model_dump(), sort_keys=True).encode()
                    ).hexdigest()
                ):
                    return cache_entry.result

        start_time = datetime.now(UTC)
        try:
            cache_key = self._calculate_cache_key(tool, args)
            async with self._execution_lock:
                result = await tool(args)

            if tool.metadata.cacheable and tool.context.cache_enabled:
                cache_entry = ToolCacheEntry(
                    result=result,
                    tool_version=tool.metadata.version,
                    context_hash=hashlib.sha256(
                        json.dumps(tool.context.model_dump(), sort_keys=True).encode()
                    ).hexdigest(),
                )
                self._cache[cache_key] = cache_entry

            self._update_metrics(tool_name, start_time, success=True)
            return result

        except Exception as e:
            self._update_metrics(tool_name, start_time, success=False)
            raise ToolError(
                message=str(e),
                tool_name=tool_name,
                tool_version=tool.metadata.version,
                execution_context=tool.context.model_dump(),
            ) from e

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

        start_time = datetime.now(UTC)
        try:
            async with self._execution_lock:
                async for chunk in tool.stream(args):
                    yield chunk

            self._update_metrics(tool_name, start_time, success=True)

        except Exception as e:
            self._update_metrics(tool_name, start_time, success=False)
            raise ToolError(
                message=str(e),
                tool_name=tool_name,
                tool_version=tool.metadata.version,
                execution_context=tool.context.model_dump(),
            ) from e
