"""Resource management for AxiomMCP."""

import asyncio
import contextlib
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Final, Dict, Optional

from pydantic import AnyUrl, BaseModel, Field

from axiom_mcp.resources.advanced_types import TemplateResource

from ..exceptions import (
    ResourceUnavailableError,
    ResourceError,
)
from .base import Resource, ResourceType

logger = logging.getLogger(__name__)

MAX_RECOVERY_ATTEMPTS: Final = 3


class ResourceLock:
    """An async context manager for resource locking."""

    def __init__(self, lock: asyncio.Lock):
        self.lock = lock

    async def __aenter__(self) -> None:
        await self.lock.acquire()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.lock.release()


class ResourceHealth(BaseModel):
    """Health status for a resource."""

    is_healthy: bool = True
    last_check: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error_count: int = 0
    last_error: str | None = None
    recovery_attempts: int = 0


class ResourceStats(BaseModel):
    """Statistics for resource usage."""

    total_reads: int = 0
    total_writes: int = 0
    total_deletes: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    last_accessed: datetime | None = None
    average_response_time: float = 0.0
    error_count: int = 0
    health: ResourceHealth = Field(default_factory=ResourceHealth)


class ResourcePool(BaseModel):
    """Pool of resources with usage tracking and health monitoring."""

    model_config = {"arbitrary_types_allowed": True}
    max_size: int = Field(default=100)
    resources: dict[str, Resource] = Field(default_factory=dict)
    stats: dict[str, ResourceStats] = Field(
        default_factory=lambda: defaultdict(ResourceStats)
    )
    resource_types: dict[ResourceType, set[str]] = Field(
        default_factory=lambda: defaultdict(set)
    )
    max_error_count: int = Field(default=3)
    health_check_interval: float = Field(default=60.0)  # seconds
    recovery_backoff: float = Field(default=5.0)  # seconds
    resource_locks: dict[str, asyncio.Lock] = Field(default_factory=dict)
    access_times: dict[str, datetime] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self._global_lock = asyncio.Lock()
        self._cleanup_task = None
        self._health_check_task = None
        self._initialized = False
        self._shutting_down = False
        self._shutdown_event = asyncio.Event()  # Initialize the event

    async def initialize(self) -> None:
        """Initialize background tasks."""
        if not self._initialized:
            async with self._global_lock:
                if self._initialized:
                    return
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                self._health_check_task = asyncio.create_task(
                    self._periodic_health_check()
                )
                self._initialized = True

    async def _ensure_initialized(self) -> None:
        """Ensure pool is initialized."""
        if not self._initialized:
            await self.initialize()

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up least recently used resources."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)
                if self._shutdown_event.is_set():
                    break
                await self._cleanup_lru()
            except asyncio.CancelledError:
                break
            except Exception:
                if not self._shutdown_event.is_set():
                    logger.exception("Error in cleanup task")

    async def _cleanup_lru(self) -> None:
        """Remove least recently used resources when pool is full."""
        async with ResourceLock(self._global_lock):
            while len(self.resources) > self.max_size:
                lru_uri = min(self.access_times.items(), key=lambda x: x[1])[0]
                await self._remove_resource(lru_uri)

    async def _periodic_health_check(self) -> None:
        """Periodically check health of all resources."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                if self._shutdown_event.is_set():
                    break
                await self._check_all_resources()
            except asyncio.CancelledError:
                break
            except Exception:
                if not self._shutdown_event.is_set():
                    logger.exception("Error in health check task")

    async def _check_all_resources(self) -> None:
        """Check health of all resources."""
        async with ResourceLock(self._global_lock):
            for uri in list(self.resources.keys()):
                await self._check_resource_health(uri)

    async def _check_resource_health(self, uri: str) -> None:
        """Check health of a specific resource."""
        resource = self.resources.get(uri)
        if not resource:
            return

        stats = self.stats[uri]
        try:
            if await resource.exists():
                stats.health.is_healthy = True
                stats.health.error_count = 0
                stats.health.last_error = None
                stats.health.recovery_attempts = 0
            else:
                await self._handle_resource_error(uri, "Resource does not exist")
        except Exception as e:
            await self._handle_resource_error(uri, str(e))

    async def _handle_resource_error(self, uri: str, error: str) -> None:
        """Handle resource errors and attempt recovery."""
        stats = self.stats[uri]
        stats.health.is_healthy = False
        stats.health.error_count += 1
        stats.health.last_error = error
        stats.health.last_check = datetime.now(UTC)

        if stats.health.error_count >= self.max_error_count:
            # Attempt recovery
            await self._attempt_resource_recovery(uri)

    async def _attempt_resource_recovery(self, uri: str) -> None:
        """Attempt to recover a failed resource."""
        stats = self.stats[uri]
        stats.health.recovery_attempts += 1

        def _raise_unavailable() -> None:
            """Helper function to raise ResourceUnavailableError."""
            raise ResourceUnavailableError()

        # Exponential backoff for recovery attempts
        await asyncio.sleep(
            self.recovery_backoff * (2 ** (stats.health.recovery_attempts - 1))
        )

        try:
            resource = self.resources[uri]
            # Try to recover the resource
            if not await resource.recover():
                stats.health.is_healthy = False
                # Now using the helper function instead of raising directly
                _raise_unavailable()
            stats.health.is_healthy = True
            stats.health.error_count = 0
            logger.info("Successfully recovered resource: %s", uri)

        except Exception:
            logger.exception("Failed to recover resource %s", uri)
            if stats.health.recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
                # Remove resource after too many failed recovery attempts
                await self._remove_resource(uri)

    def _get_resource_lock(self, uri: str) -> ResourceLock:
        """Get or create a lock for a specific resource."""
        if uri not in self.resource_locks:
            self.resource_locks[uri] = asyncio.Lock()
        return ResourceLock(self.resource_locks[uri])

    async def add(self, resource: Resource) -> None:
        """Add a resource to the pool with proper locking."""
        await self._ensure_initialized()
        async with ResourceLock(self._global_lock):
            if len(self.resources) >= self.max_size:
                await self._cleanup_lru()
            uri = str(resource.uri)
            self.resources[uri] = resource
            self.resource_types[resource.resource_type].add(uri)
            self.access_times[uri] = datetime.now(UTC)

    async def get(self, uri: str) -> Resource | None:
        """Get a resource from the pool with access time update."""
        await self._ensure_initialized()
        if resource := self.resources.get(uri):
            self.access_times[uri] = datetime.now(UTC)
            return resource
        return None

    async def _remove_resource(self, uri: str) -> None:
        """Remove a resource and its associated data."""
        if resource := self.resources.pop(uri, None):
            self.resource_types[resource.resource_type].remove(uri)
            if not self.resource_types[resource.resource_type]:
                del self.resource_types[resource.resource_type]
            self.access_times.pop(uri, None)
            self.resource_locks.pop(uri, None)

    async def update_stats(
        self,
        uri: str,
        operation: str,
        bytes_count: int = 0,
        response_time: float = 0.0,
        error: bool = False,
    ) -> None:
        """Update usage statistics for a resource with locking."""
        async with self._get_resource_lock(uri):
            stats = self.stats[uri]
            stats.last_accessed = datetime.now(UTC)
            self.access_times[uri] = stats.last_accessed

            if operation == "read":
                stats.total_reads += 1
                stats.total_bytes_read += bytes_count
            elif operation == "write":
                stats.total_writes += 1
                stats.total_bytes_written += bytes_count
            elif operation == "delete":
                stats.total_deletes += 1

            if error:
                stats.error_count += 1
                await self._handle_resource_error(uri, "Operation failed")

            # Update average response time
            if response_time > 0:
                total_ops = stats.total_reads + stats.total_writes + stats.total_deletes
                stats.average_response_time = (
                    stats.average_response_time * (total_ops - 1) + response_time
                ) / total_ops

    async def shutdown(self) -> None:
        """Clean up background tasks and resources."""
        if self._shutting_down:
            return

        async with self._global_lock:
            if self._shutting_down:
                return

            self._shutting_down = True
            self._shutdown_event.set()

            # Cancel background tasks
            tasks = []
            if self._cleanup_task:
                self._cleanup_task.cancel()
                tasks.append(self._cleanup_task)
            if self._health_check_task:
                self._health_check_task.cancel()
                tasks.append(self._health_check_task)

            # Wait for tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Clean up resources
            cleanup_tasks = []
            for uri, resource in self.resources.items():
                cleanup_tasks.append(self._cleanup_resource(uri, resource))

            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            self.resources.clear()
            self.resource_types.clear()
            self.access_times.clear()
            self.resource_locks.clear()
            self.stats.clear()

            self._initialized = False
            self._shutting_down = False
            self._shutdown_event.clear()

    async def _cleanup_resource(self, uri: str, resource: Resource) -> None:
        """Clean up a single resource."""
        try:
            async with self._get_resource_lock(uri):
                try:
                    await resource.close()
                except Exception:
                    logger.exception(f"Error closing resource {uri}")
                finally:
                    self.stats[uri].health.is_healthy = False
        except Exception:
            logger.exception(f"Error cleaning up resource {uri}")

    async def cleanup(self) -> None:
        """Clean up background tasks and resources."""

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task
            self._health_check_task = None

        self._initialized = False
        self.resources.clear()
        self.resource_types.clear()
        self.access_times.clear()
        self.resource_locks.clear()
        self.stats.clear()


class ResourceManager:
    """Manages MCP resources."""

    def __init__(self, warn_on_duplicate_resources: bool = True):
        self._resources: Dict[str, Resource] = {}
        self.warn_on_duplicate_resources = warn_on_duplicate_resources

    async def add_resource(self, resource: Resource) -> Resource:
        """Add a resource to the manager."""
        uri_str = str(resource.uri)

        if uri_str in self._resources:
            if self.warn_on_duplicate_resources:
                logger.warning(f"Resource already exists: {uri_str}")
            return self._resources[uri_str]

        self._resources[uri_str] = resource
        return resource

    async def get_resource(self, uri: str | AnyUrl) -> Optional[Resource]:
        """Get a resource by URI, checking templates if no direct match."""
        uri_str = str(uri)
        logger.debug(f"Getting resource: {uri_str}")

        # First check direct resources
        if resource := self._resources.get(uri_str):
            return resource

        # Then check template resources
        for resource in self._resources.values():
            if isinstance(resource, TemplateResource):
                if params := resource.matches(uri_str):
                    try:
                        return await resource.create_resource(uri_str, params)
                    except Exception as e:
                        logger.error(f"Error creating resource from template: {e}")
                        continue

        raise ResourceError(f"Resource not found: {uri_str}")

    def list_resources(self) -> list[Resource]:
        """List all registered resources."""
        return list(self._resources.values())

    def list_templates(self) -> list[TemplateResource]:
        """List all registered template resources."""
        return [r for r in self._resources.values() if isinstance(r, TemplateResource)]

    async def read_resource(self, uri: str | AnyUrl) -> str | bytes:
        """Read content from a resource."""
        if resource := await self.get_resource(uri):
            try:
                return await resource.read()
            except Exception as e:
                raise ResourceError(f"Error reading resource {uri}: {e}")
        raise ResourceError(f"Resource not found: {uri}")
