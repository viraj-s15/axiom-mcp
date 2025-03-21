"""Resource management for AxiomMCP."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any, Final

from pydantic import AnyUrl, BaseModel, Field

from ..exceptions import (
    ResourceNotFoundError,
    ResourceUnavailableError,
    UnknownResourceTypeError,
)
from .base import Resource, ResourceType
from .types import (
    BinaryResource,
    FileResource,
    FunctionResource,
    HttpResource,
    StreamResource,
    TextResource,
)

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

    max_size: int = Field(default=100)
    resources: dict[str, Resource] = Field(default_factory=dict)
    stats: dict[str, ResourceStats] = Field(
        default_factory=lambda: defaultdict(ResourceStats)
    )
    resource_types: dict[ResourceType, set[str]] = Field(
        default_factory=lambda: defaultdict(set)
    )
    # New fields for resource management
    max_error_count: int = Field(default=3)
    health_check_interval: float = Field(default=60.0)  # seconds
    recovery_backoff: float = Field(default=5.0)  # seconds
    _locks: dict[str, asyncio.Lock] = Field(default_factory=dict)
    _access_times: dict[str, datetime] = Field(default_factory=dict)
    _cleanup_task: asyncio.Task | None = None
    _health_check_task: asyncio.Task | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._global_lock = asyncio.Lock()
        self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background tasks for cleanup and health checks."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        if not self._health_check_task:
            self._health_check_task = asyncio.create_task(self._periodic_health_check())

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up least recently used resources."""
        while True:
            try:
                await asyncio.sleep(30)  # Run cleanup every 30 seconds
                await self._cleanup_lru()
            except Exception:
                logger.exception("Error in cleanup task")

    async def _cleanup_lru(self) -> None:
        """Remove least recently used resources when pool is full."""
        async with ResourceLock(self._global_lock):
            while len(self.resources) > self.max_size:
                # Find least recently used resource
                lru_uri = min(self._access_times.items(), key=lambda x: x[1])[0]
                await self._remove_resource(lru_uri)

    async def _periodic_health_check(self) -> None:
        """Periodically check health of all resources."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_resources()
            except Exception:
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
        if uri not in self._locks:
            self._locks[uri] = asyncio.Lock()
        return ResourceLock(self._locks[uri])

    async def add(self, resource: Resource) -> None:
        """Add a resource to the pool with proper locking."""
        async with ResourceLock(self._global_lock):
            if len(self.resources) >= self.max_size:
                await self._cleanup_lru()

            uri = str(resource.uri)
            self.resources[uri] = resource
            self.resource_types[resource.resource_type].add(uri)
            self._access_times[uri] = datetime.now(UTC)

    async def get(self, uri: str) -> Resource | None:
        """Get a resource from the pool with access time update."""
        if resource := self.resources.get(uri):
            self._access_times[uri] = datetime.now(UTC)
            return resource
        return None

    async def _remove_resource(self, uri: str) -> None:
        """Remove a resource and its associated data."""
        if resource := self.resources.pop(uri, None):
            self.resource_types[resource.resource_type].remove(uri)
            if not self.resource_types[resource.resource_type]:
                del self.resource_types[resource.resource_type]
            self._access_times.pop(uri, None)
            self._locks.pop(uri, None)

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
            self._access_times[uri] = stats.last_accessed

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


class ResourceManager:
    """Core resource manager with pooling and monitoring."""

    def __init__(self, pool_size: int = 100):
        self.pool = ResourcePool(max_size=pool_size)
        self._type_registry: dict[ResourceType, type[Resource]] = {
            ResourceType.TEXT: TextResource,
            ResourceType.BINARY: BinaryResource,
            ResourceType.FILE: FileResource,
            ResourceType.HTTP: HttpResource,
            ResourceType.FUNCTION: FunctionResource,
            ResourceType.STREAM: StreamResource,
        }

    def _handle_recovery_failure(self, uri: str) -> None:
        """Handle resource recovery failure."""
        raise ResourceUnavailableError()

    def _handle_resource_not_found(self, uri: str) -> None:
        """Handle resource not found scenario."""
        raise ResourceNotFoundError(uri)

    def _handle_unknown_resource_type(self, resource_type: ResourceType) -> None:
        """Handle unknown resource type scenario."""
        raise UnknownResourceTypeError(str(resource_type))

    def _raise_resource_unavailable(self) -> None:
        """Helper method to raise ResourceUnavailableError."""
        raise ResourceUnavailableError()

    async def _update_resource_health(
        self, resource: Resource, stats: ResourceStats
    ) -> None:
        stats.health.last_check = datetime.now(UTC)

        try:
            recovery_successful = await resource.recover()

            if not recovery_successful:
                logger.warning(
                    "Resource recovery failed for %s: recovery method returned False",
                    resource.uri,
                )
                stats.health.is_healthy = False
                stats.health.last_error = "Recovery failed"
                self._raise_resource_unavailable()

            stats.health.is_healthy = True
            stats.health.error_count = 0
            stats.health.last_error = None

        except Exception as e:
            logger.exception("Resource recovery failed with exception")
            stats.health.is_healthy = False
            stats.health.last_error = str(e)
            stats.health.error_count += 1
            raise ResourceUnavailableError() from e

    async def recover_resource(self, uri: str) -> bool:
        """Attempt to recover a resource that is in an unhealthy state."""
        try:
            resource = await self.get_resource(uri)
            if resource is None:
                return False
            stats = self.pool.stats[str(resource.uri)]
            recovery_successful = await resource.recover()
            if not recovery_successful:
                stats.health.is_healthy = False
                self._raise_resource_unavailable()
                return False
            stats.health.is_healthy = True
        except Exception:
            logger.exception("Failed to recover resource: %s", uri)
            return False
        else:
            logger.info("Successfully recovered resource: %s", uri)
            return True

    async def create_resource(
        self, resource_type: ResourceType, uri: str | AnyUrl, **kwargs: Any
    ) -> Resource:
        """Create a new resource of the specified type."""
        if resource_cls := self._type_registry.get(resource_type):
            # Convert string URI to AnyUrl if needed
            if isinstance(uri, str):
                # Handle special cases for non-standard schemes
                if "://" not in uri:
                    uri = f"{resource_type.value}://{uri}"
                uri = AnyUrl(uri)

            resource = resource_cls(uri=uri, resource_type=resource_type, **kwargs)
            await self.pool.add(resource)
            return resource
        self._handle_unknown_resource_type(resource_type)
        raise UnknownResourceTypeError(str(resource_type))  # For type checking

    async def get_resource(self, uri: str) -> Resource | None:
        """Get a resource by URI."""
        return await self.pool.get(uri)

    async def read_resource(
        self, uri: str, create_if_missing: bool = False, **create_kwargs: Any
    ) -> str | bytes:
        """Read content from a resource."""
        start_time = datetime.now(UTC)
        try:
            if resource := await self.get_resource(uri):
                content = await resource.read()
                size = (
                    len(content.encode()) if isinstance(content, str) else len(content)
                )
                await self.pool.update_stats(
                    uri,
                    "read",
                    bytes_count=size,
                    response_time=(datetime.now(UTC) - start_time).total_seconds(),
                )
                return content

            if create_if_missing:
                resource = await self.create_resource(
                    create_kwargs.pop("resource_type", ResourceType.TEXT),
                    uri,
                    **create_kwargs,
                )
                return await self.read_resource(str(resource.uri))
            self._handle_resource_not_found(uri)
        except Exception:
            await self.pool.update_stats(uri, "read", error=True)
            raise
        else:
            return b""  # Not reachable due to _handle_resource_not_found raising

    async def write_resource(
        self,
        uri: str,
        content: str | bytes,
        create_if_missing: bool = True,
        **create_kwargs: Any,
    ) -> None:
        """Write content to a resource."""
        start_time = datetime.now(UTC)
        try:
            resource = await self.get_resource(uri)
            if not resource and create_if_missing:
                resource = await self.create_resource(
                    create_kwargs.pop("resource_type", ResourceType.TEXT),
                    uri,
                    **create_kwargs,
                )
            if not resource:
                self._handle_resource_not_found(uri)
                return  # Not reachable, just for type checking

            await resource.write(content)
            size = len(content.encode()) if isinstance(content, str) else len(content)
            await self.pool.update_stats(
                uri,
                "write",
                bytes_count=size,
                response_time=(datetime.now(UTC) - start_time).total_seconds(),
            )

        except Exception:
            await self.pool.update_stats(uri, "write", error=True)
            raise

    async def delete_resource(self, uri: str) -> None:
        """Delete a resource."""
        start_time = datetime.now(UTC)
        try:
            if resource := await self.get_resource(uri):
                await resource.delete()
                await self.pool._remove_resource(str(resource.uri))
                await self.pool.update_stats(
                    uri,
                    "delete",
                    response_time=(datetime.now(UTC) - start_time).total_seconds(),
                )
            else:
                self._handle_resource_not_found(uri)

        except Exception:
            await self.pool.update_stats(uri, "delete", error=True)
            raise

    async def stream_resource(
        self, uri: str, chunk_size: int = 8192
    ) -> AsyncGenerator[str | bytes, None]:
        """Stream content from a resource."""
        if resource := await self.get_resource(uri):
            async for chunk in resource.read_stream():
                yield chunk
        else:
            self._handle_resource_not_found(uri)
            return  # Not reachable, just for type checking
            yield b""  # For type checking

    def get_stats(
        self, uri: str | None = None
    ) -> ResourceStats | dict[str, ResourceStats]:
        """Get statistics for one or all resources."""
        if uri:
            return self.pool.stats[uri]
        return dict(self.pool.stats)

    def get_resources_by_type(self, resource_type: ResourceType) -> list[Resource]:
        """Get all resources of a specific type."""
        return [
            self.pool.resources[uri]
            for uri in self.pool.resource_types.get(resource_type, set())
        ]

    def register_resource_type(
        self, resource_type: ResourceType, resource_class: type[Resource]
    ) -> None:
        """Register a new resource type."""
        self._type_registry[resource_type] = resource_class
