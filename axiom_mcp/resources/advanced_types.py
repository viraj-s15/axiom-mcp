"""Advanced resource types for AxiomMCP."""

import asyncio
import gzip
import json
import lzma
import zlib
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import aiosqlite
import jinja2
from aiosqlite.core import Connection
from pydantic import BaseModel, Field

from ..exceptions import ResourceError
from .base import Resource, ResourceType


class CompressionError(ResourceError):
    """Raised when compression/decompression fails."""

    def __init__(self, algorithm: str) -> None:
        super().__init__(f"Unsupported compression algorithm: {algorithm}")


class DatabaseConnectionError(ResourceError):
    """Raised when a database connection cannot be acquired."""

    def __init__(self) -> None:
        super().__init__("Could not acquire database connection")


class EmptyIdentifierError(ResourceError):
    """Raised when a SQL identifier is empty."""

    def __init__(self) -> None:
        super().__init__("Empty identifier")


class InvalidIdentifierError(ResourceError):
    """Raised when a SQL identifier contains invalid characters."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Invalid SQL identifier: {name}")


class NumericStartIdentifierError(ResourceError):
    """Raised when a SQL identifier starts with a number."""

    def __init__(self, name: str) -> None:
        super().__init__(f"SQL identifier cannot start with a number: {name}")


class UnsupportedTemplateTypeError(ResourceError):
    """Raised when an unsupported template type is used."""

    def __init__(self, template_type: str) -> None:
        super().__init__(f"Unsupported template type: {template_type}")


class TemplateRenderError(ResourceError):
    """Raised when template rendering fails."""

    def __init__(self, error: Exception) -> None:
        super().__init__(f"Template rendering error: {error}")


def compress_content(content: str | bytes, algorithm: str) -> bytes:
    """Compress content using the specified algorithm."""
    if isinstance(content, str):
        content = content.encode("utf-8")

    match algorithm:
        case "gzip":
            return gzip.compress(content)
        case "lzma":
            return lzma.compress(content)
        case "zlib":
            return zlib.compress(content)
        case _:
            raise CompressionError(algorithm)


def decompress_content(content: bytes, algorithm: str) -> bytes:
    """Decompress content using the specified algorithm."""
    match algorithm:
        case "gzip":
            return gzip.decompress(content)
        case "lzma":
            return lzma.decompress(content)
        case "zlib":
            return zlib.decompress(content)
        case _:
            raise CompressionError(algorithm)


class DatabaseConfig(BaseModel):
    """Configuration for database connections."""

    table_name: str
    key_column: str = "key"
    value_column: str = "value"
    metadata_column: str = "metadata"
    timestamp_column: str = "timestamp"
    pool_size: int = 5
    max_overflow: int = 2
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # Recycle connections after 1 hour


class ConnectionPool:
    """A simple connection pool for SQLite connections."""

    def __init__(self, database_path: str, config: DatabaseConfig):
        self.database_path = database_path
        self.config = config
        self._pool: list[Connection] = []
        self._in_use: dict[Connection, datetime] = {}
        self._overflow = 0
        self._lock = asyncio.Lock()
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            for _ in range(self.config.pool_size):
                conn = await self._create_connection()
                self._pool.append(conn)

            self._initialized = True

    async def _create_connection(self) -> Connection:
        """Create a new database connection."""
        return await aiosqlite.connect(
            self.database_path, isolation_level=None  # Enable autocommit mode
        )

    async def get_connection(self) -> Connection:
        """Get a connection from the pool."""
        await self.initialize()

        async with self._lock:
            # Try to get an available connection
            while self._pool:
                conn = self._pool.pop()
                # Check if connection needs recycling
                now = datetime.now(UTC)
                if (
                    now.timestamp() - self._in_use.get(conn, now).timestamp()
                    > self.config.pool_recycle
                ):
                    await conn.close()
                    conn = await self._create_connection()

                self._in_use[conn] = now
                return conn

            # Handle overflow if no connections are available
            if self._overflow < self.config.max_overflow:
                conn = await self._create_connection()
                self._overflow += 1
                self._in_use[conn] = datetime.now(UTC)
                return conn

            # Wait for a connection to become available
            for _ in range(int(self.config.pool_timeout / 0.1)):
                await asyncio.sleep(0.1)
                if self._pool:
                    conn = self._pool.pop()
                    self._in_use[conn] = datetime.now(UTC)
                    return conn

            raise DatabaseConnectionError()

    async def release_connection(self, conn: Connection) -> None:
        """Release a connection back to the pool."""
        async with self._lock:
            if conn in self._in_use:
                del self._in_use[conn]
                if len(self._pool) < self.config.pool_size:
                    self._pool.append(conn)
                else:
                    self._overflow -= 1
                    await conn.close()

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            for conn in self._pool:
                await conn.close()
            for conn in self._in_use:
                await conn.close()
            self._pool.clear()
            self._in_use.clear()
            self._overflow = 0
            self._initialized = False

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """Get a pooled connection context manager."""
        connection = await self.get_connection()
        try:
            yield connection
        finally:
            await self.release_connection(connection)


class DatabaseResource(Resource):
    """Resource that reads/writes from/to a SQLite database with connection pooling."""

    database_path: str = Field(description="Path to the SQLite database")
    config: DatabaseConfig = Field(description="Database configuration")
    resource_type: ResourceType = ResourceType.DATABASE
    _connection_pools: dict[str, ConnectionPool] = {}
    _lock = asyncio.Lock()

    def __init__(self, **data):
        super().__init__(**data)
        self._ensure_pool()
        asyncio.create_task(self._ensure_table_exists())

    def _ensure_pool(self) -> None:
        """Ensure connection pool exists for this database."""
        if self.database_path not in self._connection_pools:
            self._connection_pools[self.database_path] = ConnectionPool(
                self.database_path, self.config
            )

    @property
    def pool(self) -> ConnectionPool:
        """Get the connection pool for this resource."""
        return self._connection_pools[self.database_path]

    async def _ensure_table_exists(self) -> None:
        """Ensure the required table exists."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                    {self.config.key_column} TEXT PRIMARY KEY,
                    {self.config.value_column} BLOB,
                    {self.config.metadata_column} TEXT,
                    {self.config.timestamp_column} TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            await conn.commit()


def validate_identifier(name: str) -> str:
    """Validate that a SQL identifier only contains safe characters."""
    if not name or not name.strip():
        raise EmptyIdentifierError()

    # Only allow alphanumeric characters and underscores
    if not all(c.isalnum() or c == "_" for c in name):
        raise InvalidIdentifierError(name)

    # Don't allow identifiers starting with numbers
    if name[0].isdigit():
        raise NumericStartIdentifierError(name)

    return name


async def read(self) -> str | bytes:
    """Read content from the database with connection pooling."""
    async with self.pool.acquire() as conn:
        query = """
            SELECT value, metadata
            FROM database_entries
            WHERE key = ?
        """
        params = [str(self.uri)]

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return b""

            content, metadata_json = row
            if metadata_json:
                metadata = json.loads(metadata_json)
                encoding = metadata.get("encoding", "utf-8")
                if metadata.get("content_type", "").startswith("text/"):
                    return content.decode(encoding)

            return content


async def write(self, content: str | bytes) -> None:
    """Write content to the database with connection pooling."""
    if isinstance(content, str):
        content = content.encode(self.metadata.encoding or "utf-8")

    query = """
        INSERT OR REPLACE INTO database_entries
        (key, value, metadata, timestamp)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """
    params = [str(self.uri), content, json.dumps(self.metadata.model_dump())]

    async with self.pool.acquire() as conn:
        await conn.execute(query, params)
        await conn.commit()


async def delete(self) -> None:
    """Delete content from the database with connection pooling."""
    async with self.pool.acquire() as conn:
        query = "DELETE FROM database_entries WHERE key = ?"
        params = [str(self.uri)]
        await conn.execute(query, params)
        await conn.commit()


@classmethod
async def cleanup_pools(cls) -> None:
    """Close all connection pools."""
    for pool in cls._connection_pools.values():
        await pool.close_all()
    cls._connection_pools.clear()


class TemplateConfig(BaseModel):
    """Configuration for template resources."""

    template_type: str = "jinja2"
    auto_reload: bool = True
    cache_size: int = 50
    strict_variables: bool = True


class TemplateResource(Resource):
    """Resource for template-based content generation."""

    template_string: str = Field(description="Template string or content")
    config: TemplateConfig = Field(default_factory=TemplateConfig)
    context: dict[str, Any] = Field(default_factory=dict)
    resource_type: ResourceType = ResourceType.TEMPLATE

    _template_engines: dict[str, Any] = {}
    _template_cache: dict[str, Any] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self._setup_template_engine()

    def _setup_template_engine(self) -> None:
        """Set up the template engine based on configuration."""
        if (
            self.config.template_type == "jinja2"
            and "jinja2" not in self._template_engines
        ):
            self._template_engines["jinja2"] = jinja2.Environment(
                autoescape=True,
                auto_reload=self.config.auto_reload,
                cache_size=self.config.cache_size,
                undefined=(
                    jinja2.StrictUndefined
                    if self.config.strict_variables
                    else jinja2.Undefined
                ),
            )

    def _get_template(self) -> Any:
        """Get or create a template instance."""
        cache_key = f"{self.uri}_{hash(self.template_string)}"
        if not self.config.auto_reload and cache_key in self._template_cache:
            return self._template_cache[cache_key]

        if self.config.template_type == "jinja2":
            template = self._template_engines["jinja2"].from_string(
                self.template_string
            )
            if not self.config.auto_reload:
                self._template_cache[cache_key] = template
            return template

        raise UnsupportedTemplateTypeError(self.config.template_type)

    async def read(self) -> str:
        """Render the template with the current context."""
        try:
            template = self._get_template()
            # Support async context values
            context = {}
            for key, value in self.context.items():
                context[key] = await value if asyncio.iscoroutine(value) else value

            return template.render(**context)
        except Exception as e:
            raise TemplateRenderError(e) from e

    async def write(self, content: str | bytes) -> None:
        """Update the template string."""
        if isinstance(content, bytes):
            content = content.decode(self.metadata.encoding or "utf-8")
        # Use object.__setattr__ to bypass Pydantic's immutability
        object.__setattr__(self, "template_string", str(content))
        # Clear template cache for this resource
        cache_key = f"{self.uri}_{hash(self.template_string)}"
        self._template_cache.pop(cache_key, None)


class CachedResource(Resource):
    """A wrapper resource that adds caching to any other resource."""

    wrapped_resource: Resource = Field(description="The resource to cache")
    ttl: int = Field(default=3600, description="Cache TTL in seconds")
    resource_type: ResourceType = ResourceType.CACHE

    _cache: dict[str, Any] = {}
    _cache_times: dict[str, datetime] = {}

    async def read(self) -> str | bytes:
        """Read content, using cache if available."""
        cache_key = str(self.uri)
        now = datetime.now(UTC)

        if cache_key in self._cache:
            cache_time = self._cache_times[cache_key]
            if (now - cache_time).total_seconds() < self.ttl:
                return self._cache[cache_key]

        content = await self.wrapped_resource.read()
        self._cache[cache_key] = content
        self._cache_times[cache_key] = now
        return content

    async def write(self, content: str | bytes) -> None:
        """Write content and update cache."""
        await self.wrapped_resource.write(content)
        self._cache[str(self.uri)] = content
        self._cache_times[str(self.uri)] = datetime.now(UTC)

    async def delete(self) -> None:
        """Delete content and clear cache."""
        await self.wrapped_resource.delete()
        self._cache.pop(str(self.uri), None)
        self._cache_times.pop(str(self.uri), None)

    def clear_cache(self) -> None:
        """Clear the cache for this resource."""
        self._cache.pop(str(self.uri), None)
        self._cache_times.pop(str(self.uri), None)

    @classmethod
    def clear_all_caches(cls) -> None:
        """Clear all resource caches."""
        cls._cache.clear()
        cls._cache_times.clear()


class CompressionConfig(BaseModel):
    """Configuration for compression settings."""

    algorithm: str = Field(default="gzip", pattern="^(gzip|lzma|zlib)$")
    compression_level: int = Field(default=-1)  # -1 means default level
    chunk_size: int = Field(default=8192)
    min_size: int = Field(default=1024)  # Only compress if larger than this


class CompressedResource(Resource):
    """A resource that automatically handles compression/decompression."""

    wrapped_resource: Resource = Field(description="The resource to compress")
    config: CompressionConfig = Field(default_factory=CompressionConfig)
    resource_type: ResourceType = ResourceType.COMPRESSED

    async def read(self) -> str | bytes:
        """Read and decompress content."""
        content = await self.wrapped_resource.read()
        if isinstance(content, str):
            content = content.encode(self.metadata.encoding or "utf-8")
        if self.metadata.compression:
            content = decompress_content(content, self.metadata.compression)
        if self.mime_type.startswith("text/"):
            if isinstance(content, (memoryview | bytearray)):
                content = bytes(content)
            return content.decode(self.metadata.encoding or "utf-8")
        return content

    async def write(self, content: str | bytes) -> None:
        """Compress and write content."""
        if isinstance(content, str):
            content = content.encode(self.metadata.encoding or "utf-8")
        if len(content) > self.config.min_size:
            compressed = compress_content(content, self.config.algorithm)
            self.update_metadata(compression=self.config.algorithm)
            await self.wrapped_resource.write(compressed)
        else:
            await self.wrapped_resource.write(content)


class TransformationPipeline(BaseModel):
    """A pipeline of content transformations."""

    steps: list[Callable[[str | bytes], str | bytes]]
    name: str = "custom_pipeline"
    description: str | None = None


def safe_encode(content: str | bytes | bytearray | memoryview) -> bytes:
    """Safely encode content to bytes."""
    if isinstance(content, (bytearray | memoryview)):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8")
    return content


def safe_decode(
    content: bytes | bytearray | memoryview, encoding: str = "utf-8"
) -> str:
    """Safely decode bytes to string."""
    if isinstance(content, (bytearray | memoryview)):
        content = bytes(content)
    return content.decode(encoding)


class TransformedResource(Resource):
    """A resource that applies transformations to content."""

    wrapped_resource: Resource = Field(description="The resource to transform")
    pipeline: TransformationPipeline = Field(
        description="Transformation pipeline to apply"
    )
    reverse_pipeline: bool = Field(
        default=False, description="Whether to apply pipeline in reverse for writing"
    )
    resource_type: ResourceType = ResourceType.FUNCTION

    async def read(self) -> str | bytes:
        """Read and transform content through pipeline."""
        content = await self.wrapped_resource.read()
        for transform in self.pipeline.steps:
            content = transform(content)
        return content

    async def write(self, content: str | bytes) -> None:
        """Transform and write content through pipeline."""
        if self.reverse_pipeline:
            for transform in reversed(self.pipeline.steps):
                content = transform(content)
        await self.wrapped_resource.write(content)


# Common transformation functions
def text_transform(
    fn: Callable[[str], str],
) -> Callable[[str | bytes], str | bytes]:
    """Create a text-based transformation."""

    def wrapper(content: str | bytes) -> str | bytes:
        if isinstance(content, (bytes | bytearray | memoryview)):
            content = safe_decode(content)
        result = fn(str(content))
        return safe_encode(result)

    return wrapper


def bytes_transform(
    fn: Callable[[bytes], bytes],
) -> Callable[[str | bytes], str | bytes]:
    """Create a bytes-based transformation."""

    def wrapper(content: str | bytes) -> str | bytes:
        if not isinstance(content, bytes):
            content = safe_encode(content)
        return fn(content)

    return wrapper


def create_json_pipeline() -> TransformationPipeline:
    """Create a JSON processing pipeline."""
    return TransformationPipeline(
        name="json_processing",
        description="Process JSON content with compression",
        steps=[
            text_transform(lambda x: json.dumps(json.loads(x))),
            bytes_transform(lambda x: compress_content(x, "gzip")),
        ],
    )
