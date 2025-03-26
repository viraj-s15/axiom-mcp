"""Advanced resource types for AxiomMCP."""

import asyncio
import gzip
import inspect
import json
import lzma
import re
import zlib
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import aiosqlite
from aiosqlite.core import Connection
from pydantic import BaseModel, Field, AnyUrl
from typing import Any, Dict, Optional, Union

from ..exceptions import ResourceError
from ..utilities.logging import get_logger
from .base import Resource, ResourceType

logger = get_logger(__name__)


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
    pool_recycle: int = 3600


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
        return await aiosqlite.connect(self.database_path, isolation_level=None)

    async def get_connection(self) -> Connection:
        """Get a connection from the pool."""
        await self.initialize()

        async with self._lock:
            while self._pool:
                conn = self._pool.pop()
                now = datetime.now(UTC)
                if (
                    now.timestamp() - self._in_use.get(conn, now).timestamp()
                    > self.config.pool_recycle
                ):
                    await conn.close()
                    conn = await self._create_connection()

                self._in_use[conn] = now
                return conn

            if self._overflow < self.config.max_overflow:
                conn = await self._create_connection()
                self._overflow += 1
                self._in_use[conn] = datetime.now(UTC)
                return conn

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
        # Validate all identifiers first
        table_name = validate_identifier(self.config.table_name)
        key_column = validate_identifier(self.config.key_column)
        value_column = validate_identifier(self.config.value_column)
        metadata_column = validate_identifier(self.config.metadata_column)
        timestamp_column = validate_identifier(self.config.timestamp_column)

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS [{table_name}] (
                    [{key_column}] TEXT PRIMARY KEY,
                    [{value_column}] BLOB,
                    [{metadata_column}] TEXT,
                    [{timestamp_column}] TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            await conn.commit()

    async def read(self) -> str | bytes:
        """Read content from the database with connection pooling."""
        # Validate identifiers to prevent SQL injection
        table_name = validate_identifier(self.config.table_name)
        value_column = validate_identifier(self.config.value_column)
        metadata_column = validate_identifier(self.config.metadata_column)
        key_column = validate_identifier(self.config.key_column)

        # Use proper SQLite identifier quoting with square brackets
        # All identifiers are pre-validated, eliminating SQL injection risk
        async with (
            self.pool.acquire() as conn,
            conn.execute(
                """
                SELECT [?], [?]
                FROM [?]
                WHERE [?] = ?
                """,
                [value_column, metadata_column, table_name, key_column, str(self.uri)],
            ) as cursor,
        ):
            row = await cursor.fetchone()
            if row is None:
                return b""
            content, metadata_json = row
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    encoding = metadata.get("encoding", "utf-8")
                    if metadata.get("content_type", "").startswith("text/"):
                        return content.decode(encoding)
                except json.JSONDecodeError:
                    # If metadata parsing fails, treat as raw content
                    pass
            return content

    async def write(self, content: str | bytes) -> None:
        """Write content to the database with connection pooling."""
        if isinstance(content, str):
            content = content.encode(self.metadata.encoding or "utf-8")

        # Validate identifiers
        cols = self.config
        table_name = validate_identifier(cols.table_name)
        key_col = validate_identifier(cols.key_column)
        value_col = validate_identifier(cols.value_column)
        meta_col = validate_identifier(cols.metadata_column)
        time_col = validate_identifier(cols.timestamp_column)

        # Use validated identifiers in parameterized query
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO [?]
                ([?], [?], [?], [?])
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    table_name,
                    key_col,
                    value_col,
                    meta_col,
                    time_col,
                    str(self.uri),
                    content,
                    json.dumps(self.metadata.model_dump()),
                ],
            )
            await conn.commit()

    async def delete(self) -> None:
        """Delete content from the database with connection pooling."""
        # Validate identifiers
        table_name = validate_identifier(self.config.table_name)
        key_column = validate_identifier(self.config.key_column)

        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM [?] WHERE [?] = ?", [table_name, key_column, str(self.uri)]
            )
            await conn.commit()

    @classmethod
    async def cleanup_pools(cls) -> None:
        """Close all connection pools."""
        for pool in cls._connection_pools.values():
            await pool.close_all()
        cls._connection_pools.clear()


def validate_identifier(name: str) -> str:
    """Validate that a SQL identifier only contains safe characters."""
    if not name or not name.strip():
        raise EmptyIdentifierError()

    if not all(c.isalnum() or c == "_" for c in name):
        raise InvalidIdentifierError(name)

    if name[0].isdigit():
        raise NumericStartIdentifierError(name)

    return name


class TemplateConfig(BaseModel):
    """Configuration for template resources."""

    template_type: str = "jinja2"
    auto_reload: bool = True
    cache_size: int = 50
    strict_variables: bool = True


class TemplateResource(Resource):
    """Resource for template-based content generation."""

    uri_template: str = Field(description="Template string with parameters")
    fn: Callable = Field(exclude=True)  # Function to generate content
    parameters: dict = Field(
        default_factory=dict, description="Parameter schema for the template"
    )
    resource_type: ResourceType = ResourceType.TEMPLATE

    @classmethod
    def from_function(
        cls,
        fn: Callable,
        uri_template: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> "TemplateResource":
        """Create a template from a function."""
        func_name = name or fn.__name__
        if func_name == "<lambda>":
            raise ValueError("You must provide a name for lambda functions")

        # Get function signature parameters
        params = {}
        sig = inspect.signature(fn)
        for param_name, param in sig.parameters.items():
            params[param_name] = {
                "type": "string",  # Default to string for template parameters
                "description": f"Parameter {param_name} for template",
                "required": param.default == inspect.Parameter.empty,
            }

        return cls(
            uri=AnyUrl(uri_template),  # Store template in uri field
            uri_template=uri_template,
            name=func_name,
            description=description or fn.__doc__ or "",
            mime_type=mime_type or "text/plain",
            fn=fn,
            parameters=params,
        )

    def matches(self, uri: str) -> Optional[Dict[str, str]]:
        """Check if a URI matches this template pattern."""
        pattern = self.uri_template.replace("{", "(?P<").replace("}", ">[^/]+)")
        match = re.match(f"^{pattern}$", uri)
        return match.groupdict() if match else None

    async def create_resource(self, uri: str, params: Dict[str, str]) -> Resource:
        """Create a concrete resource from this template."""
        try:
            # Call the function with extracted parameters
            result = self.fn(**params)
            if inspect.iscoroutine(result):
                result = await result

            # Create a FunctionResource that wraps the result
            from .types import FunctionResource

            return FunctionResource(
                uri=AnyUrl(uri),  # Convert string to AnyUrl
                name=self.name,
                description=self.description,
                mime_type=self.mime_type,
                fn=lambda: result,
            )

        except Exception as e:
            logger.error(f"Error creating resource from template: {e}")
            raise ValueError(f"Error creating resource from template: {e}")

    # Update return type to match base class
    async def read(self) -> Union[str, bytes]:
        """Read content with template parameters."""
        try:
            # Extract parameters from the URI
            params = self.matches(str(self.uri))
            if not params:
                raise ValueError(
                    f"URI {self.uri} does not match template {self.uri_template}"
                )

            # Create concrete resource and read it
            resource = await self.create_resource(str(self.uri), params)
            return await resource.read()  # Now correct since we match the return type

        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            raise ValueError(f"Template rendering error: {e}")


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

        if self.metadata.compression:
            if isinstance(content, str):
                content = content.encode(self.metadata.encoding or "utf-8")
            content = decompress_content(content, self.metadata.compression)

        if self.mime_type.startswith("text/"):
            if isinstance(content, (bytes | bytearray | memoryview)):
                content = bytes(content)
                return content.decode(self.metadata.encoding or "utf-8")
            return content
        if isinstance(content, str):
            return content.encode(self.metadata.encoding or "utf-8")
        return bytes(content)

    async def write(self, content: str | bytes) -> None:
        """Compress and write content."""
        if isinstance(content, str):
            content = content.encode(self.metadata.encoding or "utf-8")

        if len(content) > self.config.min_size:
            try:
                compressed = compress_content(content, self.config.algorithm)
                self.update_metadata(compression=self.config.algorithm)
                await self.wrapped_resource.write(compressed)
            except Exception:
                # If compression fails, write uncompressed
                self.update_metadata(compression=None)
                await self.wrapped_resource.write(content)
        else:
            # For small content, don't compress
            self.update_metadata(compression=None)
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

    # Try the specified encoding first
    try:
        return content.decode(encoding)
    except UnicodeDecodeError:
        # If that fails, try common encodings
        for enc in ["utf-8", "latin1", "ascii", "utf-16", "utf-32"]:
            if enc != encoding:
                try:
                    return content.decode(enc)
                except UnicodeDecodeError:
                    continue

        # If all decodings fail, use latin1 as a last resort
        # (latin1 never fails as it maps bytes directly to Unicode points 0-255)
        return content.decode("latin1")


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

        # Apply each transformation
        for transform in self.pipeline.steps:
            content = transform(content)

        # If the content is bytes and we expect text, decode it
        # But don't decode if it's compressed
        # which is indicated by non-printable characters
        if isinstance(content, bytes) and self.mime_type.startswith("text/"):
            try:
                # Try to decode as UTF-8 - if it fails, it's likely compressed
                content.decode("utf-8")
                return safe_decode(content)
            except UnicodeDecodeError:
                # Content is compressed or binary, return as bytes
                return content

        return content

    async def write(self, content: str | bytes) -> None:
        """Transform and write content through pipeline."""

        # Special case for JSON pipeline in reverse mode
        if (
            self.reverse_pipeline
            and self.pipeline.name == "json_processing"
            and isinstance(content, str)
        ):
            # If we're dealing with the JSON pipeline in reverse, we just need to
            # parse the content to validate it's proper JSON and then write it directly
            try:
                parsed = json.loads(content)
                formatted_json = json.dumps(parsed)
                await self.wrapped_resource.write(formatted_json)
                return
            except json.JSONDecodeError:
                pass

        # Handle normal cases
        if self.reverse_pipeline:
            steps = list(reversed(self.pipeline.steps))
            for transform in steps:
                content = transform(content)
        else:
            for transform in self.pipeline.steps:
                content = transform(content)

        await self.wrapped_resource.write(content)


# Common transformation functions
def text_transform(
    fn: Callable[[str], str],
) -> Callable[[str | bytes], bytes]:
    """Create a text-based transformation."""

    def wrapper(content: str | bytes) -> bytes:
        # Convert to string if needed
        if isinstance(content, (bytes | bytearray | memoryview)):
            try:
                content = safe_decode(content)
            except UnicodeDecodeError:
                # If we can't decode, we might be dealing
                # with already compressed content
                # Just return it as is for compression steps to handle
                if isinstance(content, (bytearray | memoryview)):
                    return bytes(content)
                return content
        result = fn(str(content))
        # Always return bytes
        return safe_encode(result)

    return wrapper


def bytes_transform(
    fn: Callable[[bytes], bytes],
) -> Callable[[str | bytes], str | bytes]:
    """Create a bytes-based transformation."""

    def wrapper(content: str | bytes) -> str | bytes:
        if isinstance(content, str):
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
