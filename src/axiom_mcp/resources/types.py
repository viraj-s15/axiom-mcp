"""Core resource implementations."""

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import cachetools
from aiohttp import ClientTimeout
from pydantic import Field

from ..exceptions import ResourceReadError
from .base import Resource, ResourceType


class TextResource(Resource):
    """A resource that reads from a string."""

    text: str = Field(description="Text content of the resource")
    resource_type: ResourceType = ResourceType.TEXT

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Initialize metadata for the resource."""
        content = self.text.encode("utf-8")
        self.update_metadata(
            size=len(content),
            content_hash=hashlib.sha256(content).hexdigest(),
            encoding="utf-8",
        )

    async def read(self) -> str:
        """Read the text content."""
        return self.text

    async def write(self, content: str | bytes) -> None:
        """Write new content to the resource."""
        if isinstance(content, bytes):
            content = content.decode(self.metadata.encoding or "utf-8")
        # Use object.__setattr__ to bypass Pydantic's immutability
        object.__setattr__(self, "text", str(content))
        self._initialize_metadata()


class BinaryResource(Resource):
    """A resource that reads from bytes."""

    data: bytes = Field(description="Binary content of the resource")
    resource_type: ResourceType = ResourceType.BINARY

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Initialize metadata for the resource."""
        self.update_metadata(
            size=len(self.data), content_hash=hashlib.sha256(self.data).hexdigest()
        )

    async def read(self) -> bytes:
        """Read the binary content."""
        return self.data

    async def write(self, content: str | bytes) -> None:
        """Write new content to the resource."""
        if isinstance(content, str):
            content = content.encode(self.metadata.encoding or "utf-8")
        object.__setattr__(self, "data", bytes(content))
        self._initialize_metadata()


class FileResource(Resource):
    """A file resource with streaming support."""

    path: Path = Field(description="Path to the file")
    chunk_size: int = Field(default=8192, description="Chunk size for streaming")
    resource_type: ResourceType = ResourceType.FILE

    def __init__(self, **data):
        super().__init__(**data)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure metadata is initialized."""
        if not self._initialized:
            await self._initialize_metadata()
            self._initialized = True

    async def _initialize_metadata(self) -> None:
        """Initialize metadata for the resource."""
        try:
            if self.path.exists():
                stats = self.path.stat()
                self.update_metadata(
                    size=stats.st_size,
                    modified_at=datetime.fromtimestamp(stats.st_mtime, tz=UTC),
                )
        except (FileNotFoundError, PermissionError):
            # Ignore errors during initialization
            pass

    async def read(self) -> bytes:
        """Read the file content."""
        await self._ensure_initialized()
        try:
            async with aiofiles.open(self.path, mode="rb") as f:
                content = await f.read()

            # Update metadata after successful read
            stats = self.path.stat()
            self.update_metadata(
                size=stats.st_size,
                modified_at=datetime.fromtimestamp(stats.st_mtime, tz=UTC),
            )
            # Always return bytes for consistency
            if isinstance(content, str):
                return content.encode(self.metadata.encoding or "utf-8")
            return content
        except FileNotFoundError:
            # Create parent directory if it doesn't exist
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Return empty content for new files
            return b""

    async def read_stream(self) -> AsyncGenerator[str | bytes, None]:
        """Stream the file content."""
        await self._ensure_initialized()
        try:
            async with aiofiles.open(self.path, mode="rb") as f:
                while chunk := await f.read(self.chunk_size):
                    yield chunk
        except FileNotFoundError:
            # Create parent directory if it doesn't exist
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Yield empty content for new files
            yield b""

    async def write(self, content: str | bytes) -> None:
        """Write content to the file."""
        await self._ensure_initialized()
        # Create parent directory if it doesn't exist
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, str):
            async with aiofiles.open(
                self.path, mode="w", encoding=self.metadata.encoding or "utf-8"
            ) as f:
                await f.write(content)
                await f.flush()
        else:
            async with aiofiles.open(self.path, mode="wb") as f:
                await f.write(content)
                await f.flush()

        await self._initialize_metadata()

    async def delete(self) -> None:
        """Delete the file."""
        await self._ensure_initialized()
        with suppress(FileNotFoundError):
            self.path.unlink()

    async def exists(self) -> bool:
        """Check if the file exists."""
        return self.path.exists()


class HttpResource(Resource):
    """HTTP resource with caching and streaming support."""

    url: str = Field(description="URL to fetch content from")
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = Field(default=30.0)
    max_redirects: int = Field(default=5)
    resource_type: ResourceType = ResourceType.HTTP

    _cache: cachetools.TTLCache = cachetools.TTLCache(maxsize=100, ttl=3600)

    async def read(self) -> str | bytes:
        """Read the HTTP content."""
        async with self._get_http_response() as response:
            return await response.read()

    async def read_stream(self) -> AsyncGenerator[str | bytes, None]:
        """Stream the HTTP content."""
        async with self._get_http_response() as response:
            async for chunk in response.content.iter_chunked(8192):
                yield chunk

    @asynccontextmanager
    async def _get_http_response(self) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """Get HTTP response with proper session management."""
        timeout_obj = ClientTimeout(total=self.timeout)
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                self.url,
                headers=self.headers,
                timeout=timeout_obj,
                max_redirects=self.max_redirects,
            ) as response,
        ):
            response.raise_for_status()
            yield response


class FunctionResource(Resource):
    """A resource that defers data loading by wrapping a function."""

    fn: Callable[[], Any] = Field(exclude=True)
    resource_type: ResourceType = ResourceType.FUNCTION

    _cache: cachetools.TTLCache = cachetools.TTLCache(maxsize=100, ttl=3600)

    async def read(self) -> str | bytes:
        """Read by calling the wrapped function with caching support."""
        if self.metadata.attributes.get("cached"):
            cached = self._cache.get(self.uri)
            if cached is not None:
                return cached

        try:
            result = self.fn()
            if asyncio.iscoroutine(result):
                result = await result

            if isinstance(result, Resource):
                content = await result.read()
            elif isinstance(result, str | bytes):
                content = result
            else:
                content = json.dumps(result)

            if self.metadata.attributes.get("cached"):
                self._cache[self.uri] = content

            if isinstance(content, str):
                self.update_metadata(
                    size=len(content.encode(self.metadata.encoding or "utf-8")),
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                )
            else:
                self.update_metadata(
                    size=len(content),
                    content_hash=hashlib.sha256(content).hexdigest(),
                )
        except Exception as e:
            raise ResourceReadError(str(self.uri), e) from e
        else:
            return content


class StreamResource(Resource):
    """A resource that provides streaming data."""

    generator: Callable[[], AsyncGenerator[str | bytes, None]] = Field(exclude=True)
    resource_type: ResourceType = ResourceType.STREAM

    async def read(self) -> str | bytes:
        """Read all content from the stream."""
        chunks = []
        async for chunk in self.read_stream():
            chunks.append(chunk)

        if all(isinstance(c, str) for c in chunks):
            return "".join(chunks)
        return b"".join(chunks)

    async def read_stream(self) -> AsyncGenerator[str | bytes, None]:
        """Stream the content."""
        async for chunk in self.generator():
            yield chunk
