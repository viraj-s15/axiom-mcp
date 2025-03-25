"""Utility functions and classes for tool management."""

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

try:
    import aiofiles
except ImportError:
    aiofiles = None  # type: ignore

from opentelemetry import trace  # type: ignore
from pydantic import BaseModel, Field, ValidationError

from ..exceptions import ToolError
from .base import Tool, ToolMetadata

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")
RT = TypeVar("RT", bound=Awaitable[Any])


def validate_schema(schema: dict[str, Any], data: Any) -> None:
    """Validate data against a JSON schema."""
    try:
        if isinstance(schema, dict):
            # Create a dynamic Pydantic model for validation
            model = type("DynamicModel", (BaseModel,), {"__annotations__": schema})
            if isinstance(data, dict):
                model(**data)
            else:
                model(value=data)
    except ValidationError as e:
        raise ToolError(f"Schema validation failed: {str(e)}") from e


async def ensure_async_wrapper(
    func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> R:
    """Helper function to ensure async execution."""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def ensure_async(func: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    """Ensure a function is async."""
    if asyncio.iscoroutinefunction(func):
        return func  # type: ignore

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def tool(
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    version: str = "1.0.0",
    author: str | None = None,
    examples: list[dict[str, Any]] | None = None,
    requires_auth: bool = False,
    rate_limit: int | None = None,
    cached: bool = False,
    cache_ttl: int = 300,
) -> Callable[[type[Tool]], type[Tool]]:
    """Decorator to create a tool from a class."""

    def decorator(cls: type[Tool]) -> type[Tool]:
        metadata = ToolMetadata(
            name=name or cls.__name__,
            version=version,
            description=description or cls.__doc__ or "",
            author=author,
            validation=None,
        )
        cls.metadata = metadata  # type: ignore
        return cls

    return decorator


def traced_tool(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Decorator to add OpenTelemetry tracing to a tool."""
    if not asyncio.iscoroutinefunction(fn):
        fn = ensure_async(fn)  # type: ignore

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with tracer.start_as_current_span(
            f"tool.{fn.__name__}",
            attributes={
                "tool.name": fn.__name__,
                "tool.args": str(args),
                "tool.kwargs": str(kwargs),
            },
        ) as span:
            try:
                result = await fn(*args, **kwargs)
                span.set_attribute("tool.success", True)
                return result
            except Exception as e:
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.error", str(e))
                raise

    return wrapper


def batch_tool(
    size: int = 10,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[list[R]]]]:
    """Decorator to convert a single-item tool into a batch processing tool."""

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[list[R]]]:
        if not asyncio.iscoroutinefunction(fn):
            fn = cast(Callable[P, Awaitable[R]], ensure_async(fn))

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> list[R]:
            batch_input = kwargs.pop("batch_input", [])
            if not isinstance(batch_input, Sequence):
                result = await fn(*args, **kwargs)
                return [result]

            results = []
            for i in range(0, len(batch_input), size):
                batch = batch_input[i : i + size]
                batch_results = await asyncio.gather(
                    *(fn(*args, **{**kwargs, **item}) for item in batch)
                )
                results.extend(batch_results)
            return results

        return wrapper

    return decorator


def with_validation(
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Callable[
    [Callable[Concatenate[Tool, P], Awaitable[R]]],
    Callable[Concatenate[Tool, P], Awaitable[R]],
]:
    """Decorator to add schema validation to a tool method."""

    def decorator(
        func: Callable[Concatenate[Tool, P], Awaitable[R]],
    ) -> Callable[Concatenate[Tool, P], Awaitable[R]]:
        if not asyncio.iscoroutinefunction(func):
            func = cast(
                Callable[Concatenate[Tool, P], Awaitable[R]], ensure_async(func)
            )

        @wraps(func)
        async def wrapper(self: Tool, *args: P.args, **kwargs: P.kwargs) -> R:
            if input_schema and self.context.validation_enabled:
                validate_schema(input_schema, kwargs)

            result = await func(self, *args, **kwargs)

            if output_schema and self.context.validation_enabled:
                validate_schema(output_schema, result)

            return result

        return wrapper

    return decorator


def make_streamable(
    chunk_size: int = 8192,
) -> Callable[
    [Callable[Concatenate[Tool, P], Awaitable[Any]]],
    Callable[Concatenate[Tool, P], AsyncGenerator[Any, None]],
]:
    """Decorator to make a tool method streamable."""

    def decorator(
        func: Callable[Concatenate[Tool, P], Awaitable[Any]],
    ) -> Callable[Concatenate[Tool, P], AsyncGenerator[Any, None]]:
        if not asyncio.iscoroutinefunction(func):
            func = ensure_async(func)

        @wraps(func)
        async def wrapper(
            self: Tool, *args: P.args, **kwargs: P.kwargs
        ) -> AsyncGenerator[Any, None]:
            result = await func(self, *args, **kwargs)

            if isinstance(result, (str, bytes)):
                # Stream string/bytes in chunks
                for i in range(0, len(result), chunk_size):
                    yield result[i + chunk_size]
            elif isinstance(result, AsyncIterator):
                # Pass through async iterators
                async for chunk in result:
                    yield chunk
            else:
                # For other types, yield the entire result
                yield result

        return wrapper

    return decorator


def cacheable(
    ttl: int = 3600,
    key_func: Callable[[Tool, tuple, dict], str] | None = None,
) -> Callable[
    [Callable[Concatenate[Tool, P], Awaitable[R]]],
    Callable[Concatenate[Tool, P], Awaitable[R]],
]:
    """Decorator to make a tool method cacheable."""
    cache: dict[str, tuple[R, float]] = {}

    def decorator(
        func: Callable[Concatenate[Tool, P], Awaitable[R]],
    ) -> Callable[Concatenate[Tool, P], Awaitable[R]]:
        if not asyncio.iscoroutinefunction(func):
            func = cast(
                Callable[Concatenate[Tool, P], Awaitable[R]], ensure_async(func)
            )

        @wraps(func)
        async def wrapper(self: Tool, *args: P.args, **kwargs: P.kwargs) -> R:
            if not self.context.cache_enabled:
                return await func(self, *args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(self, args, kwargs)
            else:
                key_parts = [
                    self.metadata.name,
                    self.metadata.version,
                    json.dumps(args, sort_keys=True),
                    json.dumps(kwargs, sort_keys=True),
                    json.dumps(self.context.model_dump(), sort_keys=True),
                ]
                cache_key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()

            # Check cache
            current_time = datetime.now(UTC).timestamp()
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp <= ttl:
                    return result

            # Execute function and cache result
            result = await func(self, *args, **kwargs)
            cache[cache_key] = (result, current_time)
            return result

        return wrapper

    return decorator


def with_dry_run(
    simulation_func: Callable[Concatenate[Tool, P], Awaitable[Any]] | None = None,
) -> Callable[
    [Callable[Concatenate[Tool, P], Awaitable[R]]],
    Callable[Concatenate[Tool, P], Awaitable[R]],
]:
    """Decorator to add dry run support to a tool method."""

    def decorator(
        func: Callable[Concatenate[Tool, P], Awaitable[R]],
    ) -> Callable[Concatenate[Tool, P], Awaitable[R]]:
        if not asyncio.iscoroutinefunction(func):
            func = cast(
                Callable[Concatenate[Tool, P], Awaitable[R]], ensure_async(func)
            )

        @wraps(func)
        async def wrapper(self: Tool, *args: P.args, **kwargs: P.kwargs) -> R:
            if self.context.dry_run:
                if simulation_func:
                    # type: ignore
                    return await simulation_func(self, *args, **kwargs)
                return cast(
                    R,
                    {
                        "simulated": True,
                        "tool": self.metadata.name,
                        "args": args,
                        "kwargs": kwargs,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


@contextlib.asynccontextmanager
async def atomic_write(path: Path) -> AsyncGenerator[Path, None]:
    """Context manager for atomic file writes using temporary files."""
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        yield temp_path
        if temp_path.exists():
            temp_path.rename(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


class FileSystemTool(Tool):
    """Base class for filesystem operation tools."""

    class Config(BaseModel):
        """Configuration for filesystem tools."""

        root_path: Path = Field(default_factory=Path.cwd)
        allowed_extensions: set[str] = Field(default_factory=set)
        max_file_size: int = Field(default=10 * 1024 * 1024)  # 10MB
        create_dirs: bool = Field(default=True)

    config: Config = Field(default_factory=Config)

    def _validate_path(self, path: Path | str) -> Path:
        """Validate and normalize a file path."""
        path = Path(path) if isinstance(path, str) else path
        if not path.is_absolute():
            path = self.config.root_path / path

        # Security check - ensure path is within root
        try:
            path.relative_to(self.config.root_path)
        except ValueError:
            raise ValueError(f"Path {path} is outside root directory")

        return path

    def _validate_file(self, path: Path) -> None:
        """Validate a file meets configuration requirements."""
        if (
            self.config.allowed_extensions
            and path.suffix.lower() not in self.config.allowed_extensions
        ):
            raise ValueError(f"File extension {path.suffix} not allowed")

        if path.exists() and path.stat().st_size > self.config.max_file_size:
            raise ValueError(f"File {path} exceeds maximum size")


@tool(
    name="read_file",
    description="Read a file with UTF-8 or specified encoding",
    tags=["filesystem", "io"],
    examples=[{"path": "example.txt", "encoding": "utf-8"}],
)
class ReadFileTool(FileSystemTool):
    """Tool for reading files."""

    async def execute(self, args: dict[str, Any]) -> str:
        """Read file content."""
        path = self._validate_path(args["path"])
        self._validate_file(path)
        encoding = args.get("encoding", "utf-8")

        if aiofiles is None:
            # Fallback to synchronous file operations if aiofiles is not available
            try:
                with open(path, encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                encoding = "latin1"
                with open(path, encoding=encoding) as f:
                    return f.read()
        else:
            try:
                async with aiofiles.open(path, encoding=encoding) as f:
                    return await f.read()
            except UnicodeDecodeError:
                encoding = "latin1"
                async with aiofiles.open(path, encoding=encoding) as f:
                    return await f.read()


@tool(
    name="write_file",
    description="Write content to a file atomically",
    tags=["filesystem", "io"],
)
class WriteFileTool(FileSystemTool):
    """Tool for atomic file writes."""

    async def execute(self, args: dict[str, Any]) -> None:
        """Write content to file atomically."""
        path = self._validate_path(args["path"])
        content = args["content"]
        if isinstance(content, str):
            content = content.encode("utf-8")

        self._validate_file(path)
        if not path.parent.exists() and self.config.create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        async with atomic_write(path) as temp_path:
            if aiofiles is None:
                # Fallback to synchronous file operations
                with open(temp_path, "wb") as f:
                    f.write(content)
            else:
                async with aiofiles.open(temp_path, "wb") as f:
                    await f.write(content)
