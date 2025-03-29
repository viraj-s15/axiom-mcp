"""Base classes and interfaces for AxiomMCP resources."""

import abc
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, Union, Annotated

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    UrlConstraints,
    ValidationInfo,
    field_validator,
)


class ResourceType(Enum):
    """Types of resources supported by AxiomMCP."""

    TEMPLATE = auto()
    FUNCTION = auto()
    FILE = auto()
    TEXT = auto()
    BINARY = auto()
    HTTP = auto()
    DATABASE = auto()
    CACHE = auto()
    COMPRESSED = auto()
    DIRECTORY = auto()
    STREAM = auto()  # Add STREAM type


class ResourceValidationError(ValueError):
    """Raised when resource validation fails."""

    def __init__(self) -> None:
        super().__init__("Either name or uri must be provided")


class ResourceMetadata(BaseModel):
    """Resource metadata."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    modified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    size: int | None = Field(default=None, description="Size in bytes if known", ge=0)
    content_hash: str | None = Field(
        default=None, description="Content hash if available"
    )
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    encoding: str | None = Field(default="utf-8")
    compression: str | None = Field(
        default=None, description="Compression algorithm used"
    )
    mime_type: str | None = Field(default=None, description="Content MIME type")


class Resource(BaseModel, abc.ABC):
    """Base class for all resources."""

    model_config = ConfigDict(validate_default=True)

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)] = Field(
        default=..., description="URI of the resource"
    )
    name: str | None = Field(
        description="Name of the resource",
        default=None,
        pattern=r"^[^\s]+$",  # No whitespace allowed
    )
    description: str | None = Field(
        description="Description of the resource", default=None
    )
    mime_type: str = Field(
        default="text/plain",
        description="MIME type of the resource content",
        pattern=r"^[a-zA-Z0-9]+/[a-zA-Z0-9\-+.]+$",
    )
    resource_type: ResourceType = Field(
        default=ResourceType.TEMPLATE, description="Type of resource"
    )
    metadata: ResourceMetadata = Field(default_factory=ResourceMetadata)

    @field_validator("name", mode="before")
    @classmethod
    def set_default_name(cls, name: str | None, info: ValidationInfo) -> str:
        """Set default name from URI if not provided."""
        if name:
            return name
        if uri := info.data.get("uri"):
            return str(uri).split("/")[-1]
        raise ResourceValidationError()

    @abc.abstractmethod
    async def read(self) -> Union[str, bytes]:
        """Read the resource content."""
        pass

    async def read_stream(self) -> AsyncGenerator[Union[str, bytes], None]:
        """Stream the resource content. Override for streaming support."""
        yield await self.read()

    async def write(self, content: Union[str, bytes]) -> None:
        """Write content to the resource if supported."""
        raise NotImplementedError("Write operation not supported for this resource")

    async def exists(self) -> bool:
        """Check if the resource exists."""
        try:
            await self.read()
        except Exception:  # Now catching specific base exception
            return False
        else:
            return True

    async def delete(self) -> None:
        """Delete the resource if supported."""
        raise NotImplementedError("Delete operation not supported for this resource")

    async def recover(self) -> bool:
        """Attempt to recover the resource after an error.

        Returns:
            bool: True if recovery was successful, False otherwise
        """
        try:
            # Basic recovery attempt - check if resource exists
            return await self.exists()
        except Exception:  # Now catching specific base exception
            return False

    async def close(self) -> None:
        """Close any open resources or connections.

        This method should be called when the resource is no longer needed.
        Default implementation does nothing.
        """
        pass

    def get_size(self) -> int | None:
        """Get the size of the resource if known."""
        return self.metadata.size

    def get_metadata(self) -> ResourceMetadata:
        """Get resource metadata."""
        return self.metadata

    def update_metadata(self, **kwargs: Any) -> None:
        """Update resource metadata."""
        self.metadata = ResourceMetadata(
            **{**self.metadata.model_dump(), **kwargs, "modified_at": datetime.now(UTC)}
        )
