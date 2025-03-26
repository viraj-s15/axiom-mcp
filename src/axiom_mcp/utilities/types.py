"""Type definitions and utilities for AxiomMCP."""

import base64
from typing import Literal

from mcp.types import ImageContent
from pydantic import BaseModel, Field


class Image(BaseModel):
    """Represents an image with metadata."""

    data: bytes = Field(description="Raw image data")
    mime_type: str = Field(description="MIME type of the image")
    encoding: Literal["base64"] = Field(
        default="base64", description="Encoding used for the image data"
    )

    def to_image_content(self) -> ImageContent:
        """Convert to MCP ImageContent type."""
        return ImageContent(
            type="image",
            mimeType=self.mime_type,
            data=base64.b64encode(self.data).decode(),
        )
