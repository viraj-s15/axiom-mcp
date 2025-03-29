"""Utility functions for resource management."""

import hashlib
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from .base import Resource, ResourceType

logger = logging.getLogger(__name__)

# Constants for identifier validation
MAX_IDENTIFIER_LENGTH = 64
IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_identifier(identifier: str) -> str:
    """Validate a resource identifier.

    Args:
        identifier: The identifier to validate

    Returns:
        The validated identifier

    Raises:
        TypeError: If identifier is not a string
        ValueError: If identifier is invalid
    """
    if not isinstance(identifier, str):
        raise TypeError("Identifier must be a string")

    if not identifier:
        raise ValueError("Identifier cannot be empty")

    if len(identifier) > MAX_IDENTIFIER_LENGTH:
        raise ValueError("Identifier exceeds maximum length")

    if identifier[0].isdigit():
        raise ValueError("Identifier cannot start with a number")

    if not IDENTIFIER_PATTERN.match(identifier):
        raise ValueError("Identifier contains invalid characters")

    return identifier


class ResourceFilter(Protocol):
    """Type protocol for resource filter functions."""

    def __call__(self, resource: Resource) -> bool: ...


def guess_resource_type(uri: str, content: str | bytes | None = None) -> ResourceType:
    """Guess the appropriate resource type from URI and optional content."""
    parsed = urlparse(uri)

    # Check scheme-based resources
    scheme_mapping = {
        "http": ResourceType.HTTP,
        "https": ResourceType.HTTP,
        "file": ResourceType.FILE,
        "data": ResourceType.BINARY,
        "stream": ResourceType.STREAM,
        "function": ResourceType.FUNCTION,
        "template": ResourceType.TEMPLATE,
    }

    if parsed.scheme in scheme_mapping:
        return scheme_mapping[parsed.scheme]

    # Check if URI has template parameters (e.g., {param})
    if "{" in uri and "}" in uri:
        return ResourceType.TEMPLATE

    # Check content-based type
    if content is not None:
        return ResourceType.BINARY if isinstance(content, bytes) else ResourceType.TEXT

    # Check file extension
    if "." in parsed.path:
        mime_type, _ = mimetypes.guess_type(parsed.path)
        if mime_type:
            # Handle common binary formats
            if mime_type.startswith(
                ("application/", "image/", "audio/", "video/")
            ) or parsed.path.lower().endswith(
                (".bin", ".exe", ".dll", ".so", ".dylib")
            ):
                return ResourceType.BINARY
            if mime_type.startswith("text/"):
                return ResourceType.TEXT

    return ResourceType.TEXT


def create_resource_uri(
    path: str | Path,
    scheme: str | None = None,
    resource_type: ResourceType | None = None,
) -> str:
    """Create a properly formatted resource URI."""
    from urllib.parse import quote

    if isinstance(path, Path):
        path = str(path.absolute())

    # Convert resource_type to string scheme if provided
    scheme = (
        str(resource_type.name.lower()) if resource_type else (scheme or "resource")
    )

    if "://" in path:
        return path

    # For template URIs, don't encode the template parameters
    if resource_type == ResourceType.TEMPLATE or (scheme and scheme == "template"):
        # We still need to encode other special characters, just not { and }
        parts = []
        current = ""
        i = 0
        while i < len(path):
            if path[i : i + 1] == "{":
                if current:
                    parts.append(quote(current))
                    current = ""
                template_part = ""
                i += 1
                while i < len(path) and path[i : i + 1] != "}":
                    template_part += path[i]
                    i += 1
                if i < len(path):
                    parts.append("{" + template_part + "}")
                i += 1
            else:
                current += path[i]
                i += 1
        if current:
            parts.append(quote(current))
        encoded_path = "".join(parts)
    else:
        encoded_path = quote(path)

    return f"{scheme}://{encoded_path}"


def calculate_content_hash(content: str | bytes) -> str:
    """Calculate SHA-256 hash of content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def safe_decode(content: bytes | bytearray | memoryview) -> bytes:
    """Convert various binary types to bytes."""
    if isinstance(content, bytearray | memoryview):
        return bytes(content)
    return content


COMMON_ENCODINGS = [
    "utf-8",
    "ascii",
    "iso-8859-1",
    "windows-1252",
    "utf-16",
    "utf-32",
]


def detect_encoding(content: bytes | bytearray | memoryview | str) -> str | None:
    """Attempt to detect the encoding of content using common encodings."""
    try:
        if isinstance(content, str):
            return "utf-8"

        # Convert to bytes if needed
        content = safe_decode(content)

        # Try each encoding in order
        for encoding in COMMON_ENCODINGS:
            try:
                content.decode(encoding)
            except UnicodeDecodeError:
                continue
            else:
                return encoding

    except Exception:
        logger.exception("Error detecting encoding")

    return None


def create_resource_filter(**criteria: Any) -> ResourceFilter:
    """Create a filter function for resources based on criteria."""

    def filter_resource(resource: Resource) -> bool:
        for key, value in criteria.items():
            if key == "mime_type" and not resource.mime_type.startswith(value):
                return False

            if (
                key == "min_size"
                and resource.metadata.size
                and resource.metadata.size < value
            ):
                return False

            if (
                key == "max_size"
                and resource.metadata.size
                and resource.metadata.size > value
            ):
                return False

            if key == "has_attribute" and value not in resource.metadata.attributes:
                return False

            if key == "resource_type" and resource.resource_type != value:
                return False

        return True

    return filter_resource
