"""Tests for core resource implementations."""

import tempfile
from contextlib import suppress
from pathlib import Path

import pytest
from pydantic import AnyUrl

from axiom_mcp.resources.types import (
    BinaryResource,
    FileResource,
    TextResource,
)


@pytest.fixture
def text_content():
    return "Hello, World!"


@pytest.fixture
def binary_content():
    return b"Hello, Binary World!"


@pytest.fixture
def text_resource(text_content):
    return TextResource(
        uri=AnyUrl("memory://test/text"), text=text_content, name="test_text"
    )


@pytest.fixture
def binary_resource(binary_content):
    return BinaryResource(
        uri=AnyUrl("memory://test/binary"), data=binary_content, name="test_binary"
    )


@pytest.fixture
def temp_file():

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"Hello, File World!")
        path = Path(f.name)
    yield path
    with suppress(FileNotFoundError):
        Path.unlink(path)  # File might have been already deleted by the test


@pytest.fixture
async def file_resource(temp_file):
    """Create a test file resource."""
    resource = FileResource(
        uri=AnyUrl(f"file://{temp_file}"), path=temp_file, name="test_file"
    )
    await resource._ensure_initialized()
    return resource


@pytest.mark.asyncio
class TestTextResource:
    async def test_read(self, text_resource, text_content):
        """Test reading from a text resource."""
        content = await text_resource.read()
        assert content == text_content
        assert isinstance(content, str)

    async def test_write(self, text_resource):
        """Test writing to a text resource."""
        new_content = "Updated content"
        await text_resource.write(new_content)
        content = await text_resource.read()
        assert content == new_content

    async def test_write_bytes(self, text_resource):
        """Test writing bytes to a text resource."""
        new_content = b"Updated bytes content"
        await text_resource.write(new_content)
        content = await text_resource.read()
        assert content == new_content.decode()


@pytest.mark.asyncio
class TestBinaryResource:
    async def test_read(self, binary_resource, binary_content):
        """Test reading from a binary resource."""
        content = await binary_resource.read()
        assert content == binary_content
        assert isinstance(content, bytes)

    async def test_write(self, binary_resource):
        """Test writing to a binary resource."""
        new_content = b"Updated binary content"
        await binary_resource.write(new_content)
        content = await binary_resource.read()
        assert content == new_content

    async def test_write_str(self, binary_resource):
        """Test writing string to a binary resource."""
        new_content = "Updated string content"
        await binary_resource.write(new_content)
        content = await binary_resource.read()
        assert content == new_content.encode()


@pytest.mark.asyncio
class TestFileResource:
    async def test_read(self, file_resource):
        """Test reading from a file resource."""
        content = await file_resource.read()
        # FileResource.read() returns bytes, so we should compare with bytes
        assert content == b"Hello, File World!"

    async def test_write(self, file_resource):
        """Test writing to a file resource."""
        new_content = b"Updated file content"
        await file_resource.write(new_content)
        content = await file_resource.read()
        assert content == new_content

    async def test_write_str(self, file_resource):
        """Test writing string to a file resource."""
        new_content = "Updated string content"
        await file_resource.write(new_content)
        content = await file_resource.read()
        assert content == new_content.encode()

    async def test_read_stream(self, file_resource):
        """Test streaming from a file resource."""
        chunks = []
        async for chunk in file_resource.read_stream():
            chunks.append(chunk)
        assert b"".join(chunks) == b"Hello, File World!"

    async def test_delete(self, file_resource):
        """Test deleting a file resource."""
        assert file_resource.path.exists()
        await file_resource.delete()
        assert not file_resource.path.exists()

    async def test_exists(self, file_resource, temp_file):
        """Test exists check for file resource."""
        assert await file_resource.exists() is True
        # Use the resource's delete method instead of direct file deletion
        await file_resource.delete()
        assert await file_resource.exists() is False
