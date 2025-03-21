"""Tests for base resource functionality."""

from datetime import UTC, datetime

import pytest
from pydantic import AnyUrl, ValidationError

from axiom_mcp.resources.base import Resource, ResourceMetadata, ResourceType

# Test constants
TEST_URI = "memory://test"
TEST_NAME = "test_resource"
TEST_DESCRIPTION = "Test resource"
TEST_CONTENT = "test content"
TEST_SIZE = 100
TEST_HASH = "abc123"
TEST_ENCODING = "utf-8"
TEST_MIME_TYPE = "text/plain"


class DummyResource(Resource):
    """A simple resource implementation for testing."""

    resource_type: ResourceType = ResourceType.TEXT  # Fixed: Added type annotation

    async def read(self) -> str:
        return TEST_CONTENT


@pytest.fixture
def fixed_datetime():
    return datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def resource_metadata(fixed_datetime):
    return ResourceMetadata(
        created_at=fixed_datetime,
        modified_at=fixed_datetime,
        size=TEST_SIZE,
        content_hash=TEST_HASH,
        tags=["test"],
        attributes={"key": "value"},
        encoding=TEST_ENCODING,
        mime_type=TEST_MIME_TYPE,
    )


@pytest.fixture
def dummy_resource(resource_metadata):
    return DummyResource(
        uri=AnyUrl(TEST_URI),
        name=TEST_NAME,
        description=TEST_DESCRIPTION,
        metadata=resource_metadata,
        resource_type=ResourceType.TEXT,
    )


@pytest.mark.asyncio
async def test_resource_basic_properties(dummy_resource):
    """Test basic resource properties."""
    assert dummy_resource.name == TEST_NAME
    assert dummy_resource.description == TEST_DESCRIPTION
    assert dummy_resource.resource_type == ResourceType.TEXT
    assert dummy_resource.mime_type == TEST_MIME_TYPE


@pytest.mark.asyncio
async def test_resource_metadata(dummy_resource, resource_metadata):
    """Test resource metadata handling."""
    assert dummy_resource.metadata.size == TEST_SIZE
    assert dummy_resource.metadata.content_hash == TEST_HASH
    assert dummy_resource.metadata.tags == ["test"]
    assert dummy_resource.metadata.attributes == {"key": "value"}
    assert dummy_resource.metadata.encoding == TEST_ENCODING
    assert dummy_resource.metadata.mime_type == TEST_MIME_TYPE


@pytest.mark.asyncio
async def test_resource_read(dummy_resource):
    """Test basic resource read operation."""
    content = await dummy_resource.read()
    assert content == TEST_CONTENT


@pytest.mark.asyncio
async def test_resource_exists(dummy_resource):
    """Test resource exists method."""
    assert await dummy_resource.exists() is True


@pytest.mark.asyncio
async def test_resource_get_size(dummy_resource):
    """Test getting resource size."""
    assert dummy_resource.get_size() == TEST_SIZE


@pytest.mark.asyncio
async def test_resource_update_metadata(dummy_resource):
    """Test updating resource metadata."""
    new_size = 200
    new_hash = "new123"
    dummy_resource.update_metadata(size=new_size, content_hash=new_hash)
    assert dummy_resource.metadata.size == new_size
    assert dummy_resource.metadata.content_hash == new_hash


@pytest.mark.asyncio
async def test_resource_invalid_metadata_update(dummy_resource):
    """Test updating resource metadata with invalid values."""
    with pytest.raises(ValidationError):
        dummy_resource.update_metadata(size=-1)  # Size cannot be negative


@pytest.mark.asyncio
async def test_invalid_uri():
    """Test resource creation with invalid URI."""
    with pytest.raises(ValueError):  # AnyUrl will raise ValueError for invalid URLs
        AnyUrl("invalid://uri with spaces")  # Test the validation directly


@pytest.mark.asyncio
async def test_resource_name_validation():
    """Test resource name validation."""
    with pytest.raises(ValueError):
        DummyResource(
            uri=AnyUrl(TEST_URI),
            name="invalid name with spaces",
            resource_type=ResourceType.TEXT,
        )


@pytest.mark.asyncio
async def test_resource_stream_default_implementation(dummy_resource):
    """Test default streaming implementation."""
    chunks = []
    async for chunk in dummy_resource.read_stream():
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0] == TEST_CONTENT


@pytest.mark.asyncio
async def test_unimplemented_operations(dummy_resource):
    """Test operations that are not implemented by default."""
    with pytest.raises(NotImplementedError):
        await dummy_resource.write("test")

    with pytest.raises(NotImplementedError):
        await dummy_resource.delete()


@pytest.mark.asyncio
async def test_resource_name_from_uri():
    """Test that resource name is derived from URI if not provided."""
    resource = DummyResource(
        uri=AnyUrl("memory://test/resource.txt"), resource_type=ResourceType.TEXT
    )
    assert resource.name == "resource.txt"
