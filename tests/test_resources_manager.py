"""Tests for resource manager functionality."""

import asyncio

import pytest
from pydantic import AnyUrl

from axiom_mcp.exceptions import ResourceError
from axiom_mcp.resources.advanced_types import TemplateResource
from axiom_mcp.resources.base import Resource
from axiom_mcp.resources.manager import ResourceManager
from axiom_mcp.resources.types import TextResource


@pytest.fixture
def manager():
    """Create a test resource manager."""
    return ResourceManager(warn_on_duplicate_resources=True)


@pytest.fixture
def text_resource():
    """Create a sample text resource."""
    return TextResource(
        uri=AnyUrl("memory://test/text"),
        text="Test content",
        name="test_text",
    )


@pytest.fixture
def template_resource():
    """Create a sample template resource."""

    def template_fn(param: str) -> str:
        return f"Template with {param}"

    return TemplateResource(
        uri=AnyUrl("template://test/{param}"),
        uri_template="template://test/{param}",
        name="test_template",
        fn=template_fn,
    )


@pytest.mark.asyncio
async def test_add_resource(manager: ResourceManager, text_resource: Resource):
    """Test adding a resource to the manager."""
    added = await manager.add_resource(text_resource)
    assert added == text_resource
    assert str(text_resource.uri) in manager._resources


@pytest.mark.asyncio
async def test_duplicate_resource_warning(
    manager: ResourceManager, text_resource: Resource
):
    """Test warning behavior for duplicate resources."""
    await manager.add_resource(text_resource)
    duplicate = await manager.add_resource(text_resource)
    assert duplicate == text_resource


@pytest.mark.asyncio
async def test_get_resource(manager: ResourceManager, text_resource: Resource):
    """Test retrieving a resource."""
    await manager.add_resource(text_resource)
    retrieved = await manager.get_resource(text_resource.uri)
    assert retrieved == text_resource


@pytest.mark.asyncio
async def test_get_nonexistent_resource(manager: ResourceManager):
    """Test retrieving a nonexistent resource."""
    with pytest.raises(ResourceError, match="Resource not found"):
        await manager.get_resource("memory://nonexistent")


@pytest.mark.asyncio
async def test_list_resources(
    manager: ResourceManager, text_resource: Resource, template_resource: Resource
):
    """Test listing all resources."""
    await manager.add_resource(text_resource)
    await manager.add_resource(template_resource)
    resources = manager.list_resources()
    assert len(resources) == 2
    assert text_resource in resources
    assert template_resource in resources


@pytest.mark.asyncio
async def test_list_templates(
    manager: ResourceManager, template_resource: TemplateResource
):
    """Test listing template resources."""
    await manager.add_resource(template_resource)
    templates = manager.list_templates()
    assert len(templates) == 1
    assert template_resource in templates


@pytest.mark.asyncio
async def test_read_resource(manager: ResourceManager, text_resource: Resource):
    """Test reading from a resource."""
    await manager.add_resource(text_resource)
    content = await manager.read_resource(text_resource.uri)
    assert content == "Test content"


@pytest.mark.asyncio
async def test_read_nonexistent_resource(manager: ResourceManager):
    """Test reading from a nonexistent resource."""
    with pytest.raises(ResourceError, match="Resource not found"):
        await manager.read_resource("memory://nonexistent")


@pytest.mark.asyncio
async def test_template_matching(
    manager: ResourceManager, template_resource: TemplateResource
):
    """Test template resource matching and instantiation."""
    await manager.add_resource(template_resource)

    # Test matching URI
    test_uri = "template://test/value"
    resource = await manager.get_resource(test_uri)
    assert resource is not None
    content = await resource.read()
    assert content == "Template with value"


@pytest.mark.asyncio
async def test_template_no_match(
    manager: ResourceManager, template_resource: TemplateResource
):
    """Test template resource with non-matching URI."""
    await manager.add_resource(template_resource)

    with pytest.raises(ResourceError, match="Resource not found"):
        await manager.get_resource("template://different/value")


@pytest.mark.asyncio
async def test_resource_cleanup(manager: ResourceManager, text_resource: Resource):
    """Test resource cleanup on manager shutdown."""
    await manager.add_resource(text_resource)
    manager._resources.clear()
    # Verify resources are cleared
    assert len(manager._resources) == 0


@pytest.mark.asyncio
async def test_concurrent_resource_access(
    manager: ResourceManager, text_resource: Resource
):
    """Test concurrent access to resources."""
    await manager.add_resource(text_resource)

    async def read_resource():
        return await manager.read_resource(text_resource.uri)

    # Create multiple concurrent read operations
    tasks = [read_resource() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # Verify all reads were successful
    assert all(result == "Test content" for result in results)
