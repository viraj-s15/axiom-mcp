"""Tests for the tool registry system."""

import pytest

from axiom_mcp.tools.base import Tool, ToolDependency, ToolMetadata
from axiom_mcp.tools.registry import ToolRegistry

# Test fixtures


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return ToolRegistry()


@pytest.fixture
def sample_tool():
    """Create a sample tool class for testing."""

    class TestTool(Tool):
        metadata = ToolMetadata(
            name="test_tool",
            version="1.0.0",
            description="A test tool",
            dependencies=[
                ToolDependency(name="dep_tool", version="1.0.0", command="dep_command")
            ],
        )

        async def execute(self, args):
            return {"status": "success"}

        def run(self, **kwargs):
            """Run method that doesn't create an unawaited coroutine."""
            return None

    return TestTool


@pytest.fixture
def dependent_tool():
    """Create a tool that depends on the sample tool."""

    class DependentTool(Tool):
        metadata = ToolMetadata(
            name="dependent_tool",
            version="1.0.0",
            description="A dependent tool",
            dependencies=[
                ToolDependency(
                    name="test_tool", version="1.0.0", command="test_command"
                )
            ],
        )

        async def execute(self, args):
            return {"status": "success"}

    return DependentTool


def test_register_tool(registry, sample_tool):
    """Test tool registration."""
    registry.register_tool(sample_tool)
    assert "test_tool" in registry._tools
    assert len(registry._dependencies["test_tool"]) == 1
    assert "dep_tool" in registry._dependencies["test_tool"]


def test_unregister_tool(registry, sample_tool, dependent_tool):
    """Test tool unregistration."""
    registry.register_tool(sample_tool)
    registry.register_tool(dependent_tool)

    registry.unregister_tool("test_tool")
    assert "test_tool" not in registry._tools
    assert "test_tool" not in registry._dependencies
    assert "test_tool" not in registry._reverse_deps


def test_get_tool(registry, sample_tool):
    """Test retrieving a tool."""
    registry.register_tool(sample_tool)
    tool_cls = registry.get_tool("test_tool")
    assert tool_cls == sample_tool


def test_list_tools(registry, sample_tool, dependent_tool):
    """Test listing all registered tools."""
    registry.register_tool(sample_tool)
    registry.register_tool(dependent_tool)
    tools = registry.list_tools()
    assert len(tools) == 2
    assert any(t.name == "test_tool" for t in tools)
    assert any(t.name == "dependent_tool" for t in tools)


def test_dependency_tracking(registry, sample_tool, dependent_tool):
    """Test dependency tracking system."""
    registry.register_tool(sample_tool)
    registry.register_tool(dependent_tool)

    # Check dependencies
    deps = registry.get_dependencies("dependent_tool")
    assert "test_tool" in deps

    # Check reverse dependencies
    rev_deps = registry.get_dependents("test_tool")
    assert "dependent_tool" in rev_deps


def test_check_dependencies(registry, sample_tool, dependent_tool):
    """Test dependency checking."""
    registry.register_tool(dependent_tool)
    missing = registry.check_dependencies("dependent_tool")
    assert "test_tool" in missing

    registry.register_tool(sample_tool)
    missing = registry.check_dependencies("dependent_tool")
    assert not missing


def test_plugin_path_management(registry, tmp_path):
    """Test plugin path management."""
    test_path = tmp_path / "plugins"
    test_path.mkdir()

    registry.add_plugin_path(test_path)
    assert test_path in registry._plugin_paths


def test_validate_dependencies(registry, sample_tool, dependent_tool):
    """Test dependency validation."""
    registry.register_tool(dependent_tool)
    invalid_deps = registry.validate_dependencies()
    assert "dependent_tool" in invalid_deps
    assert "test_tool" in invalid_deps["dependent_tool"]

    registry.register_tool(sample_tool)
    invalid_deps = registry.validate_dependencies()
    assert "dependent_tool" not in invalid_deps
