"""Tests for enhanced tool functionality."""

import asyncio
from typing import Any, AsyncGenerator, ClassVar

import pytest

from axiom_mcp.exceptions import ToolError
from axiom_mcp.tools.base import Tool, ToolContext, ToolMetadata, ToolValidation


@pytest.fixture
def tool_manager(tmp_path):
    """Create a tool manager instance."""
    from axiom_mcp.tools.manager import ToolManager

    metrics_dir = tmp_path / "metrics"
    return ToolManager(
        cache_size=10, default_timeout=5.0, enable_metrics=True, metrics_dir=metrics_dir
    )


@pytest.fixture
def validation_tool():
    """Create a tool with input/output validation."""

    class ValidationTool(Tool):
        metadata = ToolMetadata(
            name="validation_tool",
            version="1.0.0",
            description="Tool with validation",
            validation=ToolValidation(
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                },
            ),
        )

        async def execute(self, args: dict[str, Any]) -> dict[str, str]:
            if "text" not in args:
                raise ToolError("Invalid tool input: Missing required field 'text'")
            return {"result": f"Processed: {args['text']}"}

    return ValidationTool


@pytest.fixture
def slow_tool():
    """Create a tool that takes time to execute."""

    class SlowTool(Tool):
        metadata = ToolMetadata(
            name="slow_tool",
            version="1.0.0",
            description="A deliberately slow tool",
            validation=None,
            author=None,
        )

        async def execute(self, args: dict[str, Any]) -> dict[str, bool]:
            await asyncio.sleep(2.0)  # Make it consistently slow
            return {"done": True}

    return SlowTool


@pytest.fixture
def cacheable_tool():
    """Create a tool that supports caching."""

    class CacheableTool(Tool):
        metadata = ToolMetadata(
            name="cacheable_tool",
            version="1.0.0",
            description="Cacheable tool",
            validation=None,
            author=None,
        )

        _execution_count: ClassVar[int] = 0

        async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
            # Use manager's caching mechanism instead of context state
            if not self.context.cache_enabled:
                CacheableTool._execution_count += 1
                return {
                    "cached_result": args["input"],
                    "execution_count": CacheableTool._execution_count,
                }

            # Let the manager handle caching
            CacheableTool._execution_count += 1
            return {
                "cached_result": args["input"],
                "execution_count": CacheableTool._execution_count,
            }

    return CacheableTool


@pytest.fixture
def streaming_tool():
    """Create a tool that supports streaming."""

    class StreamingTool(Tool):
        metadata = ToolMetadata(
            name="streaming_tool",
            version="1.0.0",
            description="Streaming tool",
            validation=None,
            author=None,
            supports_streaming=True,
        )

        async def execute(self, args: dict[str, Any]) -> Any:
            return args["data"]

        async def stream(self, args: dict[str, Any]) -> AsyncGenerator[Any, None]:
            for item in args["data"]:
                yield item
                await asyncio.sleep(0.1)  # Controlled delay between items

    return StreamingTool


@pytest.fixture
def dry_run_tool():
    """Create a tool that supports dry run."""

    class DryRunTool(Tool):
        metadata = ToolMetadata(
            name="dry_run_tool",
            version="1.0.0",
            description="Tool with dry run support",
            validation=None,
            author=None,
        )

        async def execute(self, args: dict[str, Any]) -> dict[str, Any]:
            if self.context.dry_run:
                return {"simulated": True, "args": args}
            return {"actual": "executed"}

    return DryRunTool


@pytest.mark.asyncio
async def test_tool_validation(tool_manager, validation_tool):
    """Test tool input/output validation."""
    tool_manager.register_tool(validation_tool)

    # Test valid input
    result = await tool_manager.execute_tool("validation_tool", {"text": "test"})
    assert result == {"result": "Processed: test"}

    # Test invalid input
    with pytest.raises(ToolError) as exc:
        await tool_manager.execute_tool("validation_tool", {"invalid": "input"})
    assert "Invalid tool input" in str(exc.value)


@pytest.mark.asyncio
async def test_tool_caching(tool_manager, cacheable_tool):
    """Test tool result caching."""
    tool_cls = cacheable_tool
    tool_manager.register_tool(tool_cls)

    tool_cls._execution_count = 0

    args = {"input": "test_data"}
    result1 = await tool_manager.execute_tool("cacheable_tool", args)
    assert result1["execution_count"] == 1

    result2 = await tool_manager.execute_tool("cacheable_tool", args)
    assert result2["execution_count"] == 1  # Should be same as first execution
    assert result1 == result2

    metrics = tool_manager.get_metrics("cacheable_tool")
    assert metrics.total_calls == 2  # Total calls should count cache hits
    assert metrics.successful_calls == 2


@pytest.mark.asyncio
async def test_tool_streaming(tool_manager, streaming_tool):
    """Test tool streaming functionality."""
    tool_manager.register_tool(streaming_tool)
    data = ["chunk1", "chunk2", "chunk3"]

    chunks = []
    async for chunk in tool_manager.stream_tool("streaming_tool", {"data": data}):
        chunks.append(chunk)

    assert chunks == data


@pytest.mark.asyncio
async def test_dry_run(tool_manager, dry_run_tool):
    """Test tool dry run mode."""
    tool_manager.register_tool(dry_run_tool)

    # Normal execution
    result = await tool_manager.execute_tool("dry_run_tool", {})
    assert result == {"actual": "executed"}

    # Dry run
    context = ToolContext(dry_run=True)
    result = await tool_manager.execute_tool(
        "dry_run_tool", {"test": "arg"}, context=context
    )
    assert result["simulated"] is True


@pytest.mark.asyncio
async def test_tool_metrics(tool_manager, validation_tool):
    """Test tool execution metrics."""
    tool_manager.register_tool(validation_tool)

    # Successful execution
    await tool_manager.execute_tool("validation_tool", {"text": "test"})

    # Failed execution
    with pytest.raises(ToolError):
        await tool_manager.execute_tool("validation_tool", {"invalid": "input"})

    metrics = tool_manager.get_metrics("validation_tool")
    assert metrics.total_calls == 2
    assert metrics.successful_calls == 1
    assert metrics.failed_calls == 1
    assert metrics.last_used is not None
    assert isinstance(metrics.average_execution_time, float)


@pytest.mark.asyncio
async def test_tool_timeout(tool_manager, slow_tool):
    """Test tool execution timeout."""
    tool_manager.register_tool(slow_tool)
    context = ToolContext(timeout=0.5)  # Short timeout

    with pytest.raises(asyncio.TimeoutError):
        await tool_manager.execute_tool("slow_tool", {}, context=context)
