"""Tool management and execution for AxiomMCP."""

from .base import Tool, ToolContext, ToolDependency, ToolMetadata, ToolValidationSchema
from .manager import ToolManager, ToolCacheEntry, ToolMetrics
from .registry import ToolRegistry, register_tool
from .utils import (
    tool,
    traced_tool,
    batch_tool,
    ensure_async,
    validate_schema,
    with_validation,
    make_streamable,
    cacheable,
    with_dry_run,
    FileSystemTool,
    ReadFileTool,
    WriteFileTool,
)

__all__ = [
    "Tool",
    "ToolContext",
    "ToolDependency",
    "ToolMetadata",
    "ToolValidationSchema",
    "ToolManager",
    "ToolCacheEntry",
    "ToolMetrics",
    "ToolRegistry",
    "register_tool",
    "tool",
    "traced_tool",
    "batch_tool",
    "ensure_async",
    "validate_schema",
    "with_validation",
    "make_streamable",
    "cacheable",
    "with_dry_run",
    "FileSystemTool",
    "ReadFileTool",
    "WriteFileTool",
]
