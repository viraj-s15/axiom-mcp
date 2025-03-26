"""Axiom MCP Prompts - A powerful prompt management system."""

from mcp.types import EmbeddedResource, ImageContent, TextContent

from .base import (
    AssistantMessage,
    Message,
    Prompt,
    PromptResponse,
    SystemMessage,
    UserMessage,
)
from .manager import PromptManager, PromptMetrics
from .utils import batch_register, combine_prompts, prompt

__all__ = [
    # Base types from mcp.types
    "TextContent",
    "ImageContent",
    "EmbeddedResource",
    # Message types
    "Message",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "Prompt",
    "PromptResponse",
    # Manager
    "PromptManager",
    "PromptMetrics",
    # Utilities
    "prompt",
    "batch_register",
    "combine_prompts",
]
