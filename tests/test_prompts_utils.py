"""Tests for prompt utilities and decorators."""

import pytest
from mcp.types import TextContent as MCPTextContent

from axiom_mcp.prompts.base import Message, Prompt, SystemMessage, UserMessage
from axiom_mcp.prompts.manager import PromptManager
from axiom_mcp.prompts.utils import batch_register, combine_prompts, prompt


@pytest.fixture
def manager():
    """Create a test prompt manager."""
    return PromptManager(warn_on_duplicate=True)


def test_prompt_decorator_basic():
    """Test basic prompt decorator functionality."""

    @prompt(name="test", description="Test prompt", version="1.0.0", tags=["test"])
    def test_fn(name: str) -> str:
        return f"Hello {name}"

    assert hasattr(test_fn, "_prompt")
    prompt_obj = test_fn._prompt  # type: ignore
    assert isinstance(prompt_obj, Prompt)
    assert prompt_obj.name == "test"
    assert prompt_obj.description == "Test prompt"
    assert prompt_obj.version == "1.0.0"
    assert prompt_obj.tags == ["test"]


def test_prompt_decorator_auto_name():
    """Test prompt decorator with automatic name from function."""

    @prompt()
    def auto_named_fn(text: str) -> str:
        return text

    prompt_obj = auto_named_fn._prompt  # type: ignore
    assert prompt_obj.name == "auto_named_fn"


def test_prompt_decorator_with_manager(manager):
    """Test prompt decorator with manager integration."""

    @prompt(name="managed", manager=manager)
    def managed_fn(text: str) -> str:
        return text

    assert manager.get_prompt("managed") is not None
    assert managed_fn("test") == "test"


def test_batch_register(manager):
    """Test batch registration of prompts."""

    @prompt(name="first")
    def first_fn(text: str) -> str:
        return f"First: {text}"

    @prompt(name="second")
    def second_fn(text: str) -> str:
        return f"Second: {text}"

    registered = batch_register(manager, first_fn, second_fn)
    assert len(registered) == 2
    assert all(isinstance(p, Prompt) for p in registered)
    assert manager.get_prompt("first") is not None
    assert manager.get_prompt("second") is not None


def test_batch_register_mixed_types(manager):
    """Test batch registration with mixed prompt types."""

    @prompt(name="decorated")
    def decorated_fn(text: str) -> str:
        return text

    direct_prompt = Prompt.from_function(
        lambda x: x, name="direct", description="Direct prompt"
    )

    registered = batch_register(manager, decorated_fn, direct_prompt)
    assert len(registered) == 2
    assert manager.get_prompt("decorated") is not None
    assert manager.get_prompt("direct") is not None


def test_batch_register_invalid_type(manager):
    """Test batch registration with invalid type."""
    with pytest.raises(ValueError, match="Invalid prompt type"):
        batch_register(manager, lambda x: x)


@pytest.mark.asyncio
async def test_combine_prompts():
    """Test combining multiple prompts."""

    @prompt()
    def context() -> Message:
        return SystemMessage(content="System context", role="system")

    @prompt()
    def greeting(name: str) -> Message:
        return UserMessage(content=f"Hello {name}", role="user")

    combined = combine_prompts(
        context, greeting, name="full_greeting", tags=["greeting"]
    )

    assert combined.name == "full_greeting"
    assert "greeting" in combined.tags

    # First render without args for context
    result = await combined.render({})
    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[0].content, MCPTextContent)
    assert result[0].content.text == "System context"

    # Then render with name for greeting
    result = await combined.render({"name": "Test"})
    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert isinstance(result[0].content, MCPTextContent)
    assert result[0].content.text == "Hello Test"


@pytest.mark.asyncio
async def test_combine_prompts_partial_args():
    """Test combining prompts with partial argument matching."""

    @prompt()
    def first(a: str) -> str:
        return f"First: {a}"

    @prompt()
    def second(b: str) -> str:
        return f"Second: {b}"

    combined = combine_prompts(first, second, name="first_second")

    # Test first prompt
    result = await combined.render({"a": "A"})
    assert len(result) == 1
    assert isinstance(result[0].content, MCPTextContent)
    assert result[0].content.text == "First: A"

    # Test second prompt
    result = await combined.render({"b": "B"})
    assert len(result) == 1
    assert isinstance(result[0].content, MCPTextContent)
    assert result[0].content.text == "Second: B"


@pytest.mark.asyncio
async def test_combine_prompts_async():
    """Test combining async and sync prompts."""

    @prompt()
    async def async_fn() -> str:
        return "Async"

    @prompt()
    def sync_fn() -> str:
        return "Sync"

    combined = combine_prompts(async_fn, sync_fn)
    result = await combined.render()
    assert len(result) == 2
    assert isinstance(result[0].content, MCPTextContent)
    assert isinstance(result[1].content, MCPTextContent)
    assert result[0].content.text == "Async"
    assert result[1].content.text == "Sync"


def test_prompt_decorator_docstring():
    """Test prompt decorator preserves function docstring."""

    @prompt()
    def documented_fn(text: str) -> str:
        """This is a test docstring."""
        return text

    prompt_obj = documented_fn._prompt  # type: ignore
    assert prompt_obj.description == "This is a test docstring."
    assert documented_fn.__doc__ == "This is a test docstring."
