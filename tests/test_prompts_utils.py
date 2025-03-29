"""Tests for prompt utilities."""

import asyncio
from collections.abc import Callable

import pytest
from mcp.types import TextContent

from axiom_mcp.exceptions import PromptRenderError
from axiom_mcp.prompts.base import Message, Prompt
from axiom_mcp.prompts.utils import (
    ExecutableFunction,
    FunctionRegistry,
    prompt,
    registry,
)


@pytest.fixture
def registry_instance() -> FunctionRegistry:
    """Create a clean registry for testing."""
    return FunctionRegistry()


@pytest.fixture
def example_function() -> Callable[[str], TextContent]:
    """Create an example function for testing."""

    def example_function(text: str) -> TextContent:
        return TextContent(type="text", text=f"Test: {text}")

    return example_function


def test_function_info(
    registry_instance: FunctionRegistry, example_function: Callable[[str], TextContent]
) -> None:
    """Test function info creation."""
    func = ExecutableFunction(example_function)
    assert func.name == "example_function"
    assert func.info.parameters


def test_register_function(
    registry_instance: FunctionRegistry, example_function: Callable[[str], TextContent]
) -> None:
    """Test function registration."""
    func = registry_instance.register(example_function)
    assert isinstance(func, ExecutableFunction)
    assert registry_instance.get("example_function") is not None


def test_register_with_name(
    registry_instance: FunctionRegistry, example_function: Callable[[str], TextContent]
) -> None:
    """Test function registration with custom name."""
    func = registry_instance.register(example_function, name="custom")
    assert func.name == "custom"
    assert registry_instance.get("custom") is not None


def test_register_decorator(registry_instance: FunctionRegistry) -> None:
    """Test registration as decorator."""

    @registry_instance.register(name="decorated")
    def decorated_fn(text: str) -> TextContent:
        return TextContent(type="text", text=text)

    assert registry_instance.get("decorated") is not None


def test_unregister_function(
    registry_instance: FunctionRegistry, example_function: Callable[[str], TextContent]
) -> None:
    """Test function unregistration."""
    registry_instance.register(example_function)
    registry_instance.unregister("example_function")
    assert registry_instance.get("example_function") is None


def test_list_functions(
    registry_instance: FunctionRegistry, example_function: Callable[[str], TextContent]
) -> None:
    """Test listing registered functions."""
    registry_instance.register(example_function)
    functions = registry_instance.list_functions()
    assert len(functions) == 1
    assert "example_function" in functions


@pytest.mark.asyncio
async def test_execute_function(registry_instance: FunctionRegistry) -> None:
    """Test function execution."""

    @registry_instance.register
    def test_fn() -> Message:
        return Message(content=TextContent(type="text", text="test"), role="assistant")

    result = await registry_instance.execute("test_fn")
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert isinstance(result[0].content, TextContent)
    assert result[0].content.text == "test"


@pytest.mark.asyncio
async def test_execute_with_args(registry_instance: FunctionRegistry) -> None:
    """Test function execution with arguments."""

    @registry_instance.register
    def test_fn(text: str) -> Message:
        return Message(content=TextContent(type="text", text=text), role="assistant")

    result = await registry_instance.execute("test_fn", {"text": "hello"})
    assert len(result) == 1
    assert isinstance(result[0].content, TextContent)
    assert result[0].content.text == "hello"


@pytest.mark.asyncio
async def test_execute_async_function(registry_instance: FunctionRegistry) -> None:
    """Test async function execution."""

    @registry_instance.register
    async def async_fn() -> Message:
        await asyncio.sleep(0.1)
        return Message(
            content=TextContent(type="text", text="async test"), role="assistant"
        )

    result = await registry_instance.execute("async_fn")
    assert len(result) == 1
    assert isinstance(result[0].content, TextContent)
    assert result[0].content.text == "async test"


@pytest.mark.asyncio
async def test_execute_sync_function(registry_instance: FunctionRegistry) -> None:
    """Test synchronous function execution."""

    @registry_instance.register
    def sync_fn() -> Message:
        return Message(
            content=TextContent(type="text", text="sync test"), role="assistant"
        )

    result = await registry_instance.execute("sync_fn")
    assert len(result) == 1
    assert isinstance(result[0].content, TextContent)
    assert result[0].content.text == "sync test"


@pytest.mark.asyncio
async def test_execute_nonexistent(registry_instance: FunctionRegistry) -> None:
    """Test executing nonexistent function."""
    with pytest.raises(PromptRenderError):
        await registry_instance.execute("nonexistent")


def test_global_registry() -> None:
    """Test global registry instance."""

    @registry.register
    def test_fn(text: str) -> Message:
        return Message(content=TextContent(type="text", text=text), role="assistant")

    assert registry.get("test_fn") is not None
    registry.unregister("test_fn")


def test_prompt_decorator() -> None:
    """Test the @prompt decorator functionality."""

    @prompt(name="test_prompt", description="Test prompt", tags=["test"])
    def example_prompt(text: str) -> Message:
        """Example prompt function."""
        return Message(
            content=TextContent(type="text", text=f"Example: {text}"), role="assistant"
        )

    # Check if prompt was registered
    test_prompt = registry.get_prompt("test_prompt")
    assert test_prompt is not None
    assert isinstance(test_prompt, Prompt)
    assert test_prompt.name == "test_prompt"
    assert test_prompt.description == "Test prompt"
    assert test_prompt.tags == ["test"]
    assert len(test_prompt.arguments) == 1
    assert test_prompt.arguments[0].name == "text"
    assert test_prompt.arguments[0].type_hint == "str"


@pytest.mark.asyncio
async def test_prompt_decorator_async(registry_instance: FunctionRegistry) -> None:
    """Test the @prompt decorator with async functions."""

    @registry_instance.register
    async def async_prompt(text: str) -> Message:
        await asyncio.sleep(0.1)
        return Message(
            content=TextContent(type="text", text=f"Async: {text}"), role="assistant"
        )

    result = await registry_instance.execute("async_prompt", {"text": "test"})
    assert len(result) == 1
    assert isinstance(result[0].content, TextContent)
    assert result[0].content.text == "Async: test"


def test_prompt_decorator_default_values() -> None:
    """Test @prompt decorator with default parameter values."""

    @prompt()  # Test without explicit parameters
    def default_prompt(text: str = "default") -> Message:
        """Test prompt with default value."""
        return Message(
            content=TextContent(type="text", text=f"Default: {text}"), role="assistant"
        )

    test_prompt = registry.get_prompt("default_prompt")
    assert test_prompt is not None
    assert test_prompt.name == "default_prompt"
    assert test_prompt.description == "Test prompt with default value."
    assert len(test_prompt.arguments) == 1
    assert test_prompt.arguments[0].name == "text"
    assert test_prompt.arguments[0].default == "default"
    assert not test_prompt.arguments[0].required


def test_prompt_decorator_multiple_arguments() -> None:
    """Test @prompt decorator with multiple arguments."""

    @prompt(tags=["test", "multiple"])
    def multi_arg_prompt(text: str, count: int, flag: bool = False) -> Message:
        return Message(
            content=TextContent(type="text", text=f"{text} - {count} - {flag}"),
            role="assistant",
        )

    test_prompt = registry.get_prompt("multi_arg_prompt")
    assert test_prompt is not None
    assert len(test_prompt.arguments) == 3

    # Check argument details
    text_arg = next(arg for arg in test_prompt.arguments if arg.name == "text")
    assert text_arg.type_hint == "str"
    assert text_arg.required is True

    count_arg = next(arg for arg in test_prompt.arguments if arg.name == "count")
    assert count_arg.type_hint == "int"
    assert count_arg.required is True

    flag_arg = next(arg for arg in test_prompt.arguments if arg.name == "flag")
    assert flag_arg.type_hint == "bool"
    assert flag_arg.required is False
    assert flag_arg.default is False


@pytest.mark.asyncio
async def test_prompt_decorator_return_types(
    registry_instance: FunctionRegistry,
) -> None:
    """Test @prompt decorator with different return types."""

    @registry_instance.register
    def str_prompt() -> Message:
        return Message(
            content=TextContent(type="text", text="String response"), role="assistant"
        )

    @registry_instance.register
    def message_prompt() -> Message:
        return Message(
            content=TextContent(type="text", text="Message response"), role="assistant"
        )

    @registry_instance.register
    def list_prompt() -> list[Message]:
        return [
            Message(
                content=TextContent(type="text", text="First message"), role="assistant"
            ),
            Message(
                content=TextContent(type="text", text="Second message"),
                role="assistant",
            ),
        ]

    # Test function return
    result = await registry_instance.execute("str_prompt")
    assert len(result) == 1
    assert isinstance(result[0].content, TextContent)
    assert result[0].content.text == "String response"

    # Test Message return
    result = await registry_instance.execute("message_prompt")
    assert len(result) == 1
    assert isinstance(result[0].content, TextContent)
    assert result[0].content.text == "Message response"

    # Test list return
    result = await registry_instance.execute("list_prompt")
    assert len(result) == 2
    assert isinstance(result[0].content, TextContent)
    assert isinstance(result[1].content, TextContent)
    assert result[0].content.text == "First message"
    assert result[1].content.text == "Second message"
