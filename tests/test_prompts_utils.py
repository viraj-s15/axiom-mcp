"""Tests for prompt utilities."""

import asyncio
from collections.abc import Callable

import pytest

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
def example_function() -> Callable[[str], str]:
    """Create an example function for testing."""

    def func(text: str) -> str:
        return f"Test: {text}"

    return func


def test_function_info(
    registry_instance: FunctionRegistry, example_function: Callable[[str], str]
) -> None:
    """Test function info creation."""
    func = ExecutableFunction(example_function)
    assert func.name == "func"
    assert func.info.parameters


def test_register_function(
    registry_instance: FunctionRegistry, example_function: Callable[[str], str]
) -> None:
    """Test function registration."""
    func = registry_instance.register(example_function)
    assert isinstance(func, ExecutableFunction)
    assert registry_instance.get("func") is not None


def test_register_with_name(
    registry_instance: FunctionRegistry, example_function: Callable[[str], str]
) -> None:
    """Test function registration with custom name."""
    func = registry_instance.register(example_function, name="custom")
    assert func.name == "custom"
    assert registry_instance.get("custom") is not None


def test_register_decorator(registry_instance: FunctionRegistry) -> None:
    """Test registration as decorator."""

    @registry_instance.register(name="decorated")
    def decorated_fn(text: str) -> str:
        return text

    assert registry_instance.get("decorated") is not None


def test_unregister_function(
    registry_instance: FunctionRegistry, example_function: Callable[[str], str]
) -> None:
    """Test function unregistration."""
    registry_instance.register(example_function)
    registry_instance.unregister("func")
    assert registry_instance.get("func") is None


def test_list_functions(
    registry_instance: FunctionRegistry, example_function: Callable[[str], str]
) -> None:
    """Test listing registered functions."""
    registry_instance.register(example_function)
    functions = registry_instance.list_functions()
    assert len(functions) == 1
    assert "func" in functions


@pytest.mark.asyncio
async def test_execute_function(registry_instance: FunctionRegistry) -> None:
    """Test function execution."""

    @registry_instance.register
    def test_fn() -> Message:
        return Message(content="test", role="assistant")

    result = await registry_instance.execute("test_fn")
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert str(result[0]) == "test"


@pytest.mark.asyncio
async def test_execute_with_args(registry_instance: FunctionRegistry) -> None:
    """Test function execution with arguments."""

    @registry_instance.register
    def test_fn(text: str) -> Message:
        return Message(content=text, role="assistant")

    result = await registry_instance.execute("test_fn", {"text": "hello"})
    assert len(result) == 1
    assert str(result[0]) == "hello"


@pytest.mark.asyncio
async def test_execute_async_function(registry_instance: FunctionRegistry) -> None:
    """Test async function execution."""

    @registry_instance.register
    async def async_fn() -> str:
        await asyncio.sleep(0.1)
        return "async test"

    result = await registry_instance.execute("async_fn")
    assert len(result) == 1
    assert str(result[0]) == "async test"


@pytest.mark.asyncio
async def test_execute_sync_function(registry_instance: FunctionRegistry) -> None:
    """Test synchronous function execution."""

    @registry_instance.register
    def sync_fn() -> str:
        return "sync test"

    result = await registry_instance.execute("sync_fn")
    assert len(result) == 1
    assert str(result[0]) == "sync test"


@pytest.mark.asyncio
async def test_execute_nonexistent(registry_instance: FunctionRegistry) -> None:
    """Test executing nonexistent function."""
    with pytest.raises(PromptRenderError):
        await registry_instance.execute("nonexistent")


def test_global_registry() -> None:
    """Test global registry instance."""

    @registry.register
    def test_fn(text: str) -> str:
        return text

    assert registry.get("test_fn") is not None
    registry.unregister("test_fn")


def test_prompt_decorator() -> None:
    """Test the @prompt decorator functionality."""

    @prompt(name="test_prompt", description="Test prompt", tags=["test"])
    def example_prompt(text: str) -> str:
        """Example prompt function."""
        return f"Example: {text}"

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
    async def async_prompt(text: str) -> str:
        await asyncio.sleep(0.1)
        return f"Async: {text}"

    # Get the registered function
    func = registry_instance.get("async_prompt")
    assert func is not None

    # Test execution
    result = await registry_instance.execute("async_prompt", {"text": "test"})
    assert len(result) == 1
    assert str(result[0]) == "Async: test"


def test_prompt_decorator_default_values() -> None:
    """Test @prompt decorator with default parameter values."""

    @prompt()  # Test without explicit parameters
    def default_prompt(text: str = "default") -> str:
        """Test prompt with default value."""
        return f"Default: {text}"

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
    def multi_arg_prompt(text: str, count: int, flag: bool = False) -> str:
        return f"{text} - {count} - {flag}"

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

    # Define and register prompts directly with test registry
    @prompt(registry=registry_instance)
    def str_prompt() -> str:
        return "String response"

    @prompt(registry=registry_instance)
    def message_prompt() -> Message:
        return Message(content="Message response", role="assistant")

    @prompt(registry=registry_instance)
    def list_prompt() -> list[Message]:
        return [
            Message(content="First message", role="assistant"),
            Message(content="Second message", role="assistant"),
        ]

    # Test string return
    str_result = await registry_instance.execute("str_prompt")
    assert len(str_result) == 1
    assert str(str_result[0]) == "String response"

    # Test Message return
    msg_result = await registry_instance.execute("message_prompt")
    assert len(msg_result) == 1
    assert str(msg_result[0]) == "Message response"

    # Test list return
    list_result = await registry_instance.execute("list_prompt")
    assert len(list_result) == 2
    assert str(list_result[0]) == "First message"
    assert str(list_result[1]) == "Second message"
