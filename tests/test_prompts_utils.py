"""Tests for prompt utilities."""

import asyncio
from collections.abc import Callable

import pytest
from pytest import FixtureRequest

from axiom_mcp.exceptions import PromptRenderError
from axiom_mcp.prompts.base import Message
from axiom_mcp.prompts.utils import ExecutableFunction, FunctionRegistry, registry


@pytest.fixture
def registry_instance(_: FixtureRequest) -> FunctionRegistry:
    """Create a clean registry for testing."""
    return FunctionRegistry()


@pytest.fixture
def example_function(_: FixtureRequest) -> Callable[[str], str]:
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
async def test_execute_function(
    _: FixtureRequest, registry_instance: FunctionRegistry
) -> None:
    """Test function execution."""

    @registry_instance.register
    def test_fn() -> Message:
        return Message(content="test", role="assistant")

    result = await registry_instance.execute("test_fn")
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert str(result[0]) == "test"


@pytest.mark.asyncio
async def test_execute_with_args(
    _: FixtureRequest, registry_instance: FunctionRegistry
) -> None:
    """Test function execution with arguments."""

    @registry_instance.register
    def test_fn(text: str) -> Message:
        return Message(content=text, role="assistant")

    result = await registry_instance.execute("test_fn", {"text": "hello"})
    assert len(result) == 1
    assert str(result[0]) == "hello"


@pytest.mark.asyncio
async def test_execute_async_function(
    _: FixtureRequest, registry_instance: FunctionRegistry
) -> None:
    """Test async function execution."""

    @registry_instance.register
    async def async_fn() -> str:
        await asyncio.sleep(0.1)
        return "async test"

    result = await registry_instance.execute("async_fn")
    assert len(result) == 1
    assert str(result[0]) == "async test"


@pytest.mark.asyncio
async def test_execute_sync_function(
    _: FixtureRequest, registry_instance: FunctionRegistry
) -> None:
    """Test synchronous function execution."""

    @registry_instance.register
    def sync_fn() -> str:
        return "sync test"

    result = await registry_instance.execute("sync_fn")
    assert len(result) == 1
    assert str(result[0]) == "sync test"


@pytest.mark.asyncio
async def test_execute_nonexistent(
    _: FixtureRequest, registry_instance: FunctionRegistry
) -> None:
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
