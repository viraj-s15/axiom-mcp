"""Tests for prompt base classes and functionality."""

import base64
from datetime import datetime

import pytest
from mcp.types import TextContent as MCPTextContent
from mcp.types import TextResourceContents
from pydantic import AnyUrl

from axiom_mcp.prompts.base import (
    AssistantMessage,
    EmbeddedResource,
    ImageContent,
    Message,
    Prompt,
    SystemMessage,
    UserMessage,
)


def test_message_with_string_content():
    """Test message creation with string content."""
    msg = Message(content=MCPTextContent(type="text", text="Hello"), role="user")
    assert isinstance(msg.content, MCPTextContent)
    assert msg.content.text == "Hello"
    assert msg.role == "user"
    assert isinstance(msg.timestamp, datetime)


def test_message_with_text_content():
    """Test message creation with TextContent."""
    content = MCPTextContent(type="text", text="Hello")
    msg = Message(content=content, role="user")
    assert msg.content == content
    assert msg.role == "user"


def test_message_with_image_content():
    """Test message creation with ImageContent."""
    # Convert test image data to base64 string as required
    image_data = base64.b64encode(b"image data").decode("utf-8")
    content = ImageContent(type="image", data=image_data, mimeType="image/jpeg")
    msg = Message(content=content, role="assistant")
    assert msg.content == content
    assert msg.role == "assistant"


def test_message_with_embedded_resource():
    """Test message creation with EmbeddedResource."""
    resource_content = TextResourceContents(
        text="Sample text content",
        mimeType="text/plain",
        uri=AnyUrl("https://test.example.com/sample.txt"),
    )
    content = EmbeddedResource(type="resource", resource=resource_content)
    msg = Message(content=content, role="system")
    assert msg.content == content
    assert msg.role == "system"


def test_message_metadata():
    """Test message metadata handling."""
    metadata = {"source": "test", "priority": 1}
    msg = Message(
        content=MCPTextContent(type="text", text="Hello"),
        role="user",
        metadata=metadata,
    )
    assert msg.metadata == metadata


def test_specific_message_types():
    """Test user, assistant, and system message types."""
    user_msg = UserMessage(content="User input", role="user")
    assert user_msg.role == "user"

    assistant_msg = AssistantMessage(content="Assistant response", role="assistant")
    assert assistant_msg.role == "assistant"

    system_msg = SystemMessage(content="System instruction", role="system")
    assert system_msg.role == "system"


@pytest.mark.asyncio
async def test_prompt_creation():
    """Test prompt creation and basic functionality."""

    def example_fn(name: str, age: int = 0) -> str:
        return f"Hello {name}, you are {age} years old"

    prompt = Prompt.from_function(
        fn=example_fn,
        name="greeting",
        description="A greeting prompt",
        version="1.0.0",
        tags=["test", "greeting"],
    )

    assert prompt.name == "greeting"
    assert prompt.description == "A greeting prompt"
    assert prompt.version == "1.0.0"
    assert prompt.tags == ["test", "greeting"]
    assert len(prompt.arguments) == 2

    # Test argument details
    name_arg = next(arg for arg in prompt.arguments if arg.name == "name")
    assert name_arg.required is True
    assert name_arg.type_hint == "str"

    age_arg = next(arg for arg in prompt.arguments if arg.name == "age")
    assert age_arg.required is False
    assert age_arg.type_hint == "int"
    assert age_arg.default == 0


@pytest.mark.asyncio
async def test_prompt_rendering():
    """Test prompt rendering with different return types."""

    def text_fn(text: str) -> str:
        return text

    def message_fn(text: str) -> Message:
        return UserMessage(content=text, role="user")

    def list_fn(text: str) -> list[Message]:
        return [
            SystemMessage(content="Context", role="system"),
            UserMessage(content=text, role="user"),
        ]

    text_prompt = Prompt.from_function(fn=text_fn, name="text_prompt")
    message_prompt = Prompt.from_function(fn=message_fn, name="message_prompt")
    list_prompt = Prompt.from_function(fn=list_fn, name="list_prompt")

    # Test text rendering
    result = await text_prompt.render({"text": "Hello"})
    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert isinstance(result[0].content, MCPTextContent)
    assert result[0].content.text == "Hello"

    # Test message rendering
    result = await message_prompt.render({"text": "Hello"})
    assert len(result) == 1
    assert isinstance(result[0], UserMessage)
    assert isinstance(result[0].content, MCPTextContent)
    assert result[0].content.text == "Hello"

    # Test list rendering
    result = await list_prompt.render({"text": "Hello"})
    assert len(result) == 2
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], UserMessage)
    assert isinstance(result[0].content, MCPTextContent)
    assert isinstance(result[1].content, MCPTextContent)
    assert result[0].content.text == "Context"
    assert result[1].content.text == "Hello"


@pytest.mark.asyncio
async def test_prompt_validation():
    """Test prompt validation and error handling."""

    def example_fn(required_arg: str) -> str:
        return required_arg

    prompt = Prompt.from_function(fn=example_fn, name="test_prompt")

    with pytest.raises(ValueError, match="Missing required arguments"):
        await prompt.render({})

    with pytest.raises(ValueError, match="Missing required arguments"):
        await prompt.render({"wrong_arg": "value"})


def test_lambda_prompt_creation():
    """Test that lambda functions are not allowed without names."""
    with pytest.raises(
        ValueError, match="Lambda functions must be provided with a name"
    ):
        Prompt.from_function(fn=lambda x: x)


@pytest.mark.asyncio
async def test_async_prompt():
    """Test async prompt functions."""

    async def async_fn(text: str) -> str:
        return f"Async: {text}"

    prompt = Prompt.from_function(fn=async_fn, name="async_prompt")
    result = await prompt.render({"text": "Hello"})
    assert len(result) == 1
    assert isinstance(result[0].content, MCPTextContent)
    assert result[0].content.text == "Async: Hello"


def test_prompt_timestamps():
    """Test prompt creation and update timestamps."""

    def example_fn() -> str:
        return "Hello"

    prompt = Prompt.from_function(fn=example_fn, name="timestamp_test")
    assert isinstance(prompt.created_at, datetime)
    assert isinstance(prompt.updated_at, datetime)
    assert prompt.created_at <= prompt.updated_at
