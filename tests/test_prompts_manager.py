"""Tests for prompt manager functionality."""

import asyncio

import pytest

from axiom_mcp.prompts.base import Prompt
from axiom_mcp.prompts.manager import PromptManager, PromptMetrics


@pytest.fixture
def manager():
    """Create a test prompt manager."""
    return PromptManager(warn_on_duplicate=True, enable_metrics=True)


@pytest.fixture
def sample_prompt():
    """Create a sample prompt for testing."""

    def example_fn(text: str) -> str:
        return text

    return Prompt.from_function(
        fn=example_fn,
        name="test_prompt",
        description="Test prompt",
        tags=["test"],
    )


def test_add_prompt(manager, sample_prompt):
    """Test adding a prompt to the manager."""
    added_prompt = manager.add_prompt(sample_prompt)
    assert added_prompt == sample_prompt
    assert manager.get_prompt("test_prompt") == sample_prompt


def test_duplicate_prompt_warning(manager, sample_prompt):
    """Test warning behavior for duplicate prompts."""
    manager.add_prompt(sample_prompt)
    duplicate = manager.add_prompt(sample_prompt)
    assert duplicate == sample_prompt


def test_force_add_duplicate_prompt(manager, sample_prompt):
    """Test forcing addition of duplicate prompt."""
    manager.add_prompt(sample_prompt)

    modified_prompt = Prompt.from_function(
        fn=lambda x: x,
        name=sample_prompt.name,
        description="Modified prompt",
    )

    new_prompt = manager.add_prompt(modified_prompt, force=True)
    assert new_prompt == modified_prompt
    assert manager.get_prompt(sample_prompt.name) == modified_prompt


def test_remove_prompt(manager, sample_prompt):
    """Test removing a prompt."""
    manager.add_prompt(sample_prompt)
    assert manager.remove_prompt(sample_prompt.name) is True
    assert manager.get_prompt(sample_prompt.name) is None


def test_remove_nonexistent_prompt(manager):
    """Test removing a prompt that doesn't exist."""
    assert manager.remove_prompt("nonexistent") is False


def test_list_prompts(manager, sample_prompt):
    """Test listing all prompts."""
    manager.add_prompt(sample_prompt)
    prompts = manager.list_prompts()
    assert len(prompts) == 1
    assert prompts[0] == sample_prompt


def test_get_prompts_by_tag(manager, sample_prompt):
    """Test retrieving prompts by tag."""
    manager.add_prompt(sample_prompt)
    tagged_prompts = manager.get_prompts_by_tag("test")
    assert len(tagged_prompts) == 1
    assert tagged_prompts[0] == sample_prompt

    # Test nonexistent tag
    assert len(manager.get_prompts_by_tag("nonexistent")) == 0


def test_prompt_metrics(manager, sample_prompt):
    """Test prompt metrics collection."""
    manager.add_prompt(sample_prompt)
    metrics = manager.get_prompt_metrics(sample_prompt.name)
    assert isinstance(metrics, PromptMetrics)
    assert metrics.total_calls == 0
    assert metrics.successful_calls == 0
    assert metrics.failed_calls == 0


@pytest.mark.asyncio
async def test_render_prompt_metrics(manager, sample_prompt):
    """Test metrics update after rendering."""
    manager.add_prompt(sample_prompt)
    await manager.render_prompt(sample_prompt.name, {"text": "test"})

    metrics = manager.get_prompt_metrics(sample_prompt.name)
    assert metrics.total_calls == 1
    assert metrics.successful_calls == 1
    assert metrics.failed_calls == 0
    assert metrics.last_used is not None


@pytest.mark.asyncio
async def test_render_prompt_failure_metrics(manager):
    """Test metrics update after failed rendering."""

    def failing_fn(text: str) -> str:
        raise ValueError("Test error")

    prompt = Prompt.from_function(fn=failing_fn, name="failing_prompt")
    manager.add_prompt(prompt)

    with pytest.raises(Exception):
        await manager.render_prompt("failing_prompt", {"text": "test"})

    metrics = manager.get_prompt_metrics("failing_prompt")
    assert metrics.total_calls == 1
    assert metrics.successful_calls == 0
    assert metrics.failed_calls == 1


@pytest.mark.asyncio
async def test_concurrent_rendering(manager, sample_prompt):
    """Test concurrent prompt rendering."""

    async def slow_fn(text: str) -> str:
        await asyncio.sleep(0.1)
        return text

    prompt = Prompt.from_function(fn=slow_fn, name="slow_prompt")
    manager.add_prompt(prompt)

    # Test that multiple renders can happen concurrently
    tasks = [
        manager.render_prompt("slow_prompt", {"text": f"test{i}"}) for i in range(5)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    assert all(isinstance(r, list) for r in results)
    assert all(len(r) == 1 for r in results)


@pytest.mark.asyncio
async def test_render_timeout(manager):
    """Test prompt rendering with timeout."""

    async def slow_fn(text: str) -> str:
        await asyncio.sleep(0.2)
        return text

    prompt = Prompt.from_function(fn=slow_fn, name="timeout_prompt")
    manager.add_prompt(prompt)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            manager.render_prompt("timeout_prompt", {"text": "test"}), timeout=0.1
        )
