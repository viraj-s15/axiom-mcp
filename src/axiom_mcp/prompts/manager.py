"""Enhanced prompt management functionality."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field

from ..exceptions import UnknownPromptError
from .base import Message, Prompt

logger = logging.getLogger(__name__)


class PromptMetrics(PydanticBaseModel):
    """Metrics for prompt usage and performance."""

    total_calls: int = Field(default=0)
    successful_calls: int = Field(default=0)
    failed_calls: int = Field(default=0)
    average_render_time: float = Field(default=0.0)
    last_used: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PromptManager:
    """Enhanced manager for Axiom MCP prompts with advanced features."""

    def __init__(
        self,
        warn_on_duplicate: bool = True,
        max_concurrent_renders: int = 10,
        enable_metrics: bool = True,
    ):
        self._prompts: dict[str, Prompt] = {}
        self._metrics: dict[str, PromptMetrics] = {}
        self._tags_index: dict[str, set[str]] = {}
        self._render_semaphore = asyncio.Semaphore(max_concurrent_renders)
        self._executor = ThreadPoolExecutor()
        self.warn_on_duplicate = warn_on_duplicate
        self.enable_metrics = enable_metrics

    def add_prompt(self, prompt: Prompt, force: bool = False) -> Prompt:
        """Add a prompt to the manager with validation and indexing."""
        existing = self._prompts.get(prompt.name)

        if existing and not force:
            if self.warn_on_duplicate:
                logger.warning(f"Prompt already exists: {prompt.name}")
            return existing

        # Update prompt registry
        self._prompts[prompt.name] = prompt

        # Initialize metrics
        if self.enable_metrics:
            self._metrics[prompt.name] = PromptMetrics()

        # Update tags index
        for tag in prompt.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = set()
            self._tags_index[tag].add(prompt.name)

        logger.info(f"Added prompt: {prompt.name} (version: {prompt.version})")
        return prompt

    def get_prompt(self, name: str) -> Prompt | None:
        """Get a prompt by name."""
        return self._prompts.get(name)

    def remove_prompt(self, name: str) -> bool:
        """Remove a prompt and its associated data."""
        if name not in self._prompts:
            return False

        prompt = self._prompts[name]

        # Clean up tags index
        for tag in prompt.tags:
            if tag in self._tags_index:
                self._tags_index[tag].discard(name)
                if not self._tags_index[tag]:
                    del self._tags_index[tag]

        # Clean up metrics
        self._metrics.pop(name, None)

        # Remove prompt
        del self._prompts[name]

        logger.info(f"Removed prompt: {name}")
        return True

    def list_prompts(self) -> list[Prompt]:
        """List all registered prompts."""
        return list(self._prompts.values())

    def get_prompts_by_tag(self, tag: str) -> list[Prompt]:
        """Get all prompts with a specific tag."""
        if tag not in self._tags_index:
            return []
        return [self._prompts[name] for name in self._tags_index[tag]]

    def get_prompt_metrics(self, name: str) -> PromptMetrics | None:
        """Get metrics for a specific prompt."""
        return self._metrics.get(name)

    async def render_prompt(
        self, prompt_name: str, arguments: dict[str, Any] | None = None
    ) -> list[Message]:
        """Render a prompt with the given arguments."""
        prompt = self.get_prompt(prompt_name)
        if not prompt:
            raise UnknownPromptError(prompt_name)

        start_time = datetime.now(UTC)
        try:
            messages = await prompt.render(arguments)
        except Exception:
            self._record_failure(prompt_name, start_time)
            raise
        else:
            self._record_success(prompt_name, start_time)
            return messages

    def _record_success(self, prompt_name: str, start_time: datetime) -> None:
        """Record a successful prompt execution."""
        self._update_metrics(prompt_name, start_time, success=True)

    def _record_failure(self, prompt_name: str, start_time: datetime) -> None:
        """Record a failed prompt execution."""
        self._update_metrics(prompt_name, start_time, success=False)

    def _update_metrics(
        self, prompt_name: str, start_time: datetime, success: bool
    ) -> None:
        """Update metrics for a prompt execution."""
        if not self.enable_metrics:
            return

        metrics = self._metrics[prompt_name]
        now = datetime.now(UTC)
        render_time = (now - start_time).total_seconds()

        metrics.total_calls += 1
        if success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1

        # Update average render time with moving average
        if metrics.total_calls == 1:
            metrics.average_render_time = render_time
        else:
            metrics.average_render_time = (
                metrics.average_render_time * (metrics.total_calls - 1) + render_time
            ) / metrics.total_calls

        metrics.last_used = now