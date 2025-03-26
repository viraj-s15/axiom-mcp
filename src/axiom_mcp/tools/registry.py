"""Tool registry and plugin management."""

import importlib
import importlib.machinery
import importlib.util
import inspect
import logging
import pkgutil
from collections import defaultdict
from pathlib import Path

from ..exceptions import ToolError
from .base import Tool, ToolDependency, ToolMetadata

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for tool management and plugin loading."""

    def __init__(self) -> None:
        self._tools: dict[str, type[Tool]] = {}
        self._dependencies: dict[str, set[str]] = defaultdict(set)
        self._reverse_deps: dict[str, set[str]] = defaultdict(set)
        self._plugin_paths: list[Path] = []

    def register_tool(self, tool_cls: type[Tool]) -> None:
        """Register a tool and its dependencies."""
        name = tool_cls.metadata.name
        if name in self._tools:
            logger.warning(f"Tool already registered: {name}")
            return

        self._tools[name] = tool_cls

        # Register dependencies
        for dep in tool_cls.metadata.dependencies:
            if isinstance(dep, dict):
                dep = ToolDependency(**dep)
            self._dependencies[name].add(dep.tool_name)
            self._reverse_deps[dep.tool_name].add(name)

        logger.info(f"Registered tool: {name} v{tool_cls.metadata.version}")

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool and update dependency tracking."""
        if name not in self._tools:
            return

        # Remove from dependency tracking
        deps = self._dependencies.pop(name, set())
        for dep in deps:
            self._reverse_deps[dep].discard(name)

        # Remove reverse dependencies
        rev_deps = self._reverse_deps.pop(name, set())
        for rev_dep in rev_deps:
            self._dependencies[rev_dep].discard(name)

        # Remove the tool
        del self._tools[name]
        logger.info(f"Unregistered tool: {name}")

    def get_tool(self, name: str) -> type[Tool] | None:
        """Get a tool class by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolMetadata]:
        """List all registered tools."""
        return [tool_cls.metadata for tool_cls in self._tools.values()]

    def get_dependencies(self, name: str) -> set[str]:
        """Get all dependencies for a tool."""
        return self._dependencies.get(name, set())

    def get_dependents(self, name: str) -> set[str]:
        """Get all tools that depend on this tool."""
        return self._reverse_deps.get(name, set())

    def check_dependencies(self, name: str) -> list[str]:
        """Check if all dependencies for a tool are available."""
        if name not in self._tools:
            raise ToolError(f"Tool not found: {name}")

        missing = []
        for dep in self._dependencies[name]:
            if dep not in self._tools:
                missing.append(dep)
        return missing

    def add_plugin_path(self, path: Path | str) -> None:
        """Add a path to search for tool plugins."""
        path = Path(path)
        if path.is_dir() and path not in self._plugin_paths:
            self._plugin_paths.append(path)

    def discover_plugins(self) -> None:
        """Discover and load tool plugins from registered paths."""
        for path in self._plugin_paths:
            if not path.is_dir():
                continue

            for finder, name, _ in pkgutil.iter_modules([str(path)]):
                try:
                    spec = importlib.util.find_spec(name)
                    if spec is None or spec.loader is None:
                        continue

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find and register tool classes
                    for _, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, Tool)
                            and obj is not Tool
                            and hasattr(obj, "metadata")
                        ):
                            self.register_tool(obj)

                except Exception as e:
                    logger.error(f"Failed to load plugin {name}: {e}")

    def validate_dependencies(self) -> dict[str, list[str]]:
        """Validate dependencies for all registered tools."""
        invalid_deps = {}
        for name in self._tools:
            if missing := self.check_dependencies(name):
                invalid_deps[name] = missing
        return invalid_deps


# Global registry instance
registry = ToolRegistry()


def register_tool(tool_cls: type[Tool]) -> type[Tool]:
    """Decorator to register a tool class."""
    registry.register_tool(tool_cls)
    return tool_cls
