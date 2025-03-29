"""Axiom MCP - Model Context Protocol implementation."""

__version__ = "0.1.0"

from .server import AxiomMCP
from .cli import app as cli_app

__all__ = ["AxiomMCP", "cli_app"]
