"""Axiom MCP - Model Context Protocol implementation."""

__version__ = "0.1.0"

from .server import AxiomMCP
from .cli import main as cli_main

__all__ = ["AxiomMCP", "cli_main"]
