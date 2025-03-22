"""Tests for CLI functionality."""

from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from axiom_mcp.cli import app

# Configure runner to capture both stdout and stderr
runner = CliRunner(mix_stderr=True)


def test_version():
    """Test version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Axiom MCP version" in result.stdout


def test_dev_missing_file():
    """Test dev command with missing file."""
    result = runner.invoke(app, ["dev", "nonexistent.py"])
    assert result.exit_code == 1
    assert "File not found" in result.output


@pytest.mark.asyncio
async def test_dev_with_echo_server(tmp_path):
    """Test dev command with echo server."""
    # Create temporary echo server file
    server_file = tmp_path / "echo_server.py"
    server_file.write_text(
        '''
from typing import Any
from axiom_mcp.tools.base import Tool, ToolMetadata
from axiom_mcp.tools.utils import tool

@tool(
    name="echo_tool",
    description="Echo a message back",
    version="1.0.0"
)
class EchoTool(Tool):
    """Tool to echo messages back."""
    async def execute(self, args: dict[str, Any]) -> str:
        return f"Echo: {args.get('message', '')}"

    async def run(self, **kwargs):
        """Run the server."""
        # Mock implementation that just returns immediately for testing
        return True

mcp = EchoTool()
'''
    )

    # Patch asyncio.run to prevent actual server execution
    with patch("asyncio.run", new_callable=AsyncMock) as mock_run:
        result = runner.invoke(app, ["dev", str(server_file)])
        assert result.exit_code == 0
        assert "Starting development server" in result.output
        mock_run.assert_called_once()


def test_dev_with_env_file(tmp_path):
    """Test dev command with environment file."""
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR=test_value")

    result = runner.invoke(app, ["dev", "test.py", "--env-file", str(env_file)])
    assert result.exit_code == 1  # Should fail because test.py doesn't exist
    # But env file should be processed without error
    assert "Could not load module" not in result.output
