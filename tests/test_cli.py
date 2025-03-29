"""Tests for CLI functionality."""

import pytest
from axiom_mcp.cli import app
from typer.testing import CliRunner


from unittest.mock import patch, call
import asyncio
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)  # Prevent fork-related warnings


# Configure runner to capture both stdout and stderr
runner = CliRunner(mix_stderr=False)


def test_version():
    """Test version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Axiom MCP version" in result.stdout


def test_dev_missing_file():
    """Test dev command with missing file."""
    with patch("rich.console.Console.print") as mock_print:
        result = runner.invoke(app, ["dev", "nonexistent.py"])
        assert result.exit_code == 1
        # Check if the mock was called with the expected string
        mock_print.assert_any_call("[red]File not found: nonexistent.py[/]")


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

    def __init__(self):
        """Initialize tool."""
        super().__init__()
        self.settings = type('Settings', (), {'debug': False, 'log_level': 'info', 'port': 8888})()

    async def run(self, **kwargs):
        """Run the server."""
        return True

mcp = EchoTool()
'''
    )

    # Use a custom process_pid that we can control
    process_pid = None

    # Mock is_alive to initially return True, then False after checking once
    call_count = 0

    def mock_is_alive():
        nonlocal call_count
        call_count += 1
        return call_count < 2  # Return True first time, False after

    # Patch multiprocessing.Process to avoid actual process creation
    with (
        patch("multiprocessing.Process") as mock_process,
        patch("rich.console.Console.print") as mock_print,
    ):

        # Configure process mock with controlled behavior
        mock_process_instance = mock_process.return_value
        mock_process_instance.is_alive = mock_is_alive
        mock_process_instance.pid = process_pid

        # Run the command with no-reload to avoid infinite wait
        result = runner.invoke(app, ["dev", str(server_file), "--no-reload"])

        # Verify process was configured correctly
        mock_process.assert_called_once()

        # Verify console output
        expected_calls = [
            call("[green]Starting development server[/]"),
            call("*" * 50),
            call("[green]Server started successfully![/]"),
        ]
        mock_print.assert_has_calls(expected_calls, any_order=False)

        assert result.exit_code == 0


def test_dev_with_env_file(tmp_path):
    """Test dev command with environment file."""
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR=test_value")

    result = runner.invoke(app, ["dev", "test.py", "--env-file", str(env_file)])
    assert result.exit_code == 1  # Should fail because test.py doesn't exist


@pytest.mark.asyncio
async def test_production_server_with_security_features(tmp_path):
    """Test production server with various security features enabled."""
    # Create temporary server file
    server_file = tmp_path / "test_server.py"
    server_file.write_text(
        '''
from typing import Any
from axiom_mcp.tools.base import Tool, ToolMetadata
from axiom_mcp.tools.utils import tool

@tool(
    name="test_tool",
    description="Test tool",
    version="1.0.0"
)
class TestTool(Tool):
    """Tool for testing server configuration."""
    def __init__(self):
        """Initialize tool."""
        super().__init__()
        self.settings = type('Settings', (), {
            'debug': False,
            'log_level': 'info',
            'port': 8888,
            'enable_rate_limit': False,
            'validate_requests': False,
            'security_headers': None
        })()

    async def execute(self, args: dict[str, Any]) -> str:
        return "Test response"

    def run(self, **kwargs):
        """Run the server - non-async version to avoid coroutine warning."""
        # Return immediately for testing
        return True

mcp = TestTool()
'''
    )

    # Replace the asyncio.run patch with a proper implementation
    def mock_async_run(coro):
        # Create a new event loop for each test to avoid warnings
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # Test with all security features enabled
    with (
        patch("asyncio.run", side_effect=mock_async_run) as mock_run,
        patch("rich.console.Console.print") as mock_print,
    ):

        result = runner.invoke(
            app,
            [
                "run",
                str(server_file),
                "--rate-limit",
                "--validate-requests",
                "--security-headers",
            ],
        )

        # Verify console output
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert "[green]Production server started successfully![/]" in calls
        assert "[yellow]Running with the following security settings:[/]" in calls
        assert "✓ Detailed error messages hidden" in calls
        assert "✓ Rate limiting enabled" in calls
        assert "✓ Request validation enabled" in calls
        assert "✓ Security headers configured" in calls
        assert result.exit_code == 0

        # Verify the mock was called
        assert mock_run.call_count == 1

    # Test with only rate limiting
    with (
        patch("asyncio.run", side_effect=mock_async_run) as mock_run,
        patch("rich.console.Console.print") as mock_print,
    ):

        result = runner.invoke(app, ["run", str(server_file), "--rate-limit"])

        calls = [call[0][0] for call in mock_print.call_args_list]
        assert "✓ Rate limiting enabled" in calls
        assert "✓ Request validation enabled" not in calls
        assert "✓ Security headers configured" not in calls
        assert result.exit_code == 0

        assert mock_run.call_count == 1

    # Test with only request validation
    with (
        patch("asyncio.run", side_effect=mock_async_run) as mock_run,
        patch("rich.console.Console.print") as mock_print,
    ):

        result = runner.invoke(app, ["run", str(server_file), "--validate-requests"])

        calls = [call[0][0] for call in mock_print.call_args_list]
        assert "✓ Request validation enabled" in calls
        assert "✓ Rate limiting enabled" not in calls
        assert "✓ Security headers configured" not in calls
        assert result.exit_code == 0

        assert mock_run.call_count == 1

    # Test with only security headers
    with (
        patch("asyncio.run", side_effect=mock_async_run) as mock_run,
        patch("rich.console.Console.print") as mock_print,
    ):

        result = runner.invoke(app, ["run", str(server_file), "--security-headers"])

        calls = [call[0][0] for call in mock_print.call_args_list]
        assert "✓ Security headers configured" in calls
        assert "✓ Rate limiting enabled" not in calls
        assert "✓ Request validation enabled" not in calls
        assert result.exit_code == 0

        assert mock_run.call_count == 1

    # Test with no security features
    with (
        patch("asyncio.run", side_effect=mock_async_run) as mock_run,
        patch("rich.console.Console.print") as mock_print,
    ):

        result = runner.invoke(app, ["run", str(server_file)])

        calls = [call[0][0] for call in mock_print.call_args_list]
        assert "✓ Detailed error messages hidden" in calls
        assert "✓ Rate limiting enabled" not in calls
        assert "✓ Request validation enabled" not in calls
        assert "✓ Security headers configured" not in calls
        assert result.exit_code == 0

        assert mock_run.call_count == 1
