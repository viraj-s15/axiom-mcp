"""Axiom MCP CLI tools."""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Annotated

import dotenv
import typer

from axiom_mcp.exceptions import AxiomMCPError
from axiom_mcp.prompts import PromptManager
from axiom_mcp.tools.manager import ToolManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="axiom-mcp",
    help="Axiom MCP development tools",
    add_completion=False,
    no_args_is_help=True,
)


def _parse_file_path(file_spec: str) -> tuple[Path, str | None]:
    """Parse a file path that may include a server object specification.

    Args:
        file_spec: Path to file, optionally with :object suffix

    Returns:
        Tuple of (file_path, server_object)
    """
    # Handle Windows paths
    has_windows_drive = len(file_spec) > 1 and file_spec[1] == ":"

    # Split on last colon if not part of Windows drive
    if ":" in (file_spec[2:] if has_windows_drive else file_spec):
        file_str, server_object = file_spec.rsplit(":", 1)
    else:
        file_str, server_object = file_spec, None

    # Resolve file path
    file_path = Path(file_str).expanduser().resolve()
    if not file_path.exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)
    if not file_path.is_file():
        typer.echo(f"Error: Not a file: {file_path}", err=True)
        raise typer.Exit(1)

    return file_path, server_object


def _import_server(file: Path, server_object: str | None = None):
    """Import a server module and get the server object.

    Args:
        file: Path to the file
        server_object: Optional object name in format "module:object" or just "object"

    Returns:
        The server object
    """
    # Add parent dir to Python path for imports
    file_dir = str(file.parent)
    if file_dir not in sys.path:
        sys.path.insert(0, file_dir)

    # Import the module
    spec = importlib.util.spec_from_file_location("server_module", file)
    if not spec or not spec.loader:
        typer.echo(f"Error: Could not load module: {file}", err=True)
        raise typer.Exit(1)

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        typer.echo(f"Error loading module: {e}", err=True)
        raise typer.Exit(1) from e

    # If no object specified, try common names
    if not server_object:
        for name in ["mcp", "server", "app"]:
            if hasattr(module, name):
                return getattr(module, name)

        typer.echo(
            f"Error: No server object found in {file}. Please either:\n"
            "1. Use a standard variable name (mcp, server, or app)\n"
            "2. Specify the object name with file:object syntax",
            err=True,
        )
        raise typer.Exit(1)

    # Handle module:object syntax
    if ":" in server_object:
        module_name, object_name = server_object.split(":", 1)
        try:
            server_module = importlib.import_module(module_name)
            server = getattr(server_module, object_name, None)
        except ImportError as e:
            typer.echo(f"Error: Could not import module '{module_name}'", err=True)
            raise typer.Exit(1) from e
    else:
        # Just object name
        server = getattr(module, server_object, None)

    if server is None:
        typer.echo(f"Error: Server object '{server_object}' not found", err=True)
        raise typer.Exit(1)

    return server


@app.command()
def version():
    """Show the Axiom MCP version."""
    from axiom_mcp import __version__

    typer.echo(f"Axiom MCP version {__version__}")


@app.command()
def dev(
    file_spec: str = typer.Argument(
        ..., help="Python file to run, optionally with :object suffix"
    ),
    with_editable: Annotated[
        Path | None,
        typer.Option(
            "--with-editable",
            "-e",
            help="Directory containing pyproject.toml to install in editable mode",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    with_packages: Annotated[
        list[str],
        typer.Option(
            "--with",
            help="Additional packages to install",
        ),
    ] = list(),
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            "-f",
            help="Load environment variables from a .env file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
):
    """Run an Axiom MCP server with the development UI."""
    try:
        # Load environment variables
        if env_file:
            dotenv.load_dotenv(env_file)
            typer.echo(f"Loaded environment from {env_file}")

        # Parse and import server
        file_path, server_object = _parse_file_path(file_spec)
        server = _import_server(file_path, server_object)

        # Initialize managers
        tool_manager = ToolManager()
        prompt_manager = PromptManager()

        # Start development UI server
        typer.echo(f"Starting development server for {file_path}")
        try:
            asyncio.run(
                server.run(
                    tool_manager=tool_manager,
                    prompt_manager=prompt_manager,
                    dev_mode=True,
                )
            )
        except AttributeError as e:
            typer.echo("Error: Server object must implement run() method", err=True)
            raise typer.Exit(1) from e
        except Exception as e:
            typer.echo(f"Error running server: {e}", err=True)
            raise typer.Exit(1) from e

    except typer.Exit:
        raise
    except AxiomMCPError as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1) from e


def main():
    """Main entry point for CLI."""
    app()
