"""Axiom MCP CLI tools."""

import asyncio
import sys
import os
from pathlib import Path
from typing import Annotated
import importlib.util

import typer
from rich.console import Console

from . import __version__

# Configure console with immediate flush
console = Console(file=sys.stdout, force_terminal=True)

# Create the typer app with explicit name
app = typer.Typer(
    name="axiom-mcp",
    help="Axiom MCP development tools",
    add_completion=False,
    no_args_is_help=True,
)

@app.command()
def version():
    """Show version information."""
    # Use print instead of console.print for test capturing
    print(f"Axiom MCP version {__version__}")
    return 0

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
        print("Starting development server")
        print("*" * 50)
        print(f"Current directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        print("*" * 50)

        # Check if file exists
        try:
            file_path = Path(file_spec).resolve(strict=True)
        except (FileNotFoundError, RuntimeError):
            print(f"File not found: {file_spec}")
            raise typer.Exit(code=1)

        # Add file's directory to Python path
        file_dir = str(file_path.parent)
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)

        try:
            # Import module using importlib for better control
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                print(f"Could not load module specification from {file_path}")
                raise typer.Exit(code=1)
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            if spec.loader is None:
                print(f"Could not get loader for module {module_name}")
                raise typer.Exit(code=1)
                
            spec.loader.exec_module(module)
            
            server = getattr(module, "mcp", None)
            if not server:
                print("No server object 'mcp' found in module")
                raise typer.Exit(code=1)

            # Configure server
            server.settings.debug = True
            server.settings.log_level = "debug"
            server.settings.port = 8888

            # Run server
            print("Starting server...")
            asyncio.run(server.run(transport="sse"))
            return 0

        except ImportError as e:
            print(f"Could not import module: {e}")
            raise typer.Exit(code=1)
        except Exception as e:
            print(f"Error running server: {e}")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        raise typer.Exit(code=1)

def main():
    """Main entry point for CLI."""
    try:
        app()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
