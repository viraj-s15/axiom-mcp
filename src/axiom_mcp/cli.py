"""Axiom MCP CLI tools."""

import asyncio
import sys
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console


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
        print("*" * 50)
        print("TESTING AXIOM MCP")
        print(f"Current directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        print("*" * 50)

        # Add current directory to Python path
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))

        print(f"Importing server from {file_spec}...")
        # Parse file path to get module path
        file_path = Path(file_spec).resolve()

        # Construct module path based on the new src structure
        rel_path = file_path.relative_to(Path.cwd())
        if str(rel_path).startswith("src/"):
            # Remove 'src/' prefix for import
            module_path = str(rel_path)[4:].replace("/", ".").replace(".py", "")
        else:
            module_path = str(rel_path).replace("/", ".").replace(".py", "")

        # Import the module and get server object
        module = __import__(module_path, fromlist=["*"])
        server = getattr(module, "mcp", None)

        if not server:
            print("No server object 'mcp' found")
            sys.exit(1)

        print(f"Server object: {server}")
        print(f"Server settings: {server.settings}")

        # Configure server
        print("Running server...")
        server.settings.debug = True
        server.settings.log_level = "debug"

        # Get server info
        print("Getting server components...")
        tools = asyncio.run(server.list_tools())
        prompts = asyncio.run(server.list_prompts())
        resources = asyncio.run(server.list_resources())

        print(f"Tools: {tools}")
        print(f"Prompts: {prompts}")
        print(f"Resources: {resources}")

        # Use test1.py port
        server.settings.port = 8888
        print(f"Starting server on port {server.settings.port}...")

        # Run server
        asyncio.run(server.run(transport="sse"))

    except Exception as e:
        import traceback

        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    # Simple direct implementation that works just like test1.py
    try:
        # Check if running the development server
        if len(sys.argv) >= 2 and sys.argv[1] == "dev":
            if len(sys.argv) < 3:
                print("Error: Missing server file path")
                print("Usage: axiom-mcp dev <server_file.py>")
                sys.exit(1)

            file_path = sys.argv[2]
            print("*" * 50)
            print("AXIOM MCP CLI")
            print(f"Current directory: {os.getcwd()}")
            print("*" * 50)

            # Add current directory to Python path
            if os.getcwd() not in sys.path:
                sys.path.insert(0, os.getcwd())

            print(f"Importing server from {file_path}...")

            # Handle the src folder structure
            if file_path.startswith("src/"):
                module_path = file_path[4:].replace("/", ".").replace(".py", "")
            else:
                module_path = file_path.replace("/", ".").replace(".py", "")

            try:
                module = __import__(module_path, fromlist=["*"])
                server = getattr(module, "mcp")

                print(f"Server object: {server}")
                print(f"Server settings: {server.settings}")

                # Configure and run the server
                print("Running server...")
                server.settings.debug = True
                server.settings.log_level = "debug"

                # Get server components
                tools = asyncio.run(server.list_tools())
                prompts = asyncio.run(server.list_prompts())
                resources = asyncio.run(server.list_resources())

                print(f"Tools: {tools}")
                print(f"Prompts: {prompts}")
                print(f"Resources: {resources}")

                # Use port 8888
                server.settings.port = 8888
                print(f"Starting server on port {server.settings.port}...")

                # Run the server
                asyncio.run(server.run(transport="sse"))

            except Exception as e:
                import traceback

                print(f"ERROR: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            # Use typer for other commands
            app()
    except Exception as e:
        import traceback

        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
