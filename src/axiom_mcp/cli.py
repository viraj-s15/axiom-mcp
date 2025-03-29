"""Axiom MCP CLI tools."""

import asyncio
import sys
import os
import signal
from pathlib import Path
from typing import Annotated, Optional
import importlib.util
import typer
from rich.console import Console
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import multiprocessing

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


class CodeReloader(FileSystemEventHandler):
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.last_reload = time.time()
        self.process: Optional[multiprocessing.Process] = None
        self.should_reload = multiprocessing.Event()

    def on_modified(self, event):
        if event.src_path == str(self.file_path):
            current_time = time.time()
            if current_time - self.last_reload > 1:  # Debounce reloads
                console.print(
                    f"[yellow]Detected changes in {event.src_path}, reloading...[/]"
                )
                self.last_reload = current_time
                self.should_reload.set()

    def run_server_process(self, file_path: Path):
        try:
            # Add file's directory to Python path
            file_dir = str(file_path.parent)
            if file_dir not in sys.path:
                sys.path.insert(0, file_dir)

            # Import module using importlib for better control
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                console.print(
                    f"[red]Could not load module specification from {file_path}[/]"
                )
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            if spec.loader is None:
                console.print(f"[red]Could not get loader for module {module_name}[/]")
                return

            spec.loader.exec_module(module)
            server = getattr(module, "mcp", None)
            if not server:
                console.print("[red]No server object 'mcp' found in module[/]")
                return

            # Configure server
            server.settings.debug = True
            server.settings.log_level = "debug"
            server.settings.port = 8888

            # Run server
            console.print("[green]Server started successfully![/]")
            asyncio.run(server.run(transport="sse"))

        except Exception as e:
            console.print(f"[red]Error running server: {e}[/]")
            return

    def start_server(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            # Force kill if it didn't terminate gracefully
            if self.process.is_alive() and self.process.pid is not None:
                os.kill(self.process.pid, signal.SIGKILL)
                self.process.join()

        # Clear the reload flag
        self.should_reload.clear()

        # Start a new server process
        self.process = multiprocessing.Process(
            target=self.run_server_process, args=(self.file_path,)
        )
        self.process.start()

    def watch_and_reload(self):
        while True:
            try:
                if self.should_reload.is_set():
                    self.start_server()
                time.sleep(1)
            except KeyboardInterrupt:
                if self.process and self.process.is_alive():
                    self.process.terminate()
                    self.process.join()
                break


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
    no_reload: Annotated[
        bool,
        typer.Option(
            "--no-reload",
            help="Disable auto-reload on file changes",
        ),
    ] = False,
):
    """Run an Axiom MCP server with the development UI."""
    try:
        console.print("[green]Starting development server[/]")
        console.print("*" * 50)

        # Check if file exists
        try:
            file_path = Path(file_spec).resolve(strict=True)
        except (FileNotFoundError, RuntimeError):
            console.print(f"[red]File not found: {file_spec}[/]")
            raise typer.Exit(code=1)

        if not no_reload:
            # Set up watchdog observer
            observer = Observer()
            reloader = CodeReloader(file_path)
            observer.schedule(reloader, str(file_path.parent), recursive=False)
            observer.start()

            try:
                # Start initial server
                reloader.start_server()
                # Watch for changes and handle reloads
                reloader.watch_and_reload()
            except KeyboardInterrupt:
                observer.stop()
                observer.join()
                console.print("\n[yellow]Shutting down server...[/]")
        else:
            # Run without hot reloading
            reloader = CodeReloader(file_path)
            reloader.run_server_process(file_path)

        return 0

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/]")
        raise typer.Exit(code=1)


def main():
    """Main entry point for CLI."""
    try:
        if sys.platform != "win32":
            # Set up proper handling of subprocesses on Unix-like systems
            multiprocessing.set_start_method("fork")
        app()
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
