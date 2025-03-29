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

            # Run server (no success message here, it will be shown in parent process)
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

        # Show success message in parent process after process starts
        console.print("[green]Server started successfully![/]")

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


def run_production_server(
    file_path: Path,
    rate_limit: bool = False,
    request_validation: bool = False,
    security_headers: bool = False,
):
    try:
        # Add file's directory to Python path
        file_dir = str(file_path.parent)
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)

        # Import module using importlib
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

        # Configure server for production
        server.settings.debug = False
        server.settings.log_level = "INFO"
        server.settings.production_mode = True
        server.settings.optimize_memory = True
        server.settings.max_cached_items = 500
        server.settings.gc_interval = 300
        server.settings.port = 8888

        # Enhanced security settings
        server.settings.hide_error_details = True  # Hide detailed error messages

        enabled_features = ["✓ Detailed error messages hidden"]

        # Optional security features
        if rate_limit:
            server.settings.enable_rate_limit = True
            server.settings.rate_limit = {
                "requests": 100,
                "period": 60,  # requests per minute
            }
            enabled_features.append("✓ Rate limiting enabled")

        if request_validation:
            server.settings.validate_requests = True
            server.settings.max_request_size = 1024 * 1024  # 1MB
            enabled_features.append("✓ Request validation enabled")

        if security_headers:
            server.settings.security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'",
                "Referrer-Policy": "strict-origin-when-cross-origin",
            }
            enabled_features.append("✓ Security headers configured")

        # Run server
        console.print("[green]Production server started successfully![/]")
        if enabled_features:
            console.print("[yellow]Running with the following security settings:[/]")
            for feature in enabled_features:
                console.print(feature)

        asyncio.run(server.run(transport="sse"))

    except Exception:
        # In production, don't show detailed error information
        console.print(
            "[red]Server failed to start. Please check the logs for details.[/]"
        )
        return


@app.callback(invoke_without_command=False)
def callback():
    """Axiom MCP CLI - Use 'dev' for development mode or 'run' for production mode."""
    pass


@app.command()
def version():
    """Show version information."""
    print(f"Axiom MCP version {__version__}")
    return 0


@app.command()
def dev(
    file_spec: Annotated[
        str, typer.Argument(help="Python file to run in development mode")
    ],
    with_editable: Annotated[
        Optional[Path],
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
    ] = [],
    env_file: Annotated[
        Optional[Path],
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
    """Run an Axiom MCP server in development mode with hot reloading."""
    try:
        console.print("[green]Starting development server[/]")
        console.print("*" * 50)

        # Check if file exists
        try:
            file_path = Path(file_spec).resolve(strict=True)
        except (FileNotFoundError, RuntimeError):
            console.print(f"[red]File not found: {file_spec}[/]")
            raise typer.Exit(code=1)

        reloader = CodeReloader(file_path)
        if not no_reload:
            # Set up watchdog observer
            observer = Observer()
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
            # Run without hot reloading, but still use multiprocessing
            reloader.start_server()
            try:
                while reloader.process and reloader.process.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                if reloader.process and reloader.process.is_alive():
                    reloader.process.terminate()
                    reloader.process.join()
                console.print("\n[yellow]Shutting down server...[/]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/]")
        raise typer.Exit(code=1)


@app.command()
def run(
    file_spec: Annotated[
        str, typer.Argument(help="Python file to run in production mode")
    ],
    rate_limit: Annotated[
        bool,
        typer.Option("--rate-limit", help="Enable rate limiting (100 requests/minute)"),
    ] = False,
    request_validation: Annotated[
        bool,
        typer.Option(
            "--validate-requests", help="Enable request validation with 1MB size limit"
        ),
    ] = False,
    security_headers: Annotated[
        bool,
        typer.Option(
            "--security-headers", help="Enable security headers (HSTS, CSP, etc)"
        ),
    ] = False,
):
    """Run an Axiom MCP server in production mode."""
    try:
        file_path = Path(file_spec).resolve(strict=True)
        run_production_server(
            file_path, rate_limit, request_validation, security_headers
        )
    except (FileNotFoundError, RuntimeError):
        console.print(f"[red]File not found: {file_spec}[/]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]ERROR: {e}[/]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
