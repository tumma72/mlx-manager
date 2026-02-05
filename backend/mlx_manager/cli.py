"""MLX Manager CLI."""

import threading
import webbrowser

import typer
from rich.console import Console
from rich.table import Table

from mlx_manager import __version__
from mlx_manager.config import DEFAULT_PORT

app = typer.Typer(
    name="mlx-manager",
    help="MLX Model Manager - Manage MLX models on Apple Silicon",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload (dev)"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser on start"),
):
    """Start the MLX Manager web server."""
    import uvicorn

    console.print(f"[green]Starting MLX Manager on http://{host}:{port}[/green]")

    if open_browser:

        def open_delayed():
            import time

            time.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=open_delayed, daemon=True).start()

    uvicorn.run(
        "mlx_manager.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def install_service(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Port to bind to"),
):
    """Install MLX Manager as a launchd service (runs at login)."""
    from mlx_manager.services.manager_launchd import install_manager_service

    plist_path = install_manager_service(host=host, port=port)
    console.print(f"[green]Service installed: {plist_path}[/green]")
    console.print("MLX Manager will start automatically at login.")


@app.command()
def uninstall_service():
    """Remove MLX Manager launchd service."""
    from mlx_manager.services.manager_launchd import uninstall_manager_service

    if uninstall_manager_service():
        console.print("[green]Service uninstalled successfully[/green]")
    else:
        console.print("[yellow]Service was not installed[/yellow]")


@app.command()
def status():
    """Show status of running MLX servers."""
    import httpx

    try:
        response = httpx.get(f"http://127.0.0.1:{DEFAULT_PORT}/api/servers", timeout=5.0)
        servers = response.json()

        if not servers:
            console.print("[yellow]No servers running[/yellow]")
            return

        table = Table(title="Running MLX Servers")
        table.add_column("Profile", style="cyan")
        table.add_column("Port", style="green")
        table.add_column("PID", style="yellow")
        table.add_column("Status", style="magenta")
        table.add_column("Memory", style="blue")

        for server in servers:
            table.add_row(
                server["profile_name"],
                str(server["port"]),
                str(server["pid"]),
                server["health_status"],
                f"{server['memory_mb']:.1f} MB",
            )

        console.print(table)
    except httpx.ConnectError:
        console.print("[red]MLX Manager is not running[/red]")
        console.print("Start it with: mlx-manager serve")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def menubar():
    """Launch the status bar app."""
    from mlx_manager.menubar import run_menubar

    run_menubar()


@app.command()
def version():
    """Show version information."""
    console.print(f"MLX Manager v{__version__}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
