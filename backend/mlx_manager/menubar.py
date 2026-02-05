"""MLX Manager Status Bar App."""

import subprocess
import sys
import time
import webbrowser
from typing import Any

import httpx
import rumps  # type: ignore[import-untyped,import-not-found]
from loguru import logger

from mlx_manager import __version__
from mlx_manager.config import DEFAULT_PORT


class MLXManagerApp(rumps.App):
    """macOS menubar application for MLX Manager."""

    def __init__(self) -> None:
        super().__init__(
            "MLX",
            icon=None,  # Could add icon path here
            quit_button=None,  # Custom quit handling
        )

        self.server_process: subprocess.Popen | None = None
        self.server_host = "127.0.0.1"
        self.server_port = DEFAULT_PORT
        self.health_check_interval = 5

        # Build menu
        self.menu = [
            rumps.MenuItem("Open Dashboard", callback=self.open_dashboard),
            None,  # Separator
            rumps.MenuItem("Start Server", callback=self.start_server),
            rumps.MenuItem("Stop Server", callback=self.stop_server),
            None,  # Separator
            rumps.MenuItem("Status: Checking...", callback=None),
            None,  # Separator
            rumps.MenuItem(f"v{__version__}", callback=None),
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

        # Disable stop initially
        # rumps.App.menu supports string indexing at runtime
        self.menu["Stop Server"].set_callback(None)  # type: ignore[call-overload]

        # Auto-start server
        self._auto_start()

    def _auto_start(self) -> None:
        """Auto-start the server on launch."""
        # Check if already running (e.g., from launchd)
        if self._check_server_running():
            self._update_status("Running", is_running=True)
            rumps.notification(
                "MLX Manager",
                "Server Already Running",
                f"Connected to http://{self.server_host}:{self.server_port}",
            )
        else:
            # Start the server
            self.start_server(None)

    def _check_server_running(self) -> bool:
        """Check if server is responding to health checks."""
        try:
            response = httpx.get(
                f"http://{self.server_host}:{self.server_port}/health",
                timeout=2.0,
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Server status check failed: {e}")
            return False

    def _update_status(self, status: str, is_running: bool) -> None:
        """Update menu status item and icon."""
        # rumps.App.menu supports string indexing at runtime
        self.menu["Status: Checking..."].title = f"Status: {status}"  # type: ignore[call-overload]

        if is_running:
            self.title = "MLX \u25cf"  # Filled circle
            self.menu["Start Server"].set_callback(None)  # type: ignore[call-overload]
            self.menu["Stop Server"].set_callback(self.stop_server)  # type: ignore[call-overload]
        else:
            self.title = "MLX \u25cb"  # Empty circle
            self.menu["Start Server"].set_callback(self.start_server)  # type: ignore[call-overload]
            self.menu["Stop Server"].set_callback(None)  # type: ignore[call-overload]

    @rumps.timer(5)
    def check_health(self, _: Any) -> None:
        """Periodically check server health."""
        if self._check_server_running():
            self._update_status("Running", is_running=True)
        else:
            # Check if our process is still alive
            if self.server_process and self.server_process.poll() is not None:
                self.server_process = None

            self._update_status("Stopped", is_running=False)

    def start_server(self, _: Any) -> None:
        """Start the MLX Manager server."""
        if self.server_process and self.server_process.poll() is None:
            rumps.notification(
                "MLX Manager",
                "Already Running",
                "Server is already running",
            )
            return

        # Check if another instance is running
        if self._check_server_running():
            rumps.notification(
                "MLX Manager",
                "Already Running",
                "Another server instance is already running",
            )
            self._update_status("Running", is_running=True)
            return

        try:
            # Start uvicorn as subprocess
            python_path = sys.executable
            self.server_process = subprocess.Popen(
                [
                    python_path,
                    "-m",
                    "uvicorn",
                    "mlx_manager.main:app",
                    "--host",
                    self.server_host,
                    "--port",
                    str(self.server_port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Wait for startup
            time.sleep(2)

            if self._check_server_running():
                self._update_status("Running", is_running=True)
                rumps.notification(
                    "MLX Manager",
                    "Server Started",
                    f"Running on http://{self.server_host}:{self.server_port}",
                )
            else:
                rumps.notification(
                    "MLX Manager",
                    "Start Failed",
                    "Server failed to start. Check logs.",
                )
                self._update_status("Failed", is_running=False)

        except Exception as e:
            rumps.notification(
                "MLX Manager",
                "Error",
                f"Failed to start server: {e}",
            )
            self._update_status("Error", is_running=False)

    def stop_server(self, _: Any) -> None:
        """Stop the MLX Manager server."""
        if self.server_process and self.server_process.poll() is None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()

            self.server_process = None

            rumps.notification(
                "MLX Manager",
                "Server Stopped",
                "The server has been stopped",
            )

        self._update_status("Stopped", is_running=False)

    def open_dashboard(self, _: Any) -> None:
        """Open the web dashboard in default browser."""
        if self._check_server_running():
            webbrowser.open(f"http://{self.server_host}:{self.server_port}")
        else:
            rumps.notification(
                "MLX Manager",
                "Server Not Running",
                "Start the server first to access the dashboard",
            )

    def quit_app(self, _: Any) -> None:
        """Quit the application."""
        # Ask about stopping server if running
        if self.server_process and self.server_process.poll() is None:
            response = rumps.alert(
                title="Quit MLX Manager",
                message="The server is still running. Do you want to stop it before quitting?",
                ok="Stop and Quit",
                cancel="Cancel",
                other="Quit Without Stopping",
            )

            if response == 0:  # Cancel
                return
            elif response == 1:  # Stop and Quit
                self.stop_server(None)

        rumps.quit_application()


def run_menubar() -> None:
    """Run the menubar application."""
    MLXManagerApp().run()


def main() -> None:
    """Entry point for menubar app."""
    run_menubar()


if __name__ == "__main__":
    main()
