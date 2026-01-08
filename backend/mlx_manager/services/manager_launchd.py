"""Launchd service for MLX Manager itself (not MLX servers)."""

import plistlib
import subprocess
import sys
from pathlib import Path

LABEL = "com.mlx-manager.app"


def get_plist_path() -> Path:
    """Get the plist file path."""
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def install_manager_service(host: str = "127.0.0.1", port: int = 8080) -> str:
    """Install MLX Manager as a launchd service."""
    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
    launch_agents_dir.mkdir(parents=True, exist_ok=True)

    # Find the mlx-manager executable or use python module
    mlx_manager_path = Path(sys.executable).parent / "mlx-manager"

    if mlx_manager_path.exists():
        program_args = [
            str(mlx_manager_path),
            "serve",
            "--host",
            host,
            "--port",
            str(port),
            "--no-open",
        ]
    else:
        program_args = [
            sys.executable,
            "-m",
            "mlx_manager.cli",
            "serve",
            "--host",
            host,
            "--port",
            str(port),
            "--no-open",
        ]

    plist = {
        "Label": LABEL,
        "ProgramArguments": program_args,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False, "Crashed": True},
        "StandardOutPath": f"/tmp/{LABEL}.log",
        "StandardErrorPath": f"/tmp/{LABEL}.err",
        "EnvironmentVariables": {
            "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(Path.home()),
            "PYTHONUNBUFFERED": "1",
        },
        "ProcessType": "Interactive",
        "ThrottleInterval": 30,
    }

    plist_path = get_plist_path()

    # Unload if already loaded
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True, check=False)

    with open(plist_path, "wb") as f:
        plistlib.dump(plist, f)

    subprocess.run(["launchctl", "load", str(plist_path)], check=True)

    return str(plist_path)


def uninstall_manager_service() -> bool:
    """Uninstall the MLX Manager launchd service."""
    plist_path = get_plist_path()

    if not plist_path.exists():
        return False

    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True, check=False)
    plist_path.unlink(missing_ok=True)

    return True


def is_service_installed() -> bool:
    """Check if the service is installed."""
    return get_plist_path().exists()


def is_service_running() -> bool:
    """Check if the service is running."""
    result = subprocess.run(["launchctl", "list", LABEL], capture_output=True, text=True)
    return result.returncode == 0
