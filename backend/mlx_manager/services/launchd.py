"""macOS launchd service manager."""

import plistlib
import subprocess
import sys
from pathlib import Path

from mlx_manager.models import ServerProfile


class LaunchdManager:
    """Manages launchd service configuration."""

    def __init__(self):
        self.launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
        self.label_prefix = "com.mlx-manager"

    def get_label(self, profile: ServerProfile) -> str:
        """Get the launchd label for a profile."""
        # Sanitize profile name for use in label
        safe_name = profile.name.lower().replace(" ", "-").replace("_", "-")
        # Remove any other special characters
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "-")
        return f"{self.label_prefix}.{safe_name}"

    def get_plist_path(self, profile: ServerProfile) -> Path:
        """Get the plist file path for a profile."""
        return self.launch_agents_dir / f"{self.get_label(profile)}.plist"

    def generate_plist(self, profile: ServerProfile) -> dict:
        """Generate a launchd plist dictionary for a profile."""
        label = self.get_label(profile)

        # Build program arguments
        program_args = [
            sys.executable,
            "-m",
            "mlx_openai_server.main",
            "--model-path",
            profile.model_path,
            "--model-type",
            profile.model_type,
            "--port",
            str(profile.port),
            "--host",
            profile.host,
            "--max-concurrency",
            str(profile.max_concurrency),
            "--queue-timeout",
            str(profile.queue_timeout),
            "--queue-size",
            str(profile.queue_size),
        ]

        if profile.context_length:
            program_args.extend(["--context-length", str(profile.context_length)])

        if profile.tool_call_parser:
            program_args.extend(["--tool-call-parser", profile.tool_call_parser])

        if profile.reasoning_parser:
            program_args.extend(["--reasoning-parser", profile.reasoning_parser])

        if profile.enable_auto_tool_choice:
            program_args.append("--enable-auto-tool-choice")

        if profile.trust_remote_code:
            program_args.append("--trust-remote-code")

        program_args.extend(["--log-level", profile.log_level])
        program_args.append("--no-log-file")  # Use launchd logging instead

        # Build plist dictionary
        plist = {
            "Label": label,
            "ProgramArguments": program_args,
            "RunAtLoad": profile.auto_start,
            "KeepAlive": {"SuccessfulExit": False, "Crashed": True},
            "StandardOutPath": f"/tmp/{label}.log",
            "StandardErrorPath": f"/tmp/{label}.err",
            "EnvironmentVariables": {
                "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
                "HOME": str(Path.home()),
                "PYTHONUNBUFFERED": "1",
            },
            "ProcessType": "Interactive",
            "LowPriorityIO": False,
            "ThrottleInterval": 30,
        }

        return plist

    def install(self, profile: ServerProfile) -> str:
        """Install a launchd service for a profile."""
        # Ensure LaunchAgents directory exists
        self.launch_agents_dir.mkdir(parents=True, exist_ok=True)

        # Generate and write plist
        plist = self.generate_plist(profile)
        plist_path = self.get_plist_path(profile)

        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)

        # Load the service
        subprocess.run(["launchctl", "load", str(plist_path)], check=True, capture_output=True)

        return str(plist_path)

    def uninstall(self, profile: ServerProfile) -> bool:
        """Uninstall a launchd service."""
        plist_path = self.get_plist_path(profile)

        if not plist_path.exists():
            return False

        # Unload the service
        try:
            subprocess.run(
                ["launchctl", "unload", str(plist_path)], check=True, capture_output=True
            )
        except subprocess.CalledProcessError:
            pass  # May not be loaded

        # Remove plist file
        plist_path.unlink(missing_ok=True)

        return True

    def is_installed(self, profile: ServerProfile) -> bool:
        """Check if a launchd service is installed."""
        return self.get_plist_path(profile).exists()

    def is_running(self, profile: ServerProfile) -> bool:
        """Check if a launchd service is running."""
        label = self.get_label(profile)

        result = subprocess.run(["launchctl", "list", label], capture_output=True, text=True)

        return result.returncode == 0

    def start(self, profile: ServerProfile) -> bool:
        """Start a launchd service."""
        label = self.get_label(profile)

        result = subprocess.run(["launchctl", "start", label], capture_output=True)

        return result.returncode == 0

    def stop(self, profile: ServerProfile) -> bool:
        """Stop a launchd service."""
        label = self.get_label(profile)

        result = subprocess.run(["launchctl", "stop", label], capture_output=True)

        return result.returncode == 0

    def get_status(self, profile: ServerProfile) -> dict:
        """Get detailed status of a launchd service."""
        label = self.get_label(profile)

        result = subprocess.run(["launchctl", "list", label], capture_output=True, text=True)

        if result.returncode != 0:
            return {
                "installed": self.is_installed(profile),
                "running": False,
                "label": label,
            }

        # Parse output: PID\tStatus\tLabel
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 1:
            parts = lines[0].split("\t")
            pid = int(parts[0]) if parts[0] != "-" else None

            return {
                "installed": True,
                "running": pid is not None,
                "pid": pid,
                "label": label,
            }

        return {"installed": True, "running": False, "label": label}


# Singleton instance
launchd_manager = LaunchdManager()
