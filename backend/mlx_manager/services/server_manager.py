"""Server process manager service."""

import asyncio
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import httpx
import psutil

from mlx_manager.models import ServerProfile


def _find_mlx_openai_server() -> str:
    """Find the mlx-openai-server executable.

    First checks the same directory as the Python executable (for venv installs),
    then falls back to system PATH.
    """
    # Check alongside the Python executable (handles venv correctly)
    python_dir = Path(sys.executable).parent
    local_cmd = python_dir / "mlx-openai-server"
    if local_cmd.exists():
        return str(local_cmd)

    # Fall back to PATH lookup
    path_cmd = shutil.which("mlx-openai-server")
    if path_cmd:
        return path_cmd

    raise RuntimeError(
        "mlx-openai-server not found. Please install it with: pip install mlx-openai-server"
    )


class ServerManager:
    """Manages mlx-openai-server processes."""

    def __init__(self):
        self.processes: dict[int, subprocess.Popen] = {}  # profile_id -> process

    async def start_server(self, profile: ServerProfile) -> int:
        """Start an mlx-openai-server instance for the given profile."""
        # Check if already running
        if profile.id in self.processes:
            proc = self.processes[profile.id]
            if proc.poll() is None:  # Still running
                raise RuntimeError(f"Server for profile {profile.name} is already running")

        # Build command
        cmd = self._build_command(profile)

        # Start process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        self.processes[profile.id] = proc

        # Wait briefly for startup
        await asyncio.sleep(2)

        # Check if process is still alive
        if proc.poll() is not None:
            # Process exited
            stdout = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"Server failed to start: {stdout}")

        return proc.pid

    def _build_command(self, profile: ServerProfile) -> list[str]:
        """Build the mlx-openai-server command from profile.

        Note: mlx-openai-server CLI uses 'launch' subcommand and supports only:
        --model-path, --model-type (lm|multimodal), --port, --host,
        --max-concurrency, --queue-timeout, --queue-size
        """
        # Map our model types to mlx-openai-server supported types
        # mlx-openai-server only supports 'lm' and 'multimodal'
        model_type = profile.model_type
        if model_type not in ("lm", "multimodal"):
            model_type = "lm"  # Default to lm for unsupported types

        cmd = [
            _find_mlx_openai_server(),
            "launch",  # Required subcommand
            "--model-path",
            profile.model_path,
            "--model-type",
            model_type,
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

        return cmd

    async def stop_server(self, profile_id: int, force: bool = False) -> bool:
        """Stop a running server."""
        if profile_id not in self.processes:
            return False

        proc = self.processes[profile_id]

        if proc.poll() is not None:
            # Already stopped
            del self.processes[profile_id]
            return True

        # Send SIGTERM (graceful) or SIGKILL (force)
        sig = signal.SIGKILL if force else signal.SIGTERM
        proc.send_signal(sig)

        # Wait for process to exit
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        del self.processes[profile_id]
        return True

    async def check_health(self, profile: ServerProfile) -> dict:
        """Check health of a running server."""
        url = f"http://{profile.host}:{profile.port}/health"

        try:
            async with httpx.AsyncClient() as client:
                start = asyncio.get_event_loop().time()
                response = await client.get(url, timeout=5.0)
                elapsed = (asyncio.get_event_loop().time() - start) * 1000

                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "response_time_ms": round(elapsed, 2),
                        "model_loaded": True,
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "response_time_ms": round(elapsed, 2),
                        "error": f"HTTP {response.status_code}",
                    }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_server_stats(self, profile_id: int) -> dict | None:
        """Get memory and CPU stats for a running server."""
        if profile_id not in self.processes:
            return None

        proc = self.processes[profile_id]
        if proc.poll() is not None:
            return None

        try:
            p = psutil.Process(proc.pid)
            memory_info = p.memory_info()

            return {
                "pid": proc.pid,
                "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": p.cpu_percent(),
                "status": p.status(),
                "create_time": p.create_time(),
            }
        except psutil.NoSuchProcess:
            return None

    def get_log_lines(self, profile_id: int) -> list[str]:
        """Get available log lines from a running server."""
        if profile_id not in self.processes:
            return []

        proc = self.processes[profile_id]
        if proc.stdout is None:
            return []

        lines = []
        try:
            # Non-blocking read of available lines
            import select

            while select.select([proc.stdout], [], [], 0)[0]:
                line = proc.stdout.readline()
                if not line:
                    break
                lines.append(line.strip())
        except Exception:
            pass

        return lines

    def is_running(self, profile_id: int) -> bool:
        """Check if a server is running for the given profile."""
        if profile_id not in self.processes:
            return False
        return self.processes[profile_id].poll() is None

    def get_all_running(self) -> list[dict]:
        """Get info about all running servers."""
        running = []

        for profile_id, proc in list(self.processes.items()):
            if proc.poll() is not None:
                # Process has exited, clean up
                del self.processes[profile_id]
                continue

            stats = self.get_server_stats(profile_id)
            if stats:
                running.append({"profile_id": profile_id, **stats})

        return running

    async def cleanup(self):
        """Stop all servers on shutdown."""
        for profile_id in list(self.processes.keys()):
            await self.stop_server(profile_id, force=True)


# Singleton instance
server_manager = ServerManager()
