"""Server process manager service."""

import asyncio
import signal
import subprocess

import httpx
import psutil

from mlx_manager.models import ServerProfile
from mlx_manager.types import HealthCheckResult, RunningServerInfo, ServerStats
from mlx_manager.utils.command_builder import build_mlx_server_command


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
        cmd = build_mlx_server_command(profile)

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

    async def check_health(self, profile: ServerProfile) -> HealthCheckResult:
        """Check health of a running server."""
        url = f"http://{profile.host}:{profile.port}/health"

        try:
            async with httpx.AsyncClient() as client:
                start = asyncio.get_event_loop().time()
                response = await client.get(url, timeout=5.0)
                elapsed = (asyncio.get_event_loop().time() - start) * 1000

                if response.status_code == 200:
                    return HealthCheckResult(
                        status="healthy",
                        response_time_ms=round(elapsed, 2),
                        model_loaded=True,
                    )
                else:
                    return HealthCheckResult(
                        status="unhealthy",
                        response_time_ms=round(elapsed, 2),
                        error=f"HTTP {response.status_code}",
                    )
        except Exception as e:
            return HealthCheckResult(status="unhealthy", error=str(e))

    def get_server_stats(self, profile_id: int) -> ServerStats | None:
        """Get memory and CPU stats for a running server."""
        if profile_id not in self.processes:
            return None

        proc = self.processes[profile_id]
        if proc.poll() is not None:
            return None

        try:
            p = psutil.Process(proc.pid)
            memory_info = p.memory_info()

            return ServerStats(
                pid=proc.pid,
                memory_mb=round(memory_info.rss / 1024 / 1024, 2),
                cpu_percent=p.cpu_percent(),
                status=p.status(),
                create_time=p.create_time(),
            )
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

    def get_all_running(self) -> list[RunningServerInfo]:
        """Get info about all running servers."""
        running: list[RunningServerInfo] = []

        for profile_id, proc in list(self.processes.items()):
            if proc.poll() is not None:
                # Process has exited, clean up
                del self.processes[profile_id]
                continue

            stats = self.get_server_stats(profile_id)
            if stats:
                running.append(
                    RunningServerInfo(
                        profile_id=profile_id,
                        pid=stats["pid"],
                        memory_mb=stats["memory_mb"],
                        cpu_percent=stats["cpu_percent"],
                        status=stats["status"],
                        create_time=stats["create_time"],
                    )
                )

        return running

    async def cleanup(self) -> None:
        """Stop all servers on shutdown."""
        for profile_id in list(self.processes.keys()):
            await self.stop_server(profile_id, force=True)


# Singleton instance
server_manager = ServerManager()
