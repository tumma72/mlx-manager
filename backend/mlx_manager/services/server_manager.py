"""Server process manager service."""

import asyncio
import logging
import signal
import subprocess

import httpx
import psutil

from mlx_manager.models import ServerProfile

logger = logging.getLogger(__name__)
from mlx_manager.types import HealthCheckResult, RunningServerInfo, ServerStats
from mlx_manager.utils.command_builder import build_mlx_server_command, get_server_log_path


class ServerManager:
    """Manages mlx-openai-server processes."""

    def __init__(self) -> None:
        self.processes: dict[int, subprocess.Popen[bytes]] = {}  # profile_id -> process
        self._log_positions: dict[int, int] = {}  # profile_id -> last read position

    async def start_server(self, profile: ServerProfile) -> int:
        """Start an mlx-openai-server instance for the given profile."""
        # Profile must be persisted to have an ID
        assert profile.id is not None, "Profile must be saved before starting server"

        logger.info(f"Starting server for profile '{profile.name}' (id={profile.id})")

        # Check if already running
        if profile.id in self.processes:
            proc = self.processes[profile.id]
            if proc.poll() is None:  # Still running
                logger.warning(f"Server for profile '{profile.name}' is already running (pid={proc.pid})")
                raise RuntimeError(f"Server for profile {profile.name} is already running")

        # Build command
        cmd = build_mlx_server_command(profile)
        logger.info(f"Command: {' '.join(cmd)}")

        # Prepare log file for capturing output
        log_path = get_server_log_path(profile.id)
        if log_path.exists():
            log_path.unlink()
        logger.debug(f"Log file: {log_path}")

        # Start process with stdout/stderr redirected to log file
        # mlx-openai-server doesn't have a --log-file option, so we capture output ourselves
        log_file = open(log_path, "w")
        self._log_files: dict[int, object] = getattr(self, "_log_files", {})
        self._log_files[profile.id] = log_file

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
        )
        logger.info(f"Process spawned with pid={proc.pid}")

        self.processes[profile.id] = proc
        self._log_positions[profile.id] = 0

        # Wait briefly for startup
        await asyncio.sleep(2)

        # Check if process is still alive
        if proc.poll() is not None:
            # Process exited - flush and read log file for error
            error_msg = ""
            try:
                log_file.flush()
                log_file.close()
                if profile.id in self._log_files:
                    del self._log_files[profile.id]
            except Exception:
                pass
            if log_path.exists():
                error_msg = log_path.read_text()[-2000:]  # Last 2000 chars
            logger.error(f"Server failed to start for profile '{profile.name}' (exit_code={proc.poll()})")
            logger.error(f"Server log: {error_msg[:500]}")
            del self.processes[profile.id]
            raise RuntimeError(f"Server failed to start: {error_msg}")

        logger.info(f"Server started successfully for profile '{profile.name}' (pid={proc.pid})")
        return proc.pid

    async def stop_server(self, profile_id: int, force: bool = False) -> bool:
        """Stop a running server."""
        logger.info(f"Stopping server for profile_id={profile_id} (force={force})")

        if profile_id not in self.processes:
            logger.debug(f"No process found for profile_id={profile_id}")
            return False

        proc = self.processes[profile_id]

        if proc.poll() is not None:
            # Already stopped
            logger.debug(f"Process already stopped for profile_id={profile_id}")
            del self.processes[profile_id]
            self._log_positions.pop(profile_id, None)
            return True

        # Send SIGTERM (graceful) or SIGKILL (force)
        sig = signal.SIGKILL if force else signal.SIGTERM
        proc.send_signal(sig)
        logger.debug(f"Sent signal {sig} to pid={proc.pid}")

        # Wait for process to exit
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(f"Process did not exit gracefully, killing pid={proc.pid}")
            proc.kill()
            proc.wait()

        del self.processes[profile_id]
        self._log_positions.pop(profile_id, None)
        # Close log file if open
        if hasattr(self, "_log_files") and profile_id in self._log_files:
            try:
                self._log_files[profile_id].close()
            except Exception:
                pass
            del self._log_files[profile_id]
        logger.info(f"Server stopped for profile_id={profile_id}")
        return True

    async def check_health(self, profile: ServerProfile) -> HealthCheckResult:
        """Check health of a running server.

        Uses /v1/models endpoint since mlx-openai-server doesn't have /health.
        """
        url = f"http://{profile.host}:{profile.port}/v1/models"

        try:
            async with httpx.AsyncClient() as client:
                start = asyncio.get_event_loop().time()
                response = await client.get(url, timeout=5.0)
                elapsed = (asyncio.get_event_loop().time() - start) * 1000

                if response.status_code == 200:
                    # Check if models are loaded
                    data = response.json()
                    model_loaded = bool(data.get("data"))
                    return HealthCheckResult(
                        status="healthy",
                        response_time_ms=round(elapsed, 2),
                        model_loaded=model_loaded,
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

    def get_log_lines(self, profile_id: int, max_lines: int = 100) -> list[str]:
        """Get new log lines from a running server's log file."""
        if profile_id not in self.processes:
            return []

        log_path = get_server_log_path(profile_id)
        if not log_path.exists():
            return []

        # Flush log file to ensure we read latest content
        if hasattr(self, "_log_files") and profile_id in self._log_files:
            try:
                self._log_files[profile_id].flush()
            except Exception:
                pass

        lines: list[str] = []
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                # Seek to last read position
                last_pos = self._log_positions.get(profile_id, 0)
                f.seek(last_pos)

                # Read new lines
                new_lines = f.readlines()
                lines = [line.rstrip() for line in new_lines[-max_lines:]]

                # Update position
                self._log_positions[profile_id] = f.tell()
        except Exception:
            pass

        return lines

    def is_running(self, profile_id: int) -> bool:
        """Check if a server is running for the given profile."""
        if profile_id not in self.processes:
            return False
        return self.processes[profile_id].poll() is None

    def get_process_status(self, profile_id: int) -> dict:
        """Get detailed process status including exit code and error message."""
        if profile_id not in self.processes:
            # Process is not tracked - but check if there's a recent log file with errors
            # This handles the case where the process crashed and was cleaned up before
            # the status was checked
            log_path = get_server_log_path(profile_id)
            if log_path.exists():
                try:
                    content = log_path.read_text()
                    if content:
                        error_msg = content[-1000:]
                        error_patterns = ["ERROR", "Error", "failed", "Failed", "exception", "Exception"]
                        has_error = any(pattern in error_msg for pattern in error_patterns)
                        if has_error:
                            return {
                                "running": False,
                                "tracked": False,
                                "failed": True,
                                "error_message": error_msg,
                            }
                except Exception:
                    pass
            return {"running": False, "tracked": False, "failed": False}

        proc = self.processes[profile_id]
        exit_code = proc.poll()

        if exit_code is not None:
            # Process has exited - close log file and read error
            log_path = get_server_log_path(profile_id)
            if hasattr(self, "_log_files") and profile_id in self._log_files:
                try:
                    self._log_files[profile_id].flush()
                    self._log_files[profile_id].close()
                except Exception:
                    pass
                del self._log_files[profile_id]

            error_msg = None
            has_error_in_log = False
            if log_path.exists():
                content = log_path.read_text()
                error_msg = content[-1000:] if content else "No log output"
                # Check for error patterns in log (mlx-openai-server may exit with code 0 on error)
                if error_msg:
                    error_patterns = ["ERROR", "Error", "failed", "Failed", "exception", "Exception"]
                    has_error_in_log = any(pattern in error_msg for pattern in error_patterns)

            # Clean up the dead process
            del self.processes[profile_id]
            self._log_positions.pop(profile_id, None)

            # Consider it failed if exit code is non-zero OR if log contains error patterns
            is_failed = exit_code != 0 or has_error_in_log

            return {
                "running": False,
                "tracked": True,
                "exit_code": exit_code,
                "failed": is_failed,
                "error_message": error_msg,
            }

        return {
            "running": True,
            "tracked": True,
            "pid": proc.pid,
        }

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
        logger.info(f"Cleaning up {len(self.processes)} running servers...")
        for profile_id in list(self.processes.keys()):
            await self.stop_server(profile_id, force=True)
        logger.info("Cleanup complete")


# Singleton instance
server_manager = ServerManager()
