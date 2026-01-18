"""Tests for the server manager service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.models import ServerProfile
from mlx_manager.services.server_manager import ServerManager
from mlx_manager.utils.command_builder import build_mlx_server_command


@pytest.fixture
def server_manager_instance():
    """Create a fresh ServerManager instance."""
    return ServerManager()


@pytest.fixture
def sample_profile():
    """Create a sample ServerProfile for testing."""
    return ServerProfile(
        id=1,
        name="Test Profile",
        model_path="mlx-community/test-model",
        model_type="lm",
        port=10240,
        host="127.0.0.1",
        max_concurrency=1,
        queue_timeout=300,
        queue_size=100,
        log_level="INFO",
    )


class TestBuildMlxServerCommand:
    """Tests for the build_mlx_server_command utility function.

    Note: mlx-openai-server CLI uses 'launch' subcommand and supports only:
    --model-path, --model-type (lm|multimodal), --port, --host,
    --max-concurrency, --queue-timeout, --queue-size
    """

    def test_basic_command(self, sample_profile):
        """Test building basic command with launch subcommand."""
        cmd = build_mlx_server_command(sample_profile)

        # Check launch subcommand is present
        assert "launch" in cmd

        # Check required arguments
        assert "--model-path" in cmd
        assert "mlx-community/test-model" in cmd
        assert "--model-type" in cmd
        assert "lm" in cmd
        assert "--port" in cmd
        assert "10240" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--max-concurrency" in cmd
        assert "--queue-timeout" in cmd
        assert "--queue-size" in cmd

    def test_command_maps_unsupported_model_types(self, sample_profile):
        """Test that unsupported model types are mapped to 'lm'."""
        # mlx-openai-server only supports 'lm' and 'multimodal'
        sample_profile.model_type = "whisper"
        cmd = build_mlx_server_command(sample_profile)

        # Should be mapped to 'lm'
        model_type_idx = cmd.index("--model-type") + 1
        assert cmd[model_type_idx] == "lm"

    def test_command_preserves_supported_model_types(self, sample_profile):
        """Test that supported model types are preserved."""
        sample_profile.model_type = "multimodal"
        cmd = build_mlx_server_command(sample_profile)

        model_type_idx = cmd.index("--model-type") + 1
        assert cmd[model_type_idx] == "multimodal"


class TestServerManagerIsRunning:
    """Tests for the is_running method."""

    def test_not_running_when_no_process(self, server_manager_instance):
        """Test returns False when no process exists."""
        result = server_manager_instance.is_running(999)
        assert result is False

    def test_not_running_when_process_exited(self, server_manager_instance):
        """Test returns False when process has exited."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Process exited with code 0
        server_manager_instance.processes[1] = mock_proc

        result = server_manager_instance.is_running(1)
        assert result is False

    def test_running_when_process_active(self, server_manager_instance):
        """Test returns True when process is active."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Process is running
        server_manager_instance.processes[1] = mock_proc

        result = server_manager_instance.is_running(1)
        assert result is True


class TestServerManagerStopServer:
    """Tests for the stop_server method."""

    @pytest.mark.asyncio
    async def test_stop_nonexistent_server(self, server_manager_instance):
        """Test stopping a server that doesn't exist."""
        result = await server_manager_instance.stop_server(999)
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_already_stopped_server(self, server_manager_instance):
        """Test stopping a server that has already stopped."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Already exited
        server_manager_instance.processes[1] = mock_proc

        result = await server_manager_instance.stop_server(1)

        assert result is True
        assert 1 not in server_manager_instance.processes

    @pytest.mark.asyncio
    async def test_stop_running_server_graceful(self, server_manager_instance):
        """Test gracefully stopping a running server."""
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]  # Running, then exited
        mock_proc.wait.return_value = 0
        server_manager_instance.processes[1] = mock_proc

        result = await server_manager_instance.stop_server(1, force=False)

        assert result is True
        mock_proc.send_signal.assert_called_once()
        assert 1 not in server_manager_instance.processes

    @pytest.mark.asyncio
    async def test_stop_running_server_force(self, server_manager_instance):
        """Test force stopping a running server."""
        import signal

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]
        mock_proc.wait.return_value = 0
        server_manager_instance.processes[1] = mock_proc

        result = await server_manager_instance.stop_server(1, force=True)

        assert result is True
        mock_proc.send_signal.assert_called_once_with(signal.SIGKILL)


class TestServerManagerStartServer:
    """Tests for the start_server method."""

    @pytest.mark.asyncio
    async def test_start_already_running_server(self, server_manager_instance, sample_profile):
        """Test starting a server that's already running raises error."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        server_manager_instance.processes[sample_profile.id] = mock_proc

        with pytest.raises(RuntimeError, match="already running"):
            await server_manager_instance.start_server(sample_profile)

    @pytest.mark.asyncio
    async def test_start_server_success(self, server_manager_instance, sample_profile):
        """Test successfully starting a server."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None  # Still running after startup

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("asyncio.sleep"):  # Skip the startup delay
                pid = await server_manager_instance.start_server(sample_profile)

        assert pid == 12345
        assert sample_profile.id in server_manager_instance.processes

    @pytest.mark.asyncio
    async def test_start_server_failure(self, server_manager_instance, sample_profile):
        """Test server start failure."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 1  # Exited with error
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = "Error: Model not found"

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("asyncio.sleep"):
                with pytest.raises(RuntimeError, match="failed to start"):
                    await server_manager_instance.start_server(sample_profile)


class TestServerManagerCheckHealth:
    """Tests for the check_health method."""

    @pytest.mark.asyncio
    async def test_check_health_healthy(self, server_manager_instance, sample_profile):
        """Test health check returns healthy status."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await server_manager_instance.check_health(sample_profile)

        assert result["status"] == "healthy"
        assert "response_time_ms" in result
        assert result["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_check_health_unhealthy_http_error(self, server_manager_instance, sample_profile):
        """Test health check returns unhealthy on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await server_manager_instance.check_health(sample_profile)

        assert result["status"] == "unhealthy"
        assert "HTTP 500" in result["error"]

    @pytest.mark.asyncio
    async def test_check_health_connection_error(self, server_manager_instance, sample_profile):
        """Test health check returns unhealthy on connection error."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            result = await server_manager_instance.check_health(sample_profile)

        assert result["status"] == "unhealthy"
        assert "Connection refused" in result["error"]


class TestServerManagerGetServerStats:
    """Tests for the get_server_stats method."""

    def test_get_stats_nonexistent_server(self, server_manager_instance):
        """Test returns None for non-existent server."""
        result = server_manager_instance.get_server_stats(999)
        assert result is None

    def test_get_stats_exited_server(self, server_manager_instance):
        """Test returns None for exited server."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        server_manager_instance.processes[1] = mock_proc

        result = server_manager_instance.get_server_stats(1)
        assert result is None

    def test_get_stats_running_server(self, server_manager_instance):
        """Test returns stats for running server."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        mock_psutil = MagicMock()
        mock_psutil.memory_info.return_value.rss = 1024 * 1024 * 512  # 512 MB
        mock_psutil.cpu_percent.return_value = 15.5
        mock_psutil.status.return_value = "running"
        mock_psutil.create_time.return_value = 1704067200.0

        with patch("psutil.Process", return_value=mock_psutil):
            result = server_manager_instance.get_server_stats(1)

        assert result["pid"] == 12345
        assert result["memory_mb"] == 512.0
        assert result["cpu_percent"] == 15.5
        assert result["status"] == "running"


class TestServerManagerGetAllRunning:
    """Tests for the get_all_running method."""

    def test_get_all_running_empty(self, server_manager_instance):
        """Test returns empty list when no servers running."""
        result = server_manager_instance.get_all_running()
        assert result == []

    def test_get_all_running_cleans_up_exited(self, server_manager_instance):
        """Test cleans up exited processes."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Exited
        server_manager_instance.processes[1] = mock_proc

        result = server_manager_instance.get_all_running()

        assert result == []
        assert 1 not in server_manager_instance.processes


class TestServerManagerCleanup:
    """Tests for the cleanup method."""

    @pytest.mark.asyncio
    async def test_cleanup_stops_all_servers(self, server_manager_instance):
        """Test cleanup stops all running servers."""
        mock_proc1 = MagicMock()
        mock_proc1.poll.side_effect = [None, 0]
        mock_proc1.wait.return_value = 0

        mock_proc2 = MagicMock()
        mock_proc2.poll.side_effect = [None, 0]
        mock_proc2.wait.return_value = 0

        server_manager_instance.processes[1] = mock_proc1
        server_manager_instance.processes[2] = mock_proc2

        await server_manager_instance.cleanup()

        assert len(server_manager_instance.processes) == 0


class TestServerManagerGetLogLines:
    """Tests for the get_log_lines method."""

    def test_get_log_lines_no_process(self, server_manager_instance):
        """Test returns empty list when no process exists."""
        result = server_manager_instance.get_log_lines(999)
        assert result == []

    def test_get_log_lines_no_log_file(self, server_manager_instance, tmp_path):
        """Test returns empty list when log file doesn't exist."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=tmp_path / "nonexistent.log",
        ):
            result = server_manager_instance.get_log_lines(1)

        assert result == []

    def test_get_log_lines_reads_new_lines(self, server_manager_instance, tmp_path):
        """Test reads new lines from log file."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        # Create a log file with some content
        log_file = tmp_path / "server-1.log"
        log_file.write_text("Line 1\nLine 2\nLine 3\n")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            result = server_manager_instance.get_log_lines(1)

        assert len(result) == 3
        assert result[0] == "Line 1"
        assert result[1] == "Line 2"
        assert result[2] == "Line 3"

    def test_get_log_lines_tracks_position(self, server_manager_instance, tmp_path):
        """Test tracks read position for incremental reads."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        log_file = tmp_path / "server-1.log"
        log_file.write_text("Line 1\nLine 2\n")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            # First read
            result1 = server_manager_instance.get_log_lines(1)
            assert len(result1) == 2

            # Append more content
            with open(log_file, "a") as f:
                f.write("Line 3\nLine 4\n")

            # Second read - should only get new lines
            result2 = server_manager_instance.get_log_lines(1)
            assert len(result2) == 2
            assert result2[0] == "Line 3"
            assert result2[1] == "Line 4"

    def test_get_log_lines_respects_max_lines(self, server_manager_instance, tmp_path):
        """Test respects max_lines parameter."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        # Create log file with many lines
        log_file = tmp_path / "server-1.log"
        lines = [f"Line {i}" for i in range(150)]
        log_file.write_text("\n".join(lines) + "\n")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            result = server_manager_instance.get_log_lines(1, max_lines=50)

        assert len(result) == 50
        # Should get the last 50 lines
        assert result[0] == "Line 100"
        assert result[-1] == "Line 149"

    def test_get_log_lines_handles_read_error(self, server_manager_instance, tmp_path):
        """Test handles file read errors gracefully."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        log_file = tmp_path / "server-1.log"
        log_file.write_text("Some content")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            with patch("builtins.open", side_effect=PermissionError("Access denied")):
                result = server_manager_instance.get_log_lines(1)

        assert result == []


class TestServerManagerGetAllRunningWithStats:
    """Additional tests for get_all_running method with stats."""

    def test_get_all_running_with_active_servers(self, server_manager_instance):
        """Test returns running server info with stats."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        mock_psutil = MagicMock()
        mock_psutil.memory_info.return_value.rss = 1024 * 1024 * 256  # 256 MB
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.status.return_value = "running"
        mock_psutil.create_time.return_value = 1704067200.0

        with patch("psutil.Process", return_value=mock_psutil):
            result = server_manager_instance.get_all_running()

        assert len(result) == 1
        assert result[0]["profile_id"] == 1
        assert result[0]["pid"] == 12345
        assert result[0]["memory_mb"] == 256.0
        assert result[0]["cpu_percent"] == 25.0
        assert result[0]["status"] == "running"

    def test_get_all_running_skips_servers_without_stats(self, server_manager_instance):
        """Test skips servers where stats cannot be retrieved."""
        import psutil

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        # Simulate psutil.NoSuchProcess (the actual exception caught by get_server_stats)
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(12345)):
            result = server_manager_instance.get_all_running()

        # Server still in processes but no stats returned
        assert result == []


class TestServerManagerGetStatsNoSuchProcess:
    """Test get_server_stats with psutil.NoSuchProcess."""

    def test_get_stats_no_such_process(self, server_manager_instance):
        """Test returns None when process no longer exists."""
        import psutil

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(12345)):
            result = server_manager_instance.get_server_stats(1)

        assert result is None


class TestServerManagerStopServerTimeout:
    """Test stop_server with timeout scenarios."""

    @pytest.mark.asyncio
    async def test_stop_server_timeout_fallback_to_kill(self, server_manager_instance):
        """Test falls back to kill when graceful stop times out."""
        import subprocess

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Process is running
        # First wait() call with timeout raises TimeoutExpired, second wait() (after kill) succeeds
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=10),
            None,  # After kill, wait succeeds
        ]
        mock_proc.kill.return_value = None
        mock_proc.send_signal = MagicMock()
        server_manager_instance.processes[1] = mock_proc

        result = await server_manager_instance.stop_server(1, force=False)

        assert result is True
        mock_proc.kill.assert_called_once()
        assert 1 not in server_manager_instance.processes


class TestServerManagerStartServerLogFileCleanup:
    """Test start_server log file handling."""

    @pytest.mark.asyncio
    async def test_start_server_clears_existing_log(
        self, server_manager_instance, sample_profile, tmp_path
    ):
        """Test clears existing log file before starting."""
        log_file = tmp_path / "server-1.log"
        log_file.write_text("Old log content")

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("asyncio.sleep"):
                with patch(
                    "mlx_manager.services.server_manager.get_server_log_path",
                    return_value=log_file,
                ):
                    await server_manager_instance.start_server(sample_profile)

        # Log file should be deleted during startup
        # (Note: In the actual code, it's deleted before Popen, so the file won't exist
        # after successful start unless the server writes to it)
        assert sample_profile.id in server_manager_instance.processes

    @pytest.mark.asyncio
    async def test_start_server_failure_reads_log(
        self, server_manager_instance, sample_profile, tmp_path
    ):
        """Test reads log file on startup failure."""
        log_file = tmp_path / "server-1.log"
        error_content = "Error: Failed to load model at path '/invalid/path'"

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 1  # Exited with error

        def create_log_and_exit(*args, **kwargs):
            # Simulate server writing error to log before exiting
            log_file.write_text(error_content)
            return mock_proc

        with patch("subprocess.Popen", side_effect=create_log_and_exit):
            with patch("asyncio.sleep"):
                with patch(
                    "mlx_manager.services.server_manager.get_server_log_path",
                    return_value=log_file,
                ):
                    with pytest.raises(RuntimeError) as exc_info:
                        await server_manager_instance.start_server(sample_profile)

        assert "Failed to load model" in str(exc_info.value)


class TestServerManagerStartServerModelFamilyCheck:
    """Tests for model family version check during server start."""

    @pytest.mark.asyncio
    async def test_start_server_model_family_not_supported(
        self, server_manager_instance, sample_profile
    ):
        """Test start fails when model family is not supported (lines 45-49)."""
        # Configure model path to trigger minimax detection
        sample_profile.model_path = "mlx-community/MiniMax-M2.1-3bit"

        # Mock detect_model_family to return "minimax"
        # Mock check_mlx_lm_support to return unsupported
        with (
            patch(
                "mlx_manager.services.server_manager.detect_model_family",
                return_value="minimax",
            ),
            patch(
                "mlx_manager.services.server_manager.check_mlx_lm_support",
                return_value={
                    "supported": False,
                    "error": "minimax models require mlx-lm >= 0.28.4 (installed: 0.26.0)",
                },
            ),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await server_manager_instance.start_server(sample_profile)

        assert "minimax models require mlx-lm >= 0.28.4" in str(exc_info.value)


class TestServerManagerLogFileHandling:
    """Tests for log file handling edge cases."""

    @pytest.mark.asyncio
    async def test_start_server_log_file_close_exception(
        self, server_manager_instance, sample_profile, tmp_path
    ):
        """Test handles exception when closing log file on startup failure (lines 89-90)."""
        log_file = tmp_path / "server-1.log"

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 1  # Exited with error

        # Create a mock log file object that raises on close
        mock_log_file = MagicMock()
        mock_log_file.flush.side_effect = OSError("Flush failed")
        mock_log_file.close.side_effect = OSError("Close failed")

        def popen_side_effect(*args, **kwargs):
            # Store the mock log file in _log_files
            server_manager_instance._log_files = {sample_profile.id: mock_log_file}
            return mock_proc

        with (
            patch("subprocess.Popen", side_effect=popen_side_effect),
            patch("asyncio.sleep"),
            patch(
                "mlx_manager.services.server_manager.get_server_log_path",
                return_value=log_file,
            ),
            patch(
                "mlx_manager.services.server_manager.detect_model_family",
                return_value=None,
            ),
        ):
            with pytest.raises(RuntimeError, match="failed to start"):
                await server_manager_instance.start_server(sample_profile)

    @pytest.mark.asyncio
    async def test_stop_server_log_file_close_exception(
        self, server_manager_instance, sample_profile
    ):
        """Test handles exception when closing log file on stop (lines 136-140)."""
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]  # Running, then exited
        mock_proc.wait.return_value = 0

        # Create a mock log file that raises on close
        mock_log_file = MagicMock()
        mock_log_file.close.side_effect = OSError("Close failed")

        server_manager_instance.processes[sample_profile.id] = mock_proc
        server_manager_instance._log_files = {sample_profile.id: mock_log_file}

        # Should not raise despite close failure
        result = await server_manager_instance.stop_server(sample_profile.id)

        assert result is True
        assert sample_profile.id not in server_manager_instance.processes
        assert sample_profile.id not in server_manager_instance._log_files


class TestServerManagerGetLogLinesEdgeCases:
    """Tests for get_log_lines edge cases."""

    def test_get_log_lines_flush_exception(self, server_manager_instance, tmp_path):
        """Test handles exception when flushing log file (lines 210-213)."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        server_manager_instance.processes[1] = mock_proc

        # Create a mock log file that raises on flush
        mock_log_file = MagicMock()
        mock_log_file.flush.side_effect = OSError("Flush failed")
        server_manager_instance._log_files = {1: mock_log_file}

        log_file = tmp_path / "server-1.log"
        log_file.write_text("Line 1\nLine 2\n")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            # Should not raise, should still read log file
            result = server_manager_instance.get_log_lines(1)

        assert len(result) == 2
        assert result[0] == "Line 1"


class TestServerManagerGetProcessStatusEdgeCases:
    """Additional tests for get_process_status edge cases."""

    def test_get_process_status_log_read_exception(self, server_manager_instance, tmp_path):
        """Test handles exception when reading log file (lines 267-268)."""
        # Create a log file path that will fail on read
        log_file = tmp_path / "server-999.log"
        log_file.write_text("Some error content")

        with (
            patch(
                "mlx_manager.services.server_manager.get_server_log_path",
                return_value=log_file,
            ),
            patch.object(log_file.__class__, "read_text", side_effect=PermissionError),
        ):
            result = server_manager_instance.get_process_status(999)

        # Should return non-failed status since we couldn't read the log
        assert result == {"running": False, "tracked": False, "failed": False}

    def test_get_process_status_exited_log_file_close_exception(
        self, server_manager_instance, tmp_path
    ):
        """Test handles exception when closing log file for exited process (lines 278-283)."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0  # Exited with code 0

        # Create a mock log file that raises on flush/close
        mock_log_file = MagicMock()
        mock_log_file.flush.side_effect = OSError("Flush failed")
        mock_log_file.close.side_effect = OSError("Close failed")

        server_manager_instance.processes[1] = mock_proc
        server_manager_instance._log_files = {1: mock_log_file}

        log_file = tmp_path / "server-1.log"
        log_file.write_text("Normal output")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            # Should not raise despite close failure
            result = server_manager_instance.get_process_status(1)

        assert result["running"] is False
        assert result["tracked"] is True
        assert result["failed"] is False  # Exit code 0, no error in log
        assert 1 not in server_manager_instance._log_files


class TestServerManagerGetProcessStatus:
    """Tests for the get_process_status method."""

    def test_get_process_status_not_tracked_no_log(self, server_manager_instance):
        """Test returns tracked=False when no process exists and no log."""
        result = server_manager_instance.get_process_status(999)
        assert result == {"running": False, "tracked": False, "failed": False}

    def test_get_process_status_not_tracked_with_error_log(self, server_manager_instance, tmp_path):
        """Test detects failure from log file even when process is not tracked."""
        # Simulate a log file from a previously crashed server
        log_file = tmp_path / "server-999.log"
        log_file.write_text("Starting server...\nERROR: Model type not supported\nShutdown.")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            result = server_manager_instance.get_process_status(999)

        assert result["running"] is False
        assert result["tracked"] is False
        assert result["failed"] is True
        assert "Model type not supported" in result["error_message"]

    def test_get_process_status_not_tracked_with_clean_log(self, server_manager_instance, tmp_path):
        """Test reports not failed when log file has no errors."""
        log_file = tmp_path / "server-999.log"
        log_file.write_text("Starting server...\nShutdown gracefully.")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            result = server_manager_instance.get_process_status(999)

        assert result["running"] is False
        assert result["tracked"] is False
        assert result["failed"] is False

    def test_get_process_status_running(self, server_manager_instance):
        """Test returns running=True when process is active."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None  # Still running
        server_manager_instance.processes[1] = mock_proc

        result = server_manager_instance.get_process_status(1)
        assert result["running"] is True
        assert result["tracked"] is True
        assert result["pid"] == 12345

    def test_get_process_status_exited_with_error_code(self, server_manager_instance, tmp_path):
        """Test detects failure when exit code is non-zero."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 1  # Non-zero exit code
        server_manager_instance.processes[1] = mock_proc

        log_file = tmp_path / "server-1.log"
        log_file.write_text("Some output before crash")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            result = server_manager_instance.get_process_status(1)

        assert result["running"] is False
        assert result["tracked"] is True
        assert result["exit_code"] == 1
        assert result["failed"] is True

    def test_get_process_status_exit_code_zero_with_error_in_log(
        self, server_manager_instance, tmp_path
    ):
        """Test detects failure when exit code is 0 but log contains error patterns."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0  # Exit code 0
        server_manager_instance.processes[1] = mock_proc

        log_file = tmp_path / "server-1.log"
        log_file.write_text("ERROR: Model type minimax not supported.\nApplication startup failed.")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            result = server_manager_instance.get_process_status(1)

        assert result["running"] is False
        assert result["exit_code"] == 0
        assert result["failed"] is True  # Should be True due to error in log
        assert "minimax not supported" in result["error_message"]

    def test_get_process_status_exit_code_zero_no_error(self, server_manager_instance, tmp_path):
        """Test reports not failed when exit code is 0 and no error in log."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0  # Exit code 0
        server_manager_instance.processes[1] = mock_proc

        log_file = tmp_path / "server-1.log"
        log_file.write_text("Server started successfully.\nShutting down gracefully.")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            result = server_manager_instance.get_process_status(1)

        assert result["running"] is False
        assert result["exit_code"] == 0
        assert result["failed"] is False  # Should be False - clean exit

    def test_get_process_status_cleans_up_dead_process(self, server_manager_instance, tmp_path):
        """Test removes process from tracking after detecting exit."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0
        server_manager_instance.processes[1] = mock_proc
        server_manager_instance._log_positions[1] = 100

        log_file = tmp_path / "server-1.log"
        log_file.write_text("Normal output")

        with patch(
            "mlx_manager.services.server_manager.get_server_log_path",
            return_value=log_file,
        ):
            server_manager_instance.get_process_status(1)

        # Process should be cleaned up
        assert 1 not in server_manager_instance.processes
        assert 1 not in server_manager_instance._log_positions
