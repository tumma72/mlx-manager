"""Tests for the server manager service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.models import ServerProfile
from mlx_manager.services.server_manager import ServerManager


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


class TestServerManagerBuildCommand:
    """Tests for the _build_command method."""

    def test_basic_command(self, server_manager_instance, sample_profile):
        """Test building basic command."""
        cmd = server_manager_instance._build_command(sample_profile)

        assert "--model-path" in cmd
        assert "mlx-community/test-model" in cmd
        assert "--model-type" in cmd
        assert "lm" in cmd
        assert "--port" in cmd
        assert "10240" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd

    def test_command_with_context_length(self, server_manager_instance, sample_profile):
        """Test command includes context length when set."""
        sample_profile.context_length = 8192
        cmd = server_manager_instance._build_command(sample_profile)

        assert "--context-length" in cmd
        assert "8192" in cmd

    def test_command_with_tool_call_parser(self, server_manager_instance, sample_profile):
        """Test command includes tool call parser when set."""
        sample_profile.tool_call_parser = "qwen3"
        cmd = server_manager_instance._build_command(sample_profile)

        assert "--tool-call-parser" in cmd
        assert "qwen3" in cmd

    def test_command_with_reasoning_parser(self, server_manager_instance, sample_profile):
        """Test command includes reasoning parser when set."""
        sample_profile.reasoning_parser = "deepseek_r1"
        cmd = server_manager_instance._build_command(sample_profile)

        assert "--reasoning-parser" in cmd
        assert "deepseek_r1" in cmd

    def test_command_with_auto_tool_choice(self, server_manager_instance, sample_profile):
        """Test command includes auto tool choice flag when enabled."""
        sample_profile.enable_auto_tool_choice = True
        cmd = server_manager_instance._build_command(sample_profile)

        assert "--enable-auto-tool-choice" in cmd

    def test_command_with_trust_remote_code(self, server_manager_instance, sample_profile):
        """Test command includes trust remote code flag when enabled."""
        sample_profile.trust_remote_code = True
        cmd = server_manager_instance._build_command(sample_profile)

        assert "--trust-remote-code" in cmd

    def test_command_with_chat_template(self, server_manager_instance, sample_profile):
        """Test command includes chat template file when set."""
        sample_profile.chat_template_file = "/path/to/template.jinja"
        cmd = server_manager_instance._build_command(sample_profile)

        assert "--chat-template-file" in cmd
        assert "/path/to/template.jinja" in cmd

    def test_command_with_no_log_file(self, server_manager_instance, sample_profile):
        """Test command includes no-log-file flag when set."""
        sample_profile.no_log_file = True
        cmd = server_manager_instance._build_command(sample_profile)

        assert "--no-log-file" in cmd

    def test_command_with_log_file(self, server_manager_instance, sample_profile):
        """Test command includes log file path when set."""
        sample_profile.log_file = "/var/log/mlx.log"
        cmd = server_manager_instance._build_command(sample_profile)

        assert "--log-file" in cmd
        assert "/var/log/mlx.log" in cmd


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
    async def test_start_already_running_server(
        self, server_manager_instance, sample_profile
    ):
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
    async def test_check_health_unhealthy_http_error(
        self, server_manager_instance, sample_profile
    ):
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
    async def test_check_health_connection_error(
        self, server_manager_instance, sample_profile
    ):
        """Test health check returns unhealthy on connection error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection refused")
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
