"""Tests for the command builder utilities."""

import shutil
import sys
from unittest.mock import patch

import pytest

from mlx_manager.models import ServerProfile
from mlx_manager.utils.command_builder import (
    build_mlx_server_command,
    find_mlx_openai_server,
    get_server_log_path,
)


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


class TestFindMlxOpenaiServer:
    """Tests for find_mlx_openai_server function."""

    def test_finds_in_python_dir(self, tmp_path):
        """Test finds mlx-openai-server in same directory as Python executable."""
        mock_python_dir = tmp_path / "bin"
        mock_python_dir.mkdir()
        mock_server = mock_python_dir / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(mock_python_dir / "python")):
            result = find_mlx_openai_server()

        assert result == str(mock_server)

    def test_finds_in_system_path(self, tmp_path):
        """Test falls back to system PATH when not in Python dir."""
        mock_python_dir = tmp_path / "python_bin"
        mock_python_dir.mkdir()

        system_server = tmp_path / "system_bin" / "mlx-openai-server"
        system_server.parent.mkdir()
        system_server.touch()

        with patch.object(sys, "executable", str(mock_python_dir / "python")):
            with patch.object(shutil, "which", return_value=str(system_server)):
                result = find_mlx_openai_server()

        assert result == str(system_server)

    def test_raises_when_not_found(self, tmp_path):
        """Test raises RuntimeError when server not found anywhere."""
        mock_python_dir = tmp_path / "python_bin"
        mock_python_dir.mkdir()

        with patch.object(sys, "executable", str(mock_python_dir / "python")):
            with patch.object(shutil, "which", return_value=None):
                with pytest.raises(RuntimeError, match="mlx-openai-server not found"):
                    find_mlx_openai_server()


class TestGetServerLogPath:
    """Tests for get_server_log_path function."""

    def test_returns_log_path(self, tmp_path):
        """Test returns correct log path for profile ID."""
        with patch("mlx_manager.utils.command_builder.Path.home", return_value=tmp_path):
            result = get_server_log_path(1)

        expected = tmp_path / ".mlx-manager" / "logs" / "server-1.log"
        assert result == expected

    def test_creates_log_directory(self, tmp_path):
        """Test creates log directory if it doesn't exist."""
        with patch("mlx_manager.utils.command_builder.Path.home", return_value=tmp_path):
            result = get_server_log_path(42)

        # Directory should be created
        assert result.parent.exists()
        assert result.parent.is_dir()

    def test_different_profile_ids_get_different_paths(self, tmp_path):
        """Test different profile IDs get different log paths."""
        with patch("mlx_manager.utils.command_builder.Path.home", return_value=tmp_path):
            result1 = get_server_log_path(1)
            result2 = get_server_log_path(2)

        assert result1 != result2
        assert "server-1.log" in str(result1)
        assert "server-2.log" in str(result2)


class TestBuildMlxServerCommand:
    """Tests for build_mlx_server_command function."""

    def test_basic_command_structure(self, sample_profile, tmp_path):
        """Test command has correct basic structure."""
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        # Check structure
        assert cmd[0] == str(mock_server)
        assert cmd[1] == "launch"
        assert "--model-path" in cmd
        assert "--model-type" in cmd
        assert "--port" in cmd
        assert "--host" in cmd
        assert "--max-concurrency" in cmd
        assert "--queue-timeout" in cmd
        assert "--queue-size" in cmd

    def test_no_log_file_args_in_command(self, sample_profile, tmp_path):
        """Test that log file args are NOT in command (handled by server_manager)."""
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            cmd = build_mlx_server_command(sample_profile)

        # Log file options should NOT be in command
        # mlx-openai-server doesn't support them; server_manager redirects stdout/stderr
        assert "--log-file" not in cmd
        assert "--no-log-file" not in cmd

    def test_maps_whisper_to_lm(self, sample_profile, tmp_path):
        """Test maps unsupported 'whisper' type to 'lm'."""
        sample_profile.model_type = "whisper"
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        model_type_idx = cmd.index("--model-type")
        assert cmd[model_type_idx + 1] == "lm"

    def test_preserves_multimodal_type(self, sample_profile, tmp_path):
        """Test preserves 'multimodal' type."""
        sample_profile.model_type = "multimodal"
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        model_type_idx = cmd.index("--model-type")
        assert cmd[model_type_idx + 1] == "multimodal"

    def test_preserves_lm_type(self, sample_profile, tmp_path):
        """Test preserves 'lm' type."""
        sample_profile.model_type = "lm"
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        model_type_idx = cmd.index("--model-type")
        assert cmd[model_type_idx + 1] == "lm"

    def test_maps_custom_type_to_lm(self, sample_profile, tmp_path):
        """Test maps any custom/unknown type to 'lm'."""
        sample_profile.model_type = "custom_type"
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        model_type_idx = cmd.index("--model-type")
        assert cmd[model_type_idx + 1] == "lm"

    def test_all_numeric_values_converted_to_strings(self, sample_profile, tmp_path):
        """Test all numeric values are converted to strings."""
        sample_profile.port = 12345
        sample_profile.max_concurrency = 4
        sample_profile.queue_timeout = 600
        sample_profile.queue_size = 200
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        # All elements should be strings
        assert all(isinstance(item, str) for item in cmd)

        # Check specific values
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == "12345"

        concurrency_idx = cmd.index("--max-concurrency")
        assert cmd[concurrency_idx + 1] == "4"

        timeout_idx = cmd.index("--queue-timeout")
        assert cmd[timeout_idx + 1] == "600"

        size_idx = cmd.index("--queue-size")
        assert cmd[size_idx + 1] == "200"

    def test_includes_tool_call_parser(self, sample_profile, tmp_path):
        """Test includes --tool-call-parser when set."""
        sample_profile.tool_call_parser = "minimax_m2"
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        assert "--tool-call-parser" in cmd
        parser_idx = cmd.index("--tool-call-parser")
        assert cmd[parser_idx + 1] == "minimax_m2"

    def test_includes_reasoning_parser(self, sample_profile, tmp_path):
        """Test includes --reasoning-parser when set."""
        sample_profile.reasoning_parser = "qwen3"
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        assert "--reasoning-parser" in cmd
        parser_idx = cmd.index("--reasoning-parser")
        assert cmd[parser_idx + 1] == "qwen3"

    def test_includes_message_converter(self, sample_profile, tmp_path):
        """Test includes --message-converter when set."""
        sample_profile.message_converter = "glm4"
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        assert "--message-converter" in cmd
        converter_idx = cmd.index("--message-converter")
        assert cmd[converter_idx + 1] == "glm4"

    def test_includes_all_parser_options(self, sample_profile, tmp_path):
        """Test includes all parser options when set."""
        sample_profile.tool_call_parser = "minimax_m2"
        sample_profile.reasoning_parser = "minimax_m2"
        sample_profile.message_converter = "minimax_m2"
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        # All three parser options should be present
        assert "--tool-call-parser" in cmd
        assert "--reasoning-parser" in cmd
        assert "--message-converter" in cmd

        # Verify values
        assert cmd[cmd.index("--tool-call-parser") + 1] == "minimax_m2"
        assert cmd[cmd.index("--reasoning-parser") + 1] == "minimax_m2"
        assert cmd[cmd.index("--message-converter") + 1] == "minimax_m2"

    def test_excludes_parser_options_when_not_set(self, sample_profile, tmp_path):
        """Test excludes parser options when not set."""
        # Ensure parser options are None
        sample_profile.tool_call_parser = None
        sample_profile.reasoning_parser = None
        sample_profile.message_converter = None
        mock_server = tmp_path / "mlx-openai-server"
        mock_server.touch()

        with patch.object(sys, "executable", str(tmp_path / "python")):
            with patch(
                "mlx_manager.utils.command_builder.get_server_log_path",
                return_value=tmp_path / "server.log",
            ):
                cmd = build_mlx_server_command(sample_profile)

        # Parser options should not be present
        assert "--tool-call-parser" not in cmd
        assert "--reasoning-parser" not in cmd
        assert "--message-converter" not in cmd
