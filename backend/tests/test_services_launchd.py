"""Tests for the launchd manager service."""

from unittest.mock import MagicMock, patch

import pytest

from mlx_manager.models import ServerProfile
from mlx_manager.services.launchd import LaunchdManager


@pytest.fixture
def launchd_manager(tmp_path):
    """Create a LaunchdManager instance with temp directory."""
    manager = LaunchdManager()
    manager.launch_agents_dir = tmp_path
    return manager


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
        auto_start=True,
    )


class TestLaunchdManagerGetLabel:
    """Tests for the get_label method."""

    def test_basic_label(self, launchd_manager, sample_profile):
        """Test basic label generation."""
        label = launchd_manager.get_label(sample_profile)
        assert label == "com.mlx-manager.test-profile"

    def test_label_with_spaces(self, launchd_manager, sample_profile):
        """Test label generation with spaces in name."""
        sample_profile.name = "My Test Profile"
        label = launchd_manager.get_label(sample_profile)
        assert label == "com.mlx-manager.my-test-profile"

    def test_label_with_underscores(self, launchd_manager, sample_profile):
        """Test label generation with underscores."""
        sample_profile.name = "my_test_profile"
        label = launchd_manager.get_label(sample_profile)
        assert label == "com.mlx-manager.my-test-profile"

    def test_label_with_special_chars(self, launchd_manager, sample_profile):
        """Test label generation removes special characters."""
        sample_profile.name = "Test@Profile#123!"
        label = launchd_manager.get_label(sample_profile)
        assert label == "com.mlx-manager.testprofile123"


class TestLaunchdManagerGetPlistPath:
    """Tests for the get_plist_path method."""

    def test_plist_path(self, launchd_manager, sample_profile, tmp_path):
        """Test plist path generation."""
        path = launchd_manager.get_plist_path(sample_profile)
        expected = tmp_path / "com.mlx-manager.test-profile.plist"
        assert path == expected


class TestLaunchdManagerGeneratePlist:
    """Tests for the generate_plist method.

    Note: With the embedded MLX Server, launchd now runs mlx-manager serve.
    """

    def test_basic_plist(self, launchd_manager, sample_profile):
        """Test basic plist generation with launch subcommand."""
        plist = launchd_manager.generate_plist(sample_profile)

        assert plist["Label"] == "com.mlx-manager.test-profile"
        assert plist["RunAtLoad"] is True  # auto_start is True
        assert "ProgramArguments" in plist
        assert "launch" in plist["ProgramArguments"]
        assert "--model-path" in plist["ProgramArguments"]
        assert "mlx-community/test-model" in plist["ProgramArguments"]

    def test_plist_program_arguments(self, launchd_manager, sample_profile):
        """Test plist program arguments contain required options."""
        plist = launchd_manager.generate_plist(sample_profile)
        args = plist["ProgramArguments"]

        assert "launch" in args
        assert "--model-type" in args
        assert "lm" in args
        assert "--port" in args
        assert "10240" in args
        assert "--host" in args
        assert "127.0.0.1" in args
        assert "--max-concurrency" in args
        assert "--queue-timeout" in args
        assert "--queue-size" in args

    def test_plist_maps_unsupported_model_types(self, launchd_manager, sample_profile):
        """Test that unsupported model types are mapped to 'lm'."""
        sample_profile.model_type = "whisper"
        plist = launchd_manager.generate_plist(sample_profile)
        args = plist["ProgramArguments"]

        model_type_idx = args.index("--model-type") + 1
        assert args[model_type_idx] == "lm"

    def test_plist_keepalive_settings(self, launchd_manager, sample_profile):
        """Test plist has correct KeepAlive settings."""
        plist = launchd_manager.generate_plist(sample_profile)

        assert "KeepAlive" in plist
        assert plist["KeepAlive"]["SuccessfulExit"] is False
        assert plist["KeepAlive"]["Crashed"] is True

    def test_plist_environment_variables(self, launchd_manager, sample_profile):
        """Test plist has required environment variables."""
        plist = launchd_manager.generate_plist(sample_profile)

        assert "EnvironmentVariables" in plist
        env = plist["EnvironmentVariables"]
        assert "PATH" in env
        assert "HOME" in env
        assert "PYTHONUNBUFFERED" in env


class TestLaunchdManagerIsInstalled:
    """Tests for the is_installed method."""

    def test_not_installed(self, launchd_manager, sample_profile):
        """Test returns False when plist doesn't exist."""
        result = launchd_manager.is_installed(sample_profile)
        assert result is False

    def test_installed(self, launchd_manager, sample_profile, tmp_path):
        """Test returns True when plist exists."""
        plist_path = tmp_path / "com.mlx-manager.test-profile.plist"
        plist_path.write_text("test")

        result = launchd_manager.is_installed(sample_profile)
        assert result is True


class TestLaunchdManagerInstall:
    """Tests for the install method."""

    def test_install_creates_plist(self, launchd_manager, sample_profile, tmp_path):
        """Test install creates plist file."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = launchd_manager.install(sample_profile)

        plist_path = tmp_path / "com.mlx-manager.test-profile.plist"
        assert plist_path.exists()
        assert result == str(plist_path)

    def test_install_loads_service(self, launchd_manager, sample_profile):
        """Test install calls launchctl load."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            launchd_manager.install(sample_profile)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "launchctl" in call_args
        assert "load" in call_args


class TestLaunchdManagerUninstall:
    """Tests for the uninstall method."""

    def test_uninstall_nonexistent(self, launchd_manager, sample_profile):
        """Test uninstall returns False when plist doesn't exist."""
        result = launchd_manager.uninstall(sample_profile)
        assert result is False

    def test_uninstall_removes_plist(self, launchd_manager, sample_profile, tmp_path):
        """Test uninstall removes plist file."""
        plist_path = tmp_path / "com.mlx-manager.test-profile.plist"
        plist_path.write_text("test")

        with patch("subprocess.run"):
            result = launchd_manager.uninstall(sample_profile)

        assert result is True
        assert not plist_path.exists()

    def test_uninstall_calls_launchctl(self, launchd_manager, sample_profile, tmp_path):
        """Test uninstall calls launchctl unload."""
        plist_path = tmp_path / "com.mlx-manager.test-profile.plist"
        plist_path.write_text("test")

        with patch("subprocess.run") as mock_run:
            launchd_manager.uninstall(sample_profile)

        call_args = mock_run.call_args[0][0]
        assert "launchctl" in call_args
        assert "unload" in call_args


class TestLaunchdManagerIsRunning:
    """Tests for the is_running method."""

    def test_is_running_true(self, launchd_manager, sample_profile):
        """Test returns True when service is running."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = launchd_manager.is_running(sample_profile)

        assert result is True

    def test_is_running_false(self, launchd_manager, sample_profile):
        """Test returns False when service is not running."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            result = launchd_manager.is_running(sample_profile)

        assert result is False


class TestLaunchdManagerStartStop:
    """Tests for the start and stop methods."""

    def test_start_service(self, launchd_manager, sample_profile):
        """Test starting a service."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = launchd_manager.start(sample_profile)

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "launchctl" in call_args
        assert "start" in call_args

    def test_stop_service(self, launchd_manager, sample_profile):
        """Test stopping a service."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = launchd_manager.stop(sample_profile)

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "launchctl" in call_args
        assert "stop" in call_args


class TestLaunchdManagerGetStatus:
    """Tests for the get_status method."""

    def test_get_status_not_running(self, launchd_manager, sample_profile):
        """Test status when service is not running."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")

            result = launchd_manager.get_status(sample_profile)

        assert result["running"] is False
        assert result["label"] == "com.mlx-manager.test-profile"

    def test_get_status_running(self, launchd_manager, sample_profile):
        """Test status when service is running."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="12345\t0\tcom.mlx-manager.test-profile"
            )

            result = launchd_manager.get_status(sample_profile)

        assert result["installed"] is True
        assert result["running"] is True
        assert result["pid"] == 12345

    def test_get_status_installed_not_running(self, launchd_manager, sample_profile):
        """Test status when service is installed but not running."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="-\t0\tcom.mlx-manager.test-profile"
            )

            result = launchd_manager.get_status(sample_profile)

        assert result["installed"] is True
        assert result["running"] is False
        assert result["pid"] is None
