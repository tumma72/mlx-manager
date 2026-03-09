"""Tests for the launchctl_utils module."""

import subprocess
from unittest.mock import MagicMock, patch

from mlx_manager.services.launchctl_utils import (
    bootout,
    bootstrap,
    get_gui_domain,
    get_service_target,
    kickstart,
    kill_service,
)


class TestGetGuiDomain:
    """Tests for get_gui_domain."""

    def test_returns_gui_uid_format(self):
        """Test returns gui/<uid> format."""
        with patch("os.getuid", return_value=501):
            assert get_gui_domain() == "gui/501"


class TestGetServiceTarget:
    """Tests for get_service_target."""

    def test_returns_domain_slash_label(self):
        """Test returns gui/<uid>/<label> format."""
        with patch("os.getuid", return_value=501):
            assert get_service_target("com.test.app") == "gui/501/com.test.app"


class TestBootstrap:
    """Tests for bootstrap."""

    def test_bootstrap_calls_launchctl(self):
        """Test bootstrap calls launchctl bootstrap with correct args."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            bootstrap("/path/to/plist", "com.test.app")

            # First call: bootout (idempotent)
            bootout_call = mock_run.call_args_list[0]
            assert bootout_call[0][0] == ["launchctl", "bootout", "gui/501/com.test.app"]

            # Second call: bootstrap
            bootstrap_call = mock_run.call_args_list[1]
            assert bootstrap_call[0][0] == [
                "launchctl",
                "bootstrap",
                "gui/501",
                "/path/to/plist",
            ]

    def test_bootstrap_bootouts_first(self):
        """Test bootstrap calls bootout before bootstrap for idempotency."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            bootstrap("/path/to/plist", "com.test.app")

            assert mock_run.call_count == 2
            first_cmd = mock_run.call_args_list[0][0][0]
            second_cmd = mock_run.call_args_list[1][0][0]
            assert "bootout" in first_cmd
            assert "bootstrap" in second_cmd

    def test_bootstrap_raises_on_failure(self):
        """Test bootstrap raises CalledProcessError on failure."""
        import pytest

        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            # bootout succeeds, bootstrap fails
            mock_run.side_effect = [
                MagicMock(returncode=0, stderr=""),  # bootout
                MagicMock(returncode=1, stdout="", stderr="bootstrap failed"),  # bootstrap
            ]

            with pytest.raises(subprocess.CalledProcessError):
                bootstrap("/path/to/plist", "com.test.app")


class TestBootout:
    """Tests for bootout."""

    def test_bootout_calls_launchctl(self):
        """Test bootout calls launchctl bootout with correct args."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            bootout("com.test.app")

            mock_run.assert_called_once_with(
                ["launchctl", "bootout", "gui/501/com.test.app"],
                capture_output=True,
                text=True,
            )

    def test_bootout_ignores_not_found(self):
        """Test bootout silently handles 'not found' errors."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=113, stderr="Could not find specified service"
            )

            # Should not raise
            bootout("com.test.app")

    def test_bootout_ignores_no_such_process(self):
        """Test bootout silently handles 'No such process' errors."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=3, stderr="3: No such process")

            # Should not raise
            bootout("com.test.app")


class TestKickstart:
    """Tests for kickstart."""

    def test_kickstart_calls_launchctl(self):
        """Test kickstart calls launchctl kickstart with correct args."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)

            result = kickstart("com.test.app")

            assert result is True
            mock_run.assert_called_once_with(
                ["launchctl", "kickstart", "gui/501/com.test.app"],
                capture_output=True,
                text=True,
            )

    def test_kickstart_returns_false_on_failure(self):
        """Test kickstart returns False when launchctl fails."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="failed")

            result = kickstart("com.test.app")

            assert result is False


class TestKillService:
    """Tests for kill_service."""

    def test_kill_service_calls_launchctl(self):
        """Test kill_service calls launchctl kill with SIGTERM."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)

            result = kill_service("com.test.app")

            assert result is True
            mock_run.assert_called_once_with(
                ["launchctl", "kill", "15", "gui/501/com.test.app"],
                capture_output=True,
                text=True,
            )

    def test_kill_service_custom_signal(self):
        """Test kill_service with custom signal."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)

            result = kill_service("com.test.app", signal=9)

            assert result is True
            args = mock_run.call_args[0][0]
            assert "9" in args

    def test_kill_service_returns_false_on_failure(self):
        """Test kill_service returns False when launchctl fails."""
        with (
            patch("os.getuid", return_value=501),
            patch("mlx_manager.services.launchctl_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="failed")

            result = kill_service("com.test.app")

            assert result is False
