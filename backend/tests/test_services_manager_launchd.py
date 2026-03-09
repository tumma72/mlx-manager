"""Tests for the manager_launchd module."""

import plistlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_manager.services.manager_launchd import (
    LABEL,
    _find_executable,
    get_plist_path,
    install_manager_service,
    is_service_installed,
    is_service_running,
    uninstall_manager_service,
)


class TestGetPlistPath:
    """Tests for get_plist_path."""

    def test_returns_correct_path(self):
        """Test plist path is in ~/Library/LaunchAgents."""
        path = get_plist_path()
        expected = Path.home() / "Library" / "LaunchAgents" / "com.mlx-manager.app.plist"
        assert path == expected


class TestFindExecutable:
    """Tests for _find_executable."""

    def test_prefers_shutil_which(self):
        """Test uses shutil.which when available."""
        with patch("shutil.which", return_value="/usr/local/bin/mlx-manager"):
            result = _find_executable()
            assert result == "/usr/local/bin/mlx-manager"

    def test_falls_back_to_sys_executable(self, tmp_path):
        """Test falls back to sys.executable parent when which returns None."""
        fake_exe = tmp_path / "mlx-manager"
        fake_exe.touch()

        with (
            patch("shutil.which", return_value=None),
            patch("sys.executable", str(tmp_path / "python")),
        ):
            result = _find_executable()
            assert result == str(fake_exe)

    def test_returns_empty_when_nothing_found(self, tmp_path):
        """Test returns empty string when executable not found."""
        with (
            patch("shutil.which", return_value=None),
            patch("sys.executable", str(tmp_path / "nonexistent" / "python")),
        ):
            result = _find_executable()
            assert result == ""


class TestInstallManagerService:
    """Tests for install_manager_service."""

    def test_install_creates_plist_with_correct_structure(self, tmp_path):
        """Test install creates a valid plist file."""
        plist_path = tmp_path / "com.mlx-manager.app.plist"

        with (
            patch(
                "mlx_manager.services.manager_launchd.get_plist_path",
                return_value=plist_path,
            ),
            patch(
                "mlx_manager.services.manager_launchd._find_executable",
                return_value="/usr/local/bin/mlx-manager",
            ),
            patch("mlx_manager.services.manager_launchd.bootstrap") as mock_bootstrap,
        ):
            result = install_manager_service()

        assert plist_path.exists()
        assert result == str(plist_path)

        # Verify plist content
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)

        assert plist["Label"] == LABEL
        assert plist["ProgramArguments"][0] == "/usr/local/bin/mlx-manager"
        assert "serve" in plist["ProgramArguments"]
        assert plist["RunAtLoad"] is True
        assert plist["EnvironmentVariables"]["PATH"].startswith("/usr/local/bin")

        mock_bootstrap.assert_called_once_with(str(plist_path), LABEL)

    def test_install_uses_which_for_executable(self, tmp_path):
        """Test install uses shutil.which for stable path."""
        plist_path = tmp_path / "com.mlx-manager.app.plist"

        with (
            patch(
                "mlx_manager.services.manager_launchd.get_plist_path",
                return_value=plist_path,
            ),
            patch(
                "mlx_manager.services.manager_launchd._find_executable",
                return_value="/opt/homebrew/bin/mlx-manager",
            ),
            patch("mlx_manager.services.manager_launchd.bootstrap"),
        ):
            install_manager_service()

        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)

        assert plist["ProgramArguments"][0] == "/opt/homebrew/bin/mlx-manager"

    def test_install_falls_back_to_python_module(self, tmp_path):
        """Test install uses python -m when executable not found."""
        plist_path = tmp_path / "com.mlx-manager.app.plist"

        with (
            patch(
                "mlx_manager.services.manager_launchd.get_plist_path",
                return_value=plist_path,
            ),
            patch(
                "mlx_manager.services.manager_launchd._find_executable",
                return_value="",
            ),
            patch("sys.executable", "/usr/bin/python3"),
            patch("mlx_manager.services.manager_launchd.bootstrap"),
        ):
            install_manager_service()

        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)

        assert plist["ProgramArguments"][0] == "/usr/bin/python3"
        assert plist["ProgramArguments"][1] == "-m"
        assert plist["ProgramArguments"][2] == "mlx_manager.cli"

    def test_install_calls_bootstrap(self, tmp_path):
        """Test install uses modern launchctl bootstrap API."""
        plist_path = tmp_path / "com.mlx-manager.app.plist"

        with (
            patch(
                "mlx_manager.services.manager_launchd.get_plist_path",
                return_value=plist_path,
            ),
            patch(
                "mlx_manager.services.manager_launchd._find_executable",
                return_value="/usr/local/bin/mlx-manager",
            ),
            patch("mlx_manager.services.manager_launchd.bootstrap") as mock_bootstrap,
        ):
            install_manager_service()

        mock_bootstrap.assert_called_once_with(str(plist_path), LABEL)


class TestUninstallManagerService:
    """Tests for uninstall_manager_service."""

    def test_uninstall_calls_bootout_and_removes_plist(self, tmp_path):
        """Test uninstall bootouts and removes plist."""
        plist_path = tmp_path / "com.mlx-manager.app.plist"
        plist_path.write_text("test")

        with (
            patch(
                "mlx_manager.services.manager_launchd.get_plist_path",
                return_value=plist_path,
            ),
            patch("mlx_manager.services.manager_launchd.bootout") as mock_bootout,
        ):
            result = uninstall_manager_service()

        assert result is True
        assert not plist_path.exists()
        mock_bootout.assert_called_once_with(LABEL)

    def test_uninstall_nonexistent_returns_false(self, tmp_path):
        """Test uninstall returns False when plist doesn't exist."""
        plist_path = tmp_path / "com.mlx-manager.app.plist"

        with patch(
            "mlx_manager.services.manager_launchd.get_plist_path",
            return_value=plist_path,
        ):
            result = uninstall_manager_service()

        assert result is False


class TestIsServiceInstalled:
    """Tests for is_service_installed."""

    def test_installed_true(self, tmp_path):
        """Test returns True when plist exists."""
        plist_path = tmp_path / "com.mlx-manager.app.plist"
        plist_path.write_text("test")

        with patch(
            "mlx_manager.services.manager_launchd.get_plist_path",
            return_value=plist_path,
        ):
            assert is_service_installed() is True

    def test_installed_false(self, tmp_path):
        """Test returns False when plist doesn't exist."""
        plist_path = tmp_path / "com.mlx-manager.app.plist"

        with patch(
            "mlx_manager.services.manager_launchd.get_plist_path",
            return_value=plist_path,
        ):
            assert is_service_installed() is False


class TestIsServiceRunning:
    """Tests for is_service_running."""

    def test_running_true(self):
        """Test returns True when launchctl list succeeds."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert is_service_running() is True

    def test_running_false(self):
        """Test returns False when launchctl list fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert is_service_running() is False
