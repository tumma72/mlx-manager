"""Tests for the system API router."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_memory(auth_client):
    """Test getting system memory information."""
    gib = 1024**3  # Binary gibibyte (1,073,741,824 bytes)

    with (
        patch("mlx_manager.routers.system.get_physical_memory_bytes") as mock_physical,
        patch("mlx_manager.routers.system.psutil") as mock_psutil,
    ):
        # Mock physical memory (from sysctl on macOS)
        mock_physical.return_value = 128 * gib  # 128 GiB

        # Mock psutil for available memory
        mock_mem = MagicMock()
        mock_mem.available = 64 * gib  # 64 GiB
        mock_psutil.virtual_memory.return_value = mock_mem

        response = await auth_client.get("/api/system/memory")
        assert response.status_code == 200

        data = response.json()
        assert data["total_gb"] == 128.0
        assert data["available_gb"] == 64.0
        assert data["used_gb"] == 64.0
        assert data["percent_used"] == 50.0
        # Default max_memory_percent is 80
        assert data["mlx_recommended_gb"] == 102.4


@pytest.mark.asyncio
async def test_get_system_info(auth_client):
    """Test getting system information."""
    response = await auth_client.get("/api/system/info")
    assert response.status_code == 200

    data = response.json()
    # Check that required fields are present
    assert "os_version" in data
    assert "chip" in data
    assert "memory_gb" in data
    assert "python_version" in data
    # Memory should be a positive number
    assert data["memory_gb"] > 0


@pytest.mark.asyncio
async def test_install_launchd_service(auth_client, sample_profile_data, mock_launchd_manager):
    """Test installing a launchd service."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Install launchd service
    response = await auth_client.post(f"/api/system/launchd/install/{profile_id}")
    assert response.status_code == 200

    data = response.json()
    assert "plist_path" in data
    assert data["label"] == "com.mlx-manager.test"


@pytest.mark.asyncio
async def test_install_launchd_service_profile_not_found(auth_client, mock_launchd_manager):
    """Test installing launchd service for non-existent profile."""
    response = await auth_client.post("/api/system/launchd/install/999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_uninstall_launchd_service(auth_client, sample_profile_data, mock_launchd_manager):
    """Test uninstalling a launchd service."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # First install
    await auth_client.post(f"/api/system/launchd/install/{profile_id}")

    # Then uninstall
    response = await auth_client.post(f"/api/system/launchd/uninstall/{profile_id}")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_uninstall_launchd_service_profile_not_found(auth_client, mock_launchd_manager):
    """Test uninstalling launchd service for non-existent profile."""
    response = await auth_client.post("/api/system/launchd/uninstall/999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_launchd_status(auth_client, sample_profile_data, mock_launchd_manager):
    """Test getting launchd service status."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Get status
    response = await auth_client.get(f"/api/system/launchd/status/{profile_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["installed"] is False
    assert data["running"] is False
    assert data["label"] == "com.mlx-manager.test"


@pytest.mark.asyncio
async def test_get_launchd_status_profile_not_found(auth_client, mock_launchd_manager):
    """Test getting launchd status for non-existent profile."""
    response = await auth_client.get("/api/system/launchd/status/999")
    assert response.status_code == 404


# ============================================================================
# Tests for get_physical_memory_bytes exception handling (lines 38-41)
# ============================================================================


class TestGetPhysicalMemoryBytes:
    """Tests for the get_physical_memory_bytes function."""

    def test_fallback_to_psutil_on_sysctl_exception(self):
        """Test falls back to psutil when sysctl raises exception."""
        from mlx_manager.routers.system import get_physical_memory_bytes

        gib = 1024**3

        with (
            patch("mlx_manager.routers.system.platform") as mock_platform,
            patch("mlx_manager.routers.system.subprocess.run") as mock_run,
            patch("mlx_manager.routers.system.psutil") as mock_psutil,
        ):
            mock_platform.system.return_value = "Darwin"
            mock_run.side_effect = OSError("sysctl failed")

            mock_mem = MagicMock()
            mock_mem.total = 64 * gib
            mock_psutil.virtual_memory.return_value = mock_mem

            result = get_physical_memory_bytes()

        assert result == 64 * gib

    def test_fallback_to_psutil_on_non_darwin(self):
        """Test uses psutil on non-Darwin platforms."""
        from mlx_manager.routers.system import get_physical_memory_bytes

        gib = 1024**3

        with (
            patch("mlx_manager.routers.system.platform") as mock_platform,
            patch("mlx_manager.routers.system.psutil") as mock_psutil,
        ):
            mock_platform.system.return_value = "Linux"

            mock_mem = MagicMock()
            mock_mem.total = 32 * gib
            mock_psutil.virtual_memory.return_value = mock_mem

            result = get_physical_memory_bytes()

        assert result == 32 * gib


# ============================================================================
# Tests for get_system_info exception handling (lines 86-87, 101-102, 109)
# ============================================================================


class TestGetSystemInfoExceptions:
    """Tests for exception handling in get_system_info endpoint."""

    @pytest.mark.asyncio
    async def test_chip_info_exception_returns_unknown(self, auth_client):
        """Test chip info returns Unknown when sysctl fails."""
        with (
            patch("mlx_manager.routers.system.subprocess.run") as mock_run,
            patch("mlx_manager.routers.system.get_physical_memory_bytes") as mock_mem,
        ):
            mock_run.side_effect = OSError("sysctl not available")
            mock_mem.return_value = 64 * (1024**3)

            response = await auth_client.get("/api/system/info")

        assert response.status_code == 200
        data = response.json()
        assert data["chip"] == "Unknown"

    @pytest.mark.asyncio
    async def test_mlx_version_import_error_returns_none(self, auth_client):
        """Test mlx_version is None when mlx import fails."""
        with (
            patch("mlx_manager.routers.system.get_physical_memory_bytes") as mock_mem,
            patch.dict("sys.modules", {"mlx": None}),
        ):
            mock_mem.return_value = 64 * (1024**3)

            # Mock the import to fail
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if name == "mlx":
                    raise ImportError("No module named 'mlx'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                response = await auth_client.get("/api/system/info")

        # Even if mlx import fails, endpoint should succeed
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_mlx_openai_server_version_import_error(self, auth_client):
        """Test mlx_openai_server_version is None when import fails."""
        response = await auth_client.get("/api/system/info")
        assert response.status_code == 200
        # mlx_openai_server not installed in test env, should be None
        data = response.json()
        # The field should be present (possibly None)
        assert "mlx_openai_server_version" in data


# ============================================================================
# Tests for get_parser_options endpoint (line 132)
# ============================================================================


@pytest.mark.asyncio
async def test_get_parser_options_endpoint(auth_client):
    """Test the parser options endpoint returns valid data."""
    response = await auth_client.get("/api/system/parser-options")
    assert response.status_code == 200

    data = response.json()
    assert "tool_call_parsers" in data
    assert "reasoning_parsers" in data
    assert "message_converters" in data
    assert isinstance(data["tool_call_parsers"], list)
    assert len(data["tool_call_parsers"]) > 0


# ============================================================================
# Tests for install_launchd_service exception handling (lines 149-151)
# ============================================================================


@pytest.mark.asyncio
async def test_install_launchd_service_exception(auth_client, sample_profile_data):
    """Test launchd install returns 500 when exception occurs."""
    with patch("mlx_manager.routers.system.launchd_manager") as mock_launchd:
        mock_launchd.install.side_effect = Exception("Failed to write plist")
        mock_launchd.get_label.return_value = "com.mlx-manager.test"

        # Create a profile
        create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # Try to install (should fail)
        response = await auth_client.post(f"/api/system/launchd/install/{profile_id}")

    assert response.status_code == 500
    assert "Failed to write plist" in response.json()["detail"]


# ============================================================================
# Tests for mlx_openai_server version extraction (line 109)
# ============================================================================


@pytest.mark.asyncio
async def test_get_system_info_with_mlx_openai_server_installed(auth_client):
    """Test system info returns mlx_openai_server version when installed."""
    import sys
    from types import ModuleType

    # Create a mock module with __version__
    mock_module = ModuleType("mlx_openai_server")
    mock_module.__version__ = "1.5.0"

    with (
        patch("mlx_manager.routers.system.get_physical_memory_bytes") as mock_mem,
        patch.dict(sys.modules, {"mlx_openai_server": mock_module}),
    ):
        mock_mem.return_value = 64 * (1024**3)

        response = await auth_client.get("/api/system/info")

    assert response.status_code == 200
    data = response.json()
    assert data["mlx_openai_server_version"] == "1.5.0"


@pytest.mark.asyncio
async def test_get_system_info_mlx_openai_server_without_version(auth_client):
    """Test system info uses 'installed' when module lacks __version__."""
    import sys
    from types import ModuleType

    # Create a mock module without __version__
    mock_module = ModuleType("mlx_openai_server")
    # Don't set __version__

    with (
        patch("mlx_manager.routers.system.get_physical_memory_bytes") as mock_mem,
        patch.dict(sys.modules, {"mlx_openai_server": mock_module}),
    ):
        mock_mem.return_value = 64 * (1024**3)

        response = await auth_client.get("/api/system/info")

    assert response.status_code == 200
    data = response.json()
    # Falls back to "installed" when __version__ not present
    assert data["mlx_openai_server_version"] == "installed"
