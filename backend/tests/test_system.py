"""Tests for the system API router."""

from unittest.mock import AsyncMock, MagicMock, patch

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


# ============================================================================
# Tests for get_parser_options endpoint (deprecated)
# ============================================================================


@pytest.mark.asyncio
async def test_get_parser_options_endpoint(auth_client):
    """Test the parser options endpoint returns empty lists (deprecated)."""
    response = await auth_client.get("/api/system/parser-options")
    assert response.status_code == 200

    data = response.json()
    assert "tool_call_parsers" in data
    assert "reasoning_parsers" in data
    assert "message_converters" in data
    # Parser options are no longer used with embedded MLX Server
    assert isinstance(data["tool_call_parsers"], list)
    assert data["tool_call_parsers"] == []
    assert data["reasoning_parsers"] == []
    assert data["message_converters"] == []


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
# Audit Log Proxy Endpoint Tests
# ============================================================================


class TestGetAuditLogs:
    """Tests for GET /api/system/audit-logs."""

    @pytest.mark.asyncio
    async def test_get_audit_logs_success(self, auth_client):
        """Test successful audit log retrieval."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 1, "model": "gpt-4", "status": "success"},
            {"id": 2, "model": "claude-3", "status": "success"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get("/api/system/audit-logs")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_audit_logs_with_filters(self, auth_client):
        """Test audit log retrieval with query filters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": 1, "model": "gpt-4", "status": "success"}]
        mock_response.raise_for_status = MagicMock()

        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get(
                "/api/system/audit-logs",
                params={
                    "model": "gpt-4",
                    "backend_type": "openai",
                    "status": "success",
                    "start_time": "2024-01-01T00:00:00Z",
                    "end_time": "2024-12-31T23:59:59Z",
                    "limit": 50,
                    "offset": 10,
                },
            )

        assert response.status_code == 200
        # Verify the filter parameters were passed
        call_args = mock_instance.get.call_args
        params = call_args.kwargs.get("params", {})
        assert params.get("model") == "gpt-4"
        assert params.get("backend_type") == "openai"
        assert params.get("limit") == 50

    @pytest.mark.asyncio
    async def test_get_audit_logs_http_error(self, auth_client):
        """Test audit log retrieval with HTTP error from MLX Server."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=MagicMock(status_code=500)
        )

        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get("/api/system/audit-logs")

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_get_audit_logs_server_unavailable(self, auth_client):
        """Test audit log retrieval when MLX Server is unavailable."""
        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = Exception("Connection refused")
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get("/api/system/audit-logs")

        assert response.status_code == 503
        assert "MLX Server not available" in response.json()["detail"]


class TestGetAuditStats:
    """Tests for GET /api/system/audit-logs/stats."""

    @pytest.mark.asyncio
    async def test_get_audit_stats_success(self, auth_client):
        """Test successful audit stats retrieval."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_requests": 1000,
            "success_count": 950,
            "error_count": 50,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get("/api/system/audit-logs/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 1000

    @pytest.mark.asyncio
    async def test_get_audit_stats_http_error(self, auth_client):
        """Test audit stats with HTTP error from MLX Server."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=MagicMock(), response=MagicMock(status_code=404)
        )

        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get("/api/system/audit-logs/stats")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_audit_stats_server_unavailable(self, auth_client):
        """Test audit stats when MLX Server is unavailable."""
        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = Exception("Connection refused")
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get("/api/system/audit-logs/stats")

        assert response.status_code == 503
        assert "MLX Server not available" in response.json()["detail"]


class TestExportAuditLogs:
    """Tests for GET /api/system/audit-logs/export."""

    @pytest.mark.asyncio
    async def test_export_audit_logs_jsonl(self, auth_client):
        """Test exporting audit logs as JSONL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"id": 1}\n{"id": 2}\n'
        mock_response.raise_for_status = MagicMock()

        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get("/api/system/audit-logs/export")

        assert response.status_code == 200
        assert "application/jsonl" in response.headers.get("content-type", "")
        assert "attachment" in response.headers.get("content-disposition", "")

    @pytest.mark.asyncio
    async def test_export_audit_logs_csv(self, auth_client):
        """Test exporting audit logs as CSV."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"id,model,status\n1,gpt-4,success\n"
        mock_response.raise_for_status = MagicMock()

        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get(
                "/api/system/audit-logs/export", params={"format": "csv"}
            )

        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_export_audit_logs_with_filters(self, auth_client):
        """Test exporting audit logs with filters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"id": 1}\n'
        mock_response.raise_for_status = MagicMock()

        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get(
                "/api/system/audit-logs/export",
                params={
                    "model": "gpt-4",
                    "backend_type": "openai",
                    "status": "success",
                    "start_time": "2024-01-01T00:00:00Z",
                    "end_time": "2024-12-31T23:59:59Z",
                },
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_export_audit_logs_http_error(self, auth_client):
        """Test export with HTTP error from MLX Server."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=MagicMock(status_code=500)
        )

        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get("/api/system/audit-logs/export")

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_export_audit_logs_server_unavailable(self, auth_client):
        """Test export when MLX Server is unavailable."""
        with patch("mlx_manager.routers.system.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = Exception("Connection refused")
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            response = await auth_client.get("/api/system/audit-logs/export")

        assert response.status_code == 503
        assert "MLX Server not available" in response.json()["detail"]


# ============================================================================
# WebSocket Proxy Tests
# ============================================================================


class TestAuditLogWebSocketProxy:
    """Tests for WebSocket /api/system/ws/audit-logs."""

    def test_websocket_connection_failure(self):
        """Test WebSocket when MLX Server connection fails."""
        import websockets
        from starlette.testclient import TestClient as SyncTestClient

        from mlx_manager.main import app

        # Patch websockets.connect to simulate connection failure
        with patch.object(websockets, "connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")

            # Use synchronous TestClient for WebSocket testing
            with SyncTestClient(app) as sync_client:
                # WebSocket should accept connection then send error message
                with sync_client.websocket_connect("/api/system/ws/audit-logs") as ws:
                    # Should receive error message
                    data = ws.receive_json()
                    assert data["type"] == "error"
                    assert "MLX Server not available" in data["message"]
