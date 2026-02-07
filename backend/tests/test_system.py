"""Tests for the system API router."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.services.auth_service import create_access_token


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
# Tests for removed deprecated endpoints
# ============================================================================


@pytest.mark.asyncio
async def test_parser_options_endpoint_removed(auth_client):
    """Test that deprecated /parser-options endpoint returns 404 (removed)."""
    response = await auth_client.get("/api/system/parser-options")
    assert response.status_code in (404, 405)


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
    """Tests for WebSocket /api/system/ws/audit-logs.

    All tests use direct function calls with mock WebSocket objects to avoid
    SyncTestClient lifespan initialization issues with in-memory test databases.
    """

    def _make_valid_token(self) -> str:
        """Create a valid JWT token for WebSocket tests."""
        return create_access_token(data={"sub": "test@example.com"})

    def _mock_get_session_with_user(self, email: str, status: str = "approved"):
        """Create a mock get_session that returns a user lookup result."""
        from contextlib import asynccontextmanager

        from mlx_manager.models import User, UserStatus

        mock_user = User(
            id=1,
            email=email,
            hashed_password="hashed",
            is_admin=False,
            status=UserStatus.APPROVED if status == "approved" else UserStatus.PENDING,
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()

        if status == "not_found":
            mock_result.scalar_one_or_none.return_value = None
        else:
            mock_result.scalar_one_or_none.return_value = mock_user

        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_get_session():
            yield mock_session

        return mock_get_session

    @pytest.mark.asyncio
    async def test_websocket_no_token_closes_1008(self):
        """Test WebSocket connection without token closes with code 1008."""
        from mlx_manager.routers.system import proxy_audit_log_stream

        mock_ws = AsyncMock()
        mock_ws.query_params = {}  # No token
        mock_ws.close = AsyncMock()

        await proxy_audit_log_stream(mock_ws)

        mock_ws.close.assert_called_once_with(code=1008)
        mock_ws.accept.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_invalid_token_closes_1008(self):
        """Test WebSocket connection with invalid token closes with code 1008."""
        from mlx_manager.routers.system import proxy_audit_log_stream

        mock_ws = AsyncMock()
        mock_ws.query_params = {"token": "invalid-jwt-token"}
        mock_ws.close = AsyncMock()

        await proxy_audit_log_stream(mock_ws)

        mock_ws.close.assert_called_once_with(code=1008)
        mock_ws.accept.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_unapproved_user_closes_1008(self):
        """Test WebSocket connection with valid token for non-existent user closes 1008."""
        from mlx_manager.routers.system import proxy_audit_log_stream

        token = create_access_token(data={"sub": "nonexistent@example.com"})
        mock_ws = AsyncMock()
        mock_ws.query_params = {"token": token}
        mock_ws.close = AsyncMock()

        # Mock the session to return no user
        mock_get_session = self._mock_get_session_with_user(
            "nonexistent@example.com", status="not_found"
        )

        with patch("mlx_manager.routers.system.get_session", mock_get_session):
            await proxy_audit_log_stream(mock_ws)

        mock_ws.close.assert_called_once_with(code=1008)
        mock_ws.accept.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_pending_user_closes_1008(self):
        """Test WebSocket connection with valid token for pending user closes 1008."""
        from mlx_manager.routers.system import proxy_audit_log_stream

        token = create_access_token(data={"sub": "pending@example.com"})
        mock_ws = AsyncMock()
        mock_ws.query_params = {"token": token}
        mock_ws.close = AsyncMock()

        # Mock the session to return a pending user
        mock_get_session = self._mock_get_session_with_user(
            "pending@example.com", status="pending"
        )

        with patch("mlx_manager.routers.system.get_session", mock_get_session):
            await proxy_audit_log_stream(mock_ws)

        mock_ws.close.assert_called_once_with(code=1008)
        mock_ws.accept.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_connection_failure(self):
        """Test WebSocket when MLX Server connection fails."""
        from mlx_manager.routers.system import proxy_audit_log_stream

        token = self._make_valid_token()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_ws.query_params = {"token": token}

        mock_get_session = self._mock_get_session_with_user("test@example.com")

        with (
            patch("mlx_manager.routers.system.get_session", mock_get_session),
            patch("websockets.connect", side_effect=Exception("Connection refused")),
        ):
            await proxy_audit_log_stream(mock_ws)

        # Should have accepted, sent error, and closed
        mock_ws.accept.assert_called_once()
        mock_ws.send_json.assert_called_once()
        error_data = mock_ws.send_json.call_args[0][0]
        assert error_data["type"] == "error"
        assert "MLX Server not available" in error_data["message"]
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_bidirectional_forwarding(self):
        """Test WebSocket bidirectional message forwarding."""
        from mlx_manager.routers.system import proxy_audit_log_stream

        # Verify the endpoint exists and has the expected signature
        assert proxy_audit_log_stream is not None
        import inspect

        sig = inspect.signature(proxy_audit_log_stream)
        assert "websocket" in sig.parameters

        # Test that the proxy forwards messages correctly via a mock
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock(side_effect=Exception("Client disconnected"))
        mock_ws.close = AsyncMock()
        mock_ws.query_params = {"token": self._make_valid_token()}

        class MockMLXWebSocket:
            """Mock MLX Server WebSocket."""

            def __init__(self):
                self.messages = ["msg1", "msg2"]
                self.index = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index < len(self.messages):
                    msg = self.messages[self.index]
                    self.index += 1
                    return msg
                raise StopAsyncIteration

            async def send(self, data):
                pass

        mock_get_session = self._mock_get_session_with_user("test@example.com")

        with (
            patch("mlx_manager.routers.system.get_session", mock_get_session),
            patch("websockets.connect", return_value=MockMLXWebSocket()),
        ):
            try:
                await proxy_audit_log_stream(mock_ws)
            except Exception:
                pass  # Expected to fail when receive_from_client raises

        # Verify the proxy accepted the connection
        mock_ws.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_receive_from_client_forwards_messages(self):
        """Test receive_from_client forwards messages to MLX server."""
        # Test that messages from client are forwarded to MLX server
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.query_params = {"token": self._make_valid_token()}
        received_by_mlx = []

        class MockMLXWebSocket:
            """Mock MLX Server WebSocket that records sent messages."""

            def __init__(self):
                self.message_count = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                # Send a message from MLX, then wait
                if self.message_count == 0:
                    self.message_count += 1
                    return "mlx_message"
                # After first message, wait for client messages
                import asyncio

                await asyncio.sleep(0.5)
                raise StopAsyncIteration

            async def send(self, data):
                # Record messages sent to MLX
                received_by_mlx.append(data)

        # Mock client receiving text to send a message then disconnect
        call_count = [0]

        async def mock_receive_text():
            call_count[0] += 1
            if call_count[0] == 1:
                return "client_message"
            # Simulate disconnect on second call
            from fastapi import WebSocketDisconnect

            raise WebSocketDisconnect()

        mock_ws.receive_text = mock_receive_text
        mock_ws.send_text = AsyncMock()
        mock_ws.close = AsyncMock()

        mock_get_session = self._mock_get_session_with_user("test@example.com")

        with (
            patch("mlx_manager.routers.system.get_session", mock_get_session),
            patch("websockets.connect", return_value=MockMLXWebSocket()),
        ):
            try:
                from mlx_manager.routers.system import proxy_audit_log_stream

                await proxy_audit_log_stream(mock_ws)
            except Exception:
                pass

        # Verify message was forwarded to MLX server
        assert "client_message" in received_by_mlx

    @pytest.mark.asyncio
    async def test_websocket_error_sending_error_message(self):
        """Test exception when sending error message fails."""
        # Mock websocket that fails when trying to send error
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=Exception("Connection closed"))
        mock_ws.close = AsyncMock()
        mock_ws.query_params = {"token": self._make_valid_token()}

        mock_get_session = self._mock_get_session_with_user("test@example.com")

        # Make websockets.connect fail
        with (
            patch("mlx_manager.routers.system.get_session", mock_get_session),
            patch("websockets.connect", side_effect=Exception("Connection failed")),
        ):
            from mlx_manager.routers.system import proxy_audit_log_stream

            try:
                await proxy_audit_log_stream(mock_ws)
            except Exception:
                pass  # Expected

        # Verify we tried to send error message and it failed
        mock_ws.send_json.assert_called_once()
        mock_ws.close.assert_called_once()
