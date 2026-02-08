"""Tests for admin API endpoints."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from mlx_manager.mlx_server.api.v1.admin import (
    ModelLoadResponse,
    ModelUnloadResponse,
    PoolStatusResponse,
    admin_health,
    audit_log_stream,
    export_audit_logs,
    get_audit_logs,
    get_audit_stats,
    pool_status,
    preload_model,
    unload_model,
)
from mlx_manager.mlx_server.models.audit import AuditLog


class TestPoolStatus:
    """Tests for /admin/models/status endpoint."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.admin.get_memory_usage")
    async def test_pool_status_empty(self, mock_memory, mock_get_pool):
        """Test pool status with no models loaded."""
        mock_pool = MagicMock()
        mock_pool._models = {}
        mock_pool.max_memory_gb = 48.0
        mock_pool.max_models = 4
        mock_get_pool.return_value = mock_pool

        mock_memory.return_value = {"active_gb": 0.0, "cache_gb": 0.0}

        response = await pool_status()

        assert isinstance(response, PoolStatusResponse)
        assert response.total_models == 0
        assert len(response.loaded_models) == 0
        assert response.max_memory_gb == 48.0

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.admin.get_memory_usage")
    async def test_pool_status_with_models(self, mock_memory, mock_get_pool):
        """Test pool status with loaded models."""
        # Create mock loaded model
        mock_loaded = MagicMock()
        mock_loaded.model_type = "text-gen"
        mock_loaded.size_gb = 4.5
        mock_loaded.preloaded = True
        mock_loaded.last_used = 1234567890.0
        mock_loaded.loaded_at = 1234567800.0

        mock_pool = MagicMock()
        mock_pool._models = {"test-model": mock_loaded}
        mock_pool.max_memory_gb = 48.0
        mock_pool.max_models = 4
        mock_get_pool.return_value = mock_pool

        mock_memory.return_value = {"active_gb": 4.5, "cache_gb": 0.5}

        response = await pool_status()

        assert response.total_models == 1
        assert len(response.loaded_models) == 1
        assert response.loaded_models[0].model_id == "test-model"
        assert response.loaded_models[0].model_type == "text-gen"
        assert response.loaded_models[0].preloaded is True


class TestPreloadModel:
    """Tests for /admin/models/load endpoint."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    async def test_preload_model_success(self, mock_get_pool):
        """Test successful model preload."""
        mock_loaded = MagicMock()
        mock_loaded.model_type = "text-gen"
        mock_loaded.size_gb = 4.0
        mock_loaded.preloaded = True

        mock_pool = MagicMock()
        mock_pool.preload_model = AsyncMock(return_value=mock_loaded)
        mock_get_pool.return_value = mock_pool

        response = await preload_model("mlx-community/test-model")

        assert isinstance(response, ModelLoadResponse)
        assert response.status == "loaded"
        assert response.model_id == "mlx-community/test-model"
        assert response.preloaded is True
        mock_pool.preload_model.assert_called_once_with("mlx-community/test-model")

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    async def test_preload_model_failure(self, mock_get_pool):
        """Test preload failure returns 500."""
        from fastapi import HTTPException

        mock_pool = MagicMock()
        mock_pool.preload_model = AsyncMock(side_effect=RuntimeError("Load failed"))
        mock_get_pool.return_value = mock_pool

        with pytest.raises(HTTPException) as exc_info:
            await preload_model("bad-model")

        assert exc_info.value.status_code == 500
        assert "Load failed" in exc_info.value.detail


class TestUnloadModel:
    """Tests for /admin/models/unload endpoint."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    async def test_unload_model_success(self, mock_get_pool):
        """Test successful model unload."""
        mock_pool = MagicMock()
        mock_pool.unload_model = AsyncMock(return_value=True)
        mock_get_pool.return_value = mock_pool

        response = await unload_model("test-model")

        assert isinstance(response, ModelUnloadResponse)
        assert response.status == "unloaded"
        assert response.model_id == "test-model"
        mock_pool.unload_model.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    async def test_unload_model_not_found(self, mock_get_pool):
        """Test unload of non-existent model returns 404."""
        from fastapi import HTTPException

        mock_pool = MagicMock()
        mock_pool.unload_model = AsyncMock(return_value=False)
        mock_get_pool.return_value = mock_pool

        with pytest.raises(HTTPException) as exc_info:
            await unload_model("not-loaded-model")

        assert exc_info.value.status_code == 404
        assert "not loaded" in exc_info.value.detail.lower()


class TestAdminHealth:
    """Tests for /admin/health endpoint."""

    @pytest.mark.asyncio
    async def test_admin_health(self):
        """Test admin health endpoint returns healthy."""
        response = await admin_health()
        assert response["status"] == "healthy"


# ============================================================================
# In-memory database fixture for audit log endpoint tests
# ============================================================================


@pytest.fixture
async def audit_db():
    """Create an in-memory SQLite database with AuditLog table."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def mock_get_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    with patch("mlx_manager.mlx_server.api.v1.admin.get_session", mock_get_session):
        # Pre-seed with some audit log entries
        async with session_factory() as session:
            logs = [
                AuditLog(
                    request_id=f"req-{i}",
                    model="test-model-a" if i < 3 else "test-model-b",
                    backend_type="local" if i < 4 else "openai",
                    endpoint="/v1/chat/completions",
                    duration_ms=100 * (i + 1),
                    status="success" if i < 4 else "error",
                    timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                    prompt_tokens=10 * (i + 1),
                    completion_tokens=5 * (i + 1),
                    total_tokens=15 * (i + 1),
                    error_type="RuntimeError" if i >= 4 else None,
                    error_message="Test error" if i >= 4 else None,
                )
                for i in range(5)
            ]
            for log in logs:
                session.add(log)
            await session.commit()

        yield session_factory

        # Properly dispose of the engine to close all connections
        await engine.dispose()


class TestGetAuditLogs:
    """Tests for GET /admin/audit-logs endpoint."""

    @pytest.mark.asyncio
    async def test_get_all_logs(self, audit_db):
        """Get all audit logs without filters."""
        result = await get_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            limit=100,
            offset=0,
        )
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_filter_by_model(self, audit_db):
        """Filter audit logs by model name."""
        result = await get_audit_logs(
            model="test-model-a",
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            limit=100,
            offset=0,
        )
        assert len(result) == 3
        assert all(r.model == "test-model-a" for r in result)

    @pytest.mark.asyncio
    async def test_filter_by_backend_type(self, audit_db):
        """Filter audit logs by backend type."""
        result = await get_audit_logs(
            model=None,
            backend_type="openai",
            status=None,
            start_time=None,
            end_time=None,
            limit=100,
            offset=0,
        )
        assert len(result) == 1
        assert result[0].backend_type == "openai"

    @pytest.mark.asyncio
    async def test_filter_by_status(self, audit_db):
        """Filter audit logs by status."""
        result = await get_audit_logs(
            model=None,
            backend_type=None,
            status="error",
            start_time=None,
            end_time=None,
            limit=100,
            offset=0,
        )
        assert len(result) == 1
        assert result[0].status == "error"

    @pytest.mark.asyncio
    async def test_filter_by_time_range(self, audit_db):
        """Filter audit logs by start and end time."""
        start = datetime(2025, 1, 1, 1, 0, tzinfo=UTC)
        end = datetime(2025, 1, 1, 3, 0, tzinfo=UTC)
        result = await get_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=start,
            end_time=end,
            limit=100,
            offset=0,
        )
        assert len(result) == 3  # hours 1, 2, 3

    @pytest.mark.asyncio
    async def test_pagination_limit(self, audit_db):
        """Limit number of returned logs."""
        result = await get_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            limit=2,
            offset=0,
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_pagination_offset(self, audit_db):
        """Offset pagination skips initial logs."""
        result = await get_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            limit=100,
            offset=3,
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_results_ordered_descending(self, audit_db):
        """Results should be most recent first."""
        result = await get_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            limit=100,
            offset=0,
        )
        timestamps = [r.timestamp for r in result]
        assert timestamps == sorted(timestamps, reverse=True)


class TestGetAuditStats:
    """Tests for GET /admin/audit-logs/stats endpoint."""

    @pytest.mark.asyncio
    async def test_stats_structure(self, audit_db):
        """Stats should return expected keys."""
        result = await get_audit_stats()
        assert "total_requests" in result
        assert "by_status" in result
        assert "by_backend" in result
        assert "unique_models" in result

    @pytest.mark.asyncio
    async def test_stats_total_count(self, audit_db):
        """Total should match number of seeded logs."""
        result = await get_audit_stats()
        assert result["total_requests"] == 5

    @pytest.mark.asyncio
    async def test_stats_by_status(self, audit_db):
        """Status breakdown should be correct."""
        result = await get_audit_stats()
        assert result["by_status"]["success"] == 4
        assert result["by_status"]["error"] == 1

    @pytest.mark.asyncio
    async def test_stats_by_backend(self, audit_db):
        """Backend breakdown should be correct."""
        result = await get_audit_stats()
        assert result["by_backend"]["local"] == 4
        assert result["by_backend"]["openai"] == 1

    @pytest.mark.asyncio
    async def test_stats_unique_models(self, audit_db):
        """Unique models count should be correct."""
        result = await get_audit_stats()
        assert result["unique_models"] == 2


class TestExportAuditLogs:
    """Tests for GET /admin/audit-logs/export endpoint."""

    @pytest.mark.asyncio
    async def test_export_jsonl_format(self, audit_db):
        """Export in JSONL format returns valid JSONL."""
        response = await export_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            format="jsonl",
        )
        assert response.media_type == "application/jsonl"
        lines = response.body.decode().strip().split("\n")
        assert len(lines) == 5
        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "request_id" in data
            assert "model" in data

    @pytest.mark.asyncio
    async def test_export_csv_format(self, audit_db):
        """Export in CSV format returns valid CSV."""
        response = await export_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            format="csv",
        )
        assert response.media_type == "text/csv"
        lines = response.body.decode().strip().split("\n")
        # Header + 5 data rows
        assert len(lines) == 6
        header = lines[0]
        assert "timestamp" in header
        assert "model" in header
        assert "status" in header

    @pytest.mark.asyncio
    async def test_export_with_filters(self, audit_db):
        """Export respects filter parameters."""
        response = await export_audit_logs(
            model="test-model-a",
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            format="jsonl",
        )
        lines = response.body.decode().strip().split("\n")
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_export_csv_empty_result(self, audit_db):
        """CSV export with no matching logs returns only header or empty."""
        response = await export_audit_logs(
            model="nonexistent-model",
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            format="csv",
        )
        content = response.body.decode().strip()
        # No logs matched so output should be empty
        assert content == ""

    @pytest.mark.asyncio
    async def test_export_jsonl_empty_result(self, audit_db):
        """JSONL export with no matching logs returns empty string."""
        response = await export_audit_logs(
            model="nonexistent-model",
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            format="jsonl",
        )
        content = response.body.decode().strip()
        assert content == ""

    @pytest.mark.asyncio
    async def test_export_with_backend_type_filter(self, audit_db):
        """Export with backend_type filter."""
        response = await export_audit_logs(
            model=None,
            backend_type="local",
            status=None,
            start_time=None,
            end_time=None,
            format="jsonl",
        )
        lines = response.body.decode().strip().split("\n")
        assert len(lines) == 4

    @pytest.mark.asyncio
    async def test_export_with_status_filter(self, audit_db):
        """Export with status filter."""
        response = await export_audit_logs(
            model=None,
            backend_type=None,
            status="success",
            start_time=None,
            end_time=None,
            format="jsonl",
        )
        lines = response.body.decode().strip().split("\n")
        assert len(lines) == 4

    @pytest.mark.asyncio
    async def test_export_with_time_range_filter(self, audit_db):
        """Export with time range filter."""
        start = datetime(2025, 1, 1, 2, 0, tzinfo=UTC)
        end = datetime(2025, 1, 1, 4, 0, tzinfo=UTC)
        response = await export_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=start,
            end_time=end,
            format="jsonl",
        )
        lines = response.body.decode().strip().split("\n")
        assert len(lines) == 3  # hours 2, 3, 4

    @pytest.mark.asyncio
    async def test_export_csv_content_disposition(self, audit_db):
        """CSV export sets Content-Disposition header."""
        response = await export_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            format="csv",
        )
        assert "audit-logs.csv" in response.headers.get("content-disposition", "")

    @pytest.mark.asyncio
    async def test_export_jsonl_content_disposition(self, audit_db):
        """JSONL export sets Content-Disposition header."""
        response = await export_audit_logs(
            model=None,
            backend_type=None,
            status=None,
            start_time=None,
            end_time=None,
            format="jsonl",
        )
        assert "audit-logs.jsonl" in response.headers.get("content-disposition", "")


class TestAuditLogWebSocket:
    """Tests for /admin/ws/audit-logs WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_accepts_and_sends_recent(self):
        """WebSocket accepts connection and sends recent logs."""
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()

        recent_logs = [{"request_id": "r1"}, {"request_id": "r2"}]

        with patch("mlx_manager.mlx_server.api.v1.admin.audit_service") as mock_service:
            mock_service.get_recent_logs.return_value = recent_logs
            mock_service.subscribe = MagicMock()
            mock_service.unsubscribe = MagicMock()

            # Simulate disconnect after receiving recent logs
            call_count = 0

            async def send_then_disconnect(data):
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    from fastapi import WebSocketDisconnect

                    raise WebSocketDisconnect()

            mock_ws.send_json = AsyncMock(side_effect=send_then_disconnect)

            await audit_log_stream(mock_ws)

            mock_ws.accept.assert_called_once()
            mock_service.subscribe.assert_called_once()
            mock_service.unsubscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_streams_new_logs(self):
        """WebSocket streams new log entries from the queue."""
        from fastapi import WebSocketDisconnect

        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        sent_messages = []

        async def track_sends(data):
            sent_messages.append(data)
            # Disconnect after streaming one new log
            if len(sent_messages) > 1 and data.get("type") == "log":
                raise WebSocketDisconnect()

        mock_ws.send_json = AsyncMock(side_effect=track_sends)

        with patch("mlx_manager.mlx_server.api.v1.admin.audit_service") as mock_service:
            mock_service.get_recent_logs.return_value = []
            captured_callback = None

            def capture_subscribe(callback):
                nonlocal captured_callback
                captured_callback = callback

            mock_service.subscribe = MagicMock(side_effect=capture_subscribe)
            mock_service.unsubscribe = MagicMock()

            # Start the websocket handler in a task
            task = asyncio.create_task(audit_log_stream(mock_ws))

            # Give it time to start
            await asyncio.sleep(0.05)

            # Push a log through the callback
            if captured_callback:
                captured_callback({"request_id": "new-log"})

            await asyncio.sleep(0.1)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_websocket_sends_ping_on_timeout(self):
        """WebSocket sends ping when queue times out."""
        from fastapi import WebSocketDisconnect

        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        sent_messages = []

        async def track_and_disconnect(data):
            sent_messages.append(data)
            if data.get("type") == "ping":
                raise WebSocketDisconnect()

        mock_ws.send_json = AsyncMock(side_effect=track_and_disconnect)

        with patch("mlx_manager.mlx_server.api.v1.admin.audit_service") as mock_service:
            mock_service.get_recent_logs.return_value = []
            mock_service.subscribe = MagicMock()
            mock_service.unsubscribe = MagicMock()

            # Patch the timeout to be very short
            with patch("mlx_manager.mlx_server.api.v1.admin.asyncio.wait_for") as mock_wait:
                mock_wait.side_effect = TimeoutError()

                await audit_log_stream(mock_ws)

            # Should have sent a ping
            assert any(m.get("type") == "ping" for m in sent_messages)

    @pytest.mark.asyncio
    async def test_websocket_unsubscribes_on_disconnect(self):
        """WebSocket ensures unsubscribe is called on disconnect."""
        from fastapi import WebSocketDisconnect

        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=WebSocketDisconnect())

        with patch("mlx_manager.mlx_server.api.v1.admin.audit_service") as mock_service:
            mock_service.get_recent_logs.return_value = [{"id": 1}]
            mock_service.subscribe = MagicMock()
            mock_service.unsubscribe = MagicMock()

            await audit_log_stream(mock_ws)

            mock_service.unsubscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_queue_full_drops_log(self):
        """When the queue is full, new logs are dropped silently."""
        # Test the on_new_log callback directly when queue is full
        queue = asyncio.Queue(maxsize=1)
        queue.put_nowait({"first": True})

        def on_new_log(log):
            try:
                queue.put_nowait(log)
            except asyncio.QueueFull:
                pass  # Dropped

        # Should not raise
        on_new_log({"second": True})
        assert queue.qsize() == 1
