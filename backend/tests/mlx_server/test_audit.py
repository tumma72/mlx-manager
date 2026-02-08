"""Tests for audit logging service."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlx_manager.mlx_server.models.audit import AuditLog, AuditLogFilter, AuditLogResponse
from mlx_manager.mlx_server.services.audit import AuditService, RequestContext


class TestRequestContext:
    """Test RequestContext dataclass."""

    def test_duration_ms_calculation(self) -> None:
        """duration_ms calculates time since start."""
        ctx = RequestContext(
            request_id="test-123",
            model="test-model",
            endpoint="/v1/test",
        )
        time.sleep(0.1)  # 100ms
        duration = ctx.duration_ms

        # Allow some variance
        assert 100 <= duration <= 250

    def test_default_backend_type(self) -> None:
        """Default backend_type is 'local'."""
        ctx = RequestContext(
            request_id="test-123",
            model="test-model",
            endpoint="/v1/test",
        )
        assert ctx.backend_type == "local"

    def test_optional_fields_default_to_none(self) -> None:
        """Optional fields default to None."""
        ctx = RequestContext(
            request_id="test-123",
            model="test-model",
            endpoint="/v1/test",
        )
        assert ctx.status is None
        assert ctx.prompt_tokens is None
        assert ctx.completion_tokens is None
        assert ctx.total_tokens is None
        assert ctx.error_type is None
        assert ctx.error_message is None


class TestAuditService:
    """Test AuditService."""

    @pytest.fixture
    def service(self) -> AuditService:
        return AuditService(buffer_size=10)

    @pytest.mark.asyncio
    async def test_track_request_success(self, service: AuditService) -> None:
        """track_request logs successful requests."""
        # Mock the database write
        mock_write = AsyncMock()
        service._write_log_entry = mock_write

        async with service.track_request(
            request_id="req-123",
            model="test-model",
            endpoint="/v1/chat/completions",
        ) as ctx:
            ctx.prompt_tokens = 100
            ctx.completion_tokens = 50
            ctx.total_tokens = 150

        # Wait for background task
        await asyncio.sleep(0.05)

        # Verify write was called
        mock_write.assert_called_once()
        log_entry = mock_write.call_args[0][0]
        assert log_entry.request_id == "req-123"
        assert log_entry.status == "success"
        assert log_entry.prompt_tokens == 100
        assert log_entry.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_track_request_error(self, service: AuditService) -> None:
        """track_request logs errors with error details."""
        mock_write = AsyncMock()
        service._write_log_entry = mock_write

        with pytest.raises(ValueError):
            async with service.track_request(
                request_id="req-err",
                model="test-model",
                endpoint="/v1/test",
            ):
                raise ValueError("Test error")

        await asyncio.sleep(0.05)

        mock_write.assert_called_once()
        log_entry = mock_write.call_args[0][0]
        assert log_entry.status == "error"
        assert log_entry.error_type == "ValueError"
        assert "Test error" in log_entry.error_message

    @pytest.mark.asyncio
    async def test_track_request_timeout(self, service: AuditService) -> None:
        """track_request logs timeouts correctly."""
        mock_write = AsyncMock()
        service._write_log_entry = mock_write

        with pytest.raises(asyncio.TimeoutError):
            async with service.track_request(
                request_id="req-timeout",
                model="test-model",
                endpoint="/v1/test",
            ):
                raise TimeoutError()

        await asyncio.sleep(0.05)

        mock_write.assert_called_once()
        log_entry = mock_write.call_args[0][0]
        assert log_entry.status == "timeout"
        assert log_entry.error_type == "TimeoutError"

    @pytest.mark.asyncio
    async def test_log_request_manual(self, service: AuditService) -> None:
        """log_request creates log entry without context manager."""
        mock_write = AsyncMock()
        service._write_log_entry = mock_write

        await service.log_request(
            request_id="req-manual",
            model="test-model",
            endpoint="/v1/chat/completions",
            backend_type="openai",
            duration_ms=500,
            status="success",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )

        await asyncio.sleep(0.05)

        mock_write.assert_called_once()
        log_entry = mock_write.call_args[0][0]
        assert log_entry.request_id == "req-manual"
        assert log_entry.backend_type == "openai"
        assert log_entry.duration_ms == 500

    def test_subscribe_unsubscribe(self, service: AuditService) -> None:
        """Subscribers can be added and removed."""
        callback = MagicMock()

        service.subscribe(callback)
        assert callback in service._subscribers

        service.unsubscribe(callback)
        assert callback not in service._subscribers

    def test_buffer_size_limit(self, service: AuditService) -> None:
        """Recent logs buffer respects size limit."""
        for i in range(20):
            service._recent_logs.append({"id": i})
            if len(service._recent_logs) > service._buffer_size:
                service._recent_logs.pop(0)

        assert len(service._recent_logs) == 10
        assert service._recent_logs[0]["id"] == 10  # Oldest kept

    def test_get_recent_logs(self, service: AuditService) -> None:
        """get_recent_logs returns copy of buffer."""
        service._recent_logs.append({"id": 1})
        service._recent_logs.append({"id": 2})

        result = service.get_recent_logs()
        assert len(result) == 2
        # Verify it's a copy
        result.append({"id": 3})
        assert len(service._recent_logs) == 2


class TestAuditLogModel:
    """Test AuditLog SQLModel."""

    def test_audit_log_creation(self) -> None:
        """AuditLog can be created with required fields."""
        log = AuditLog(
            request_id="test-123",
            model="test-model",
            backend_type="local",
            endpoint="/v1/chat/completions",
            duration_ms=1500,
            status="success",
        )

        assert log.request_id == "test-123"
        assert log.timestamp is not None
        assert log.prompt_tokens is None  # Optional

    def test_audit_log_with_tokens(self) -> None:
        """AuditLog can include token counts."""
        log = AuditLog(
            request_id="test-456",
            model="gpt-4",
            backend_type="openai",
            endpoint="/v1/chat/completions",
            duration_ms=500,
            status="success",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )

        assert log.total_tokens == 300

    def test_audit_log_error(self) -> None:
        """AuditLog can capture error details."""
        log = AuditLog(
            request_id="test-err",
            model="test-model",
            backend_type="local",
            endpoint="/v1/test",
            duration_ms=100,
            status="error",
            error_type="RuntimeError",
            error_message="Something went wrong",
        )

        assert log.status == "error"
        assert log.error_type == "RuntimeError"

    def test_audit_log_no_content_fields(self) -> None:
        """AuditLog has no fields for prompt/response content (privacy)."""
        # Verify these fields don't exist
        log = AuditLog(
            request_id="test",
            model="test",
            backend_type="local",
            endpoint="/test",
            duration_ms=0,
            status="success",
        )
        assert not hasattr(log, "prompt")
        assert not hasattr(log, "response")
        assert not hasattr(log, "content")
        assert not hasattr(log, "messages")


class TestAuditLogResponse:
    """Test AuditLogResponse model."""

    def test_response_has_id(self) -> None:
        """AuditLogResponse includes id field."""
        response = AuditLogResponse(
            id=1,
            request_id="test-123",
            model="test-model",
            backend_type="local",
            endpoint="/v1/test",
            duration_ms=100,
            status="success",
        )
        assert response.id == 1


class TestAuditLogFilter:
    """Test AuditLogFilter model."""

    def test_filter_defaults(self) -> None:
        """AuditLogFilter has sensible defaults."""
        f = AuditLogFilter()
        assert f.limit == 100
        assert f.offset == 0
        assert f.model is None
        assert f.backend_type is None

    def test_filter_limit_validation(self) -> None:
        """AuditLogFilter validates limit."""
        f = AuditLogFilter(limit=500)
        assert f.limit == 500

        # Max is 1000
        f2 = AuditLogFilter(limit=1000)
        assert f2.limit == 1000


class TestWriteLogEntry:
    """Tests for _write_log_entry with real database operations."""

    @pytest.mark.asyncio
    async def test_write_log_entry_notifies_subscribers(self) -> None:
        """_write_log_entry calls subscriber callbacks with log dict."""
        from contextlib import asynccontextmanager
        from unittest.mock import patch

        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlmodel import SQLModel

        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
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

        service = AuditService(buffer_size=5)
        callback = MagicMock()
        service.subscribe(callback)

        log_entry = AuditLog(
            request_id="test-write",
            model="test-model",
            backend_type="local",
            endpoint="/v1/test",
            duration_ms=100,
            status="success",
        )

        try:
            with patch("mlx_manager.mlx_server.services.audit.get_session", mock_get_session):
                await service._write_log_entry(log_entry)

            # Subscriber should have been called
            callback.assert_called_once()
            log_dict = callback.call_args[0][0]
            assert log_dict["request_id"] == "test-write"

            # Recent logs should have the entry
            recent = service.get_recent_logs()
            assert len(recent) == 1
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_write_log_entry_handles_subscriber_error(self) -> None:
        """_write_log_entry handles subscriber callback errors gracefully."""
        from contextlib import asynccontextmanager
        from unittest.mock import patch

        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlmodel import SQLModel

        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
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

        service = AuditService(buffer_size=5)
        bad_callback = MagicMock(side_effect=RuntimeError("callback crash"))
        service.subscribe(bad_callback)

        log_entry = AuditLog(
            request_id="test-error",
            model="test-model",
            backend_type="local",
            endpoint="/v1/test",
            duration_ms=100,
            status="success",
        )

        try:
            with patch("mlx_manager.mlx_server.services.audit.get_session", mock_get_session):
                # Should not raise even though callback fails
                await service._write_log_entry(log_entry)
        finally:
            await engine.dispose()

    @pytest.mark.asyncio
    async def test_write_log_entry_handles_db_error(self) -> None:
        """_write_log_entry handles database errors gracefully."""
        from contextlib import asynccontextmanager
        from unittest.mock import patch

        @asynccontextmanager
        async def failing_get_session():
            raise RuntimeError("Database connection failed")
            yield  # noqa: RET503 - unreachable, needed for async generator

        service = AuditService(buffer_size=5)
        log_entry = AuditLog(
            request_id="test-db-error",
            model="test-model",
            backend_type="local",
            endpoint="/v1/test",
            duration_ms=100,
            status="success",
        )

        with patch("mlx_manager.mlx_server.services.audit.get_session", failing_get_session):
            # Should not raise
            await service._write_log_entry(log_entry)

    @pytest.mark.asyncio
    async def test_write_log_entry_trims_buffer(self) -> None:
        """_write_log_entry trims buffer when exceeding buffer_size."""
        from contextlib import asynccontextmanager
        from unittest.mock import patch

        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlmodel import SQLModel

        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
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

        service = AuditService(buffer_size=2)

        try:
            with patch("mlx_manager.mlx_server.services.audit.get_session", mock_get_session):
                for i in range(5):
                    log_entry = AuditLog(
                        request_id=f"test-{i}",
                        model="test-model",
                        backend_type="local",
                        endpoint="/v1/test",
                        duration_ms=100,
                        status="success",
                    )
                    await service._write_log_entry(log_entry)

            # Buffer should be trimmed to 2
            assert len(service.get_recent_logs()) == 2
        finally:
            await engine.dispose()
