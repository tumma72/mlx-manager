"""Tests for MLX Server database setup and session management."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_manager.mlx_server.database import (
    _get_engine,
    _get_session_factory,
    cleanup_old_logs,
    get_session,
    init_db,
    reset_for_testing,
)


@pytest.fixture(autouse=True)
def reset_db_state():
    """Reset database global state before and after each test."""
    reset_for_testing()
    yield
    reset_for_testing()


class TestEngineCreation:
    """Tests for _get_engine lazy initialization."""

    def test_get_engine_creates_engine(self):
        """_get_engine creates an AsyncEngine."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"

            engine = _get_engine()
            assert engine is not None

    def test_get_engine_returns_same_instance(self):
        """_get_engine returns cached engine on subsequent calls."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"

            engine1 = _get_engine()
            engine2 = _get_engine()
            assert engine1 is engine2


class TestSessionFactory:
    """Tests for _get_session_factory lazy initialization."""

    def test_get_session_factory_creates_factory(self):
        """_get_session_factory creates a sessionmaker."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"

            factory = _get_session_factory()
            assert factory is not None

    def test_get_session_factory_returns_same_instance(self):
        """_get_session_factory returns cached factory on subsequent calls."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"

            factory1 = _get_session_factory()
            factory2 = _get_session_factory()
            assert factory1 is factory2


class TestInitDb:
    """Tests for init_db table creation."""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables(self):
        """init_db creates AuditLog table."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"

            await init_db()
            # Should not raise - tables created successfully


class TestGetSession:
    """Tests for get_session context manager."""

    @pytest.mark.asyncio
    async def test_get_session_yields_session(self):
        """get_session yields an AsyncSession."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"

            await init_db()

            async with get_session() as session:
                assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_get_session_commits_on_success(self):
        """get_session commits on successful exit."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"

            await init_db()

            from mlx_manager.mlx_server.models.audit import AuditLog

            async with get_session() as session:
                log = AuditLog(
                    request_id="test-commit",
                    model="test-model",
                    backend_type="local",
                    endpoint="/v1/test",
                    duration_ms=100,
                    status="success",
                )
                session.add(log)

            # Verify it was committed by reading it back
            async with get_session() as session:
                from sqlmodel import select

                result = await session.execute(
                    select(AuditLog).where(AuditLog.request_id == "test-commit")
                )
                found = result.scalars().first()
                assert found is not None
                assert found.model == "test-model"

    @pytest.mark.asyncio
    async def test_get_session_rollback_on_error(self):
        """get_session rolls back on exception."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"

            await init_db()

            from mlx_manager.mlx_server.models.audit import AuditLog

            with pytest.raises(ValueError):
                async with get_session() as session:
                    log = AuditLog(
                        request_id="test-rollback",
                        model="test-model",
                        backend_type="local",
                        endpoint="/v1/test",
                        duration_ms=100,
                        status="error",
                    )
                    session.add(log)
                    raise ValueError("Force rollback")

            # Entry should NOT have been committed
            async with get_session() as session:
                from sqlmodel import select

                result = await session.execute(
                    select(AuditLog).where(AuditLog.request_id == "test-rollback")
                )
                found = result.scalars().first()
                assert found is None


class TestCleanupOldLogs:
    """Tests for cleanup_old_logs function."""

    @pytest.mark.asyncio
    async def test_cleanup_deletes_old_logs(self):
        """cleanup_old_logs removes logs older than retention period."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"
            mock_settings.return_value.audit_retention_days = 30

            await init_db()

            from mlx_manager.mlx_server.models.audit import AuditLog

            # Create old and new logs
            async with get_session() as session:
                old_log = AuditLog(
                    request_id="old-log",
                    model="test",
                    backend_type="local",
                    endpoint="/test",
                    duration_ms=100,
                    status="success",
                    timestamp=datetime.now(UTC) - timedelta(days=60),
                )
                new_log = AuditLog(
                    request_id="new-log",
                    model="test",
                    backend_type="local",
                    endpoint="/test",
                    duration_ms=100,
                    status="success",
                    timestamp=datetime.now(UTC),
                )
                session.add(old_log)
                session.add(new_log)

            deleted = await cleanup_old_logs()
            assert deleted == 1

            # Verify only new log remains
            async with get_session() as session:
                from sqlmodel import select

                result = await session.execute(select(AuditLog))
                remaining = result.scalars().all()
                assert len(remaining) == 1
                assert remaining[0].request_id == "new-log"

    @pytest.mark.asyncio
    async def test_cleanup_no_old_logs(self):
        """cleanup_old_logs returns 0 when no old logs exist."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"
            mock_settings.return_value.audit_retention_days = 30

            await init_db()

            from mlx_manager.mlx_server.models.audit import AuditLog

            # Only create a recent log
            async with get_session() as session:
                log = AuditLog(
                    request_id="recent",
                    model="test",
                    backend_type="local",
                    endpoint="/test",
                    duration_ms=100,
                    status="success",
                    timestamp=datetime.now(UTC),
                )
                session.add(log)

            deleted = await cleanup_old_logs()
            assert deleted == 0


class TestResetForTesting:
    """Tests for reset_for_testing function."""

    def test_reset_clears_globals(self):
        """reset_for_testing clears engine and session factory."""
        import mlx_manager.mlx_server.database as db_mod

        # Set to non-None values
        db_mod._engine = "fake_engine"
        db_mod._async_session = "fake_session"

        reset_for_testing()

        assert db_mod._engine is None
        assert db_mod._async_session is None
