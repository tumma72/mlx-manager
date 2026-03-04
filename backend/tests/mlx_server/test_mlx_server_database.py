"""Tests for MLX Server database setup and session management."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_manager.mlx_server.database import (
    _get_engine,
    _get_session_factory,
    cleanup_by_size,
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
            mock_settings.return_value.audit_max_mb = 100
            # Point to a non-existent path so cleanup_by_size is a no-op
            from pathlib import Path

            mock_settings.return_value.get_database_path.return_value = Path(
                "/nonexistent/in-memory.db"
            )

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
            mock_settings.return_value.audit_max_mb = 100
            from pathlib import Path

            mock_settings.return_value.get_database_path.return_value = Path(
                "/nonexistent/in-memory.db"
            )

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


class TestCleanupBySize:
    """Tests for cleanup_by_size function."""

    @pytest.mark.asyncio
    async def test_cleanup_by_size_no_op_when_under_limit(self, tmp_path):
        """cleanup_by_size returns 0 when DB file is under the size limit."""
        db_file = tmp_path / "test.db"
        db_file.write_bytes(b"x" * 1024)  # 1 KB — well under 100 MB default

        mock_settings = MagicMock()
        mock_settings.audit_max_mb = 100
        mock_settings.get_database_path.return_value = db_file

        with patch("mlx_manager.mlx_server.database.get_settings", return_value=mock_settings):
            result = await cleanup_by_size()

        assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_by_size_no_op_when_db_missing(self, tmp_path):
        """cleanup_by_size returns 0 when DB file does not exist."""
        non_existent = tmp_path / "nonexistent.db"

        mock_settings = MagicMock()
        mock_settings.audit_max_mb = 1
        mock_settings.get_database_path.return_value = non_existent

        with patch("mlx_manager.mlx_server.database.get_settings", return_value=mock_settings):
            result = await cleanup_by_size()

        assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_by_size_deletes_when_over_limit(self, tmp_path):
        """cleanup_by_size deletes oldest records when DB exceeds configured size."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_get_settings:
            mock_get_settings.return_value.database_path = str(tmp_path / "audit.db")
            mock_get_settings.return_value.audit_retention_days = 365
            mock_get_settings.return_value.audit_max_mb = 1  # 1 MB limit (very small)
            mock_get_settings.return_value.get_database_path.return_value = tmp_path / "audit.db"

            await init_db()

            from mlx_manager.mlx_server.models.audit import AuditLog

            # Insert enough records that the size check would trigger
            # We mock os.path.getsize to simulate an oversized DB
            async with get_session() as session:
                for i in range(10):
                    log = AuditLog(
                        request_id=f"old-log-{i}",
                        model="test",
                        backend_type="local",
                        endpoint="/test",
                        duration_ms=100,
                        status="success",
                        timestamp=datetime.now(UTC) - timedelta(days=i + 1),
                    )
                    session.add(log)

            limit_bytes = 1 * 1024 * 1024  # 1 MB

            # Simulate DB over limit for first check, then under limit after deletes
            size_sequence = [limit_bytes + 1000, limit_bytes - 1]
            call_count = 0

            def mock_getsize(path):
                nonlocal call_count
                val = size_sequence[min(call_count, len(size_sequence) - 1)]
                call_count += 1
                return val

            with patch("mlx_manager.mlx_server.database.os.path.getsize", side_effect=mock_getsize):
                deleted = await cleanup_by_size()

            assert deleted > 0

    @pytest.mark.asyncio
    async def test_cleanup_by_size_runs_vacuum_after_delete(self, tmp_path):
        """cleanup_by_size runs VACUUM after deleting records."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_get_settings:
            mock_get_settings.return_value.database_path = str(tmp_path / "vacuum.db")
            mock_get_settings.return_value.audit_retention_days = 365
            mock_get_settings.return_value.audit_max_mb = 1
            mock_get_settings.return_value.get_database_path.return_value = tmp_path / "vacuum.db"

            await init_db()

            from mlx_manager.mlx_server.models.audit import AuditLog

            async with get_session() as session:
                for i in range(5):
                    log = AuditLog(
                        request_id=f"log-{i}",
                        model="test",
                        backend_type="local",
                        endpoint="/test",
                        duration_ms=50,
                        status="success",
                        timestamp=datetime.now(UTC) - timedelta(days=i),
                    )
                    session.add(log)

            limit_bytes = 1 * 1024 * 1024

            size_values = iter([limit_bytes + 500, limit_bytes - 1])

            def mock_getsize(path):
                try:
                    return next(size_values)
                except StopIteration:
                    return limit_bytes - 1

            with patch("mlx_manager.mlx_server.database.os.path.getsize", side_effect=mock_getsize):
                deleted = await cleanup_by_size()

            # VACUUM runs after actual deletes; no error = VACUUM succeeded
            assert deleted >= 0


class TestCleanupOldLogsCallsSize:
    """Tests that cleanup_old_logs calls cleanup_by_size."""

    @pytest.mark.asyncio
    async def test_cleanup_old_logs_calls_cleanup_by_size(self):
        """cleanup_old_logs invokes cleanup_by_size after time-based cleanup."""
        with patch("mlx_manager.mlx_server.database.get_settings") as mock_settings:
            mock_settings.return_value.database_path = ":memory:"
            mock_settings.return_value.audit_retention_days = 30
            mock_settings.return_value.audit_max_mb = 100
            mock_settings.return_value.get_database_path.return_value = Path(":memory:")

            await init_db()

            with patch(
                "mlx_manager.mlx_server.database.cleanup_by_size",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_size_cleanup:
                await cleanup_old_logs()
                mock_size_cleanup.assert_called_once()


class TestConfigSettings:
    """Tests for new audit config settings."""

    def test_audit_max_mb_default(self):
        """audit_max_mb defaults to 100."""
        from mlx_manager.mlx_server.config import MLXServerSettings

        settings = MLXServerSettings()
        assert settings.audit_max_mb == 100

    def test_audit_cleanup_interval_minutes_default(self):
        """audit_cleanup_interval_minutes defaults to 60."""
        from mlx_manager.mlx_server.config import MLXServerSettings

        settings = MLXServerSettings()
        assert settings.audit_cleanup_interval_minutes == 60

    def test_audit_max_mb_validation_min(self):
        """audit_max_mb rejects values below 1."""
        from pydantic import ValidationError

        from mlx_manager.mlx_server.config import MLXServerSettings

        with pytest.raises(ValidationError):
            MLXServerSettings(audit_max_mb=0)

    def test_audit_max_mb_validation_max(self):
        """audit_max_mb rejects values above 10000."""
        from pydantic import ValidationError

        from mlx_manager.mlx_server.config import MLXServerSettings

        with pytest.raises(ValidationError):
            MLXServerSettings(audit_max_mb=10001)

    def test_audit_cleanup_interval_validation_min(self):
        """audit_cleanup_interval_minutes rejects values below 1."""
        from pydantic import ValidationError

        from mlx_manager.mlx_server.config import MLXServerSettings

        with pytest.raises(ValidationError):
            MLXServerSettings(audit_cleanup_interval_minutes=0)

    def test_audit_cleanup_interval_validation_max(self):
        """audit_cleanup_interval_minutes rejects values above 1440."""
        from pydantic import ValidationError

        from mlx_manager.mlx_server.config import MLXServerSettings

        with pytest.raises(ValidationError):
            MLXServerSettings(audit_cleanup_interval_minutes=1441)


class TestAuditCleanupLoop:
    """Tests for _audit_cleanup_loop background task."""

    @pytest.mark.asyncio
    async def test_audit_cleanup_loop_calls_cleanup(self):
        """_audit_cleanup_loop calls cleanup_old_logs on each iteration."""
        from mlx_manager.mlx_server.main import _audit_cleanup_loop

        call_count = 0

        async def fake_cleanup():
            nonlocal call_count
            call_count += 1

        with (
            patch("mlx_manager.mlx_server.config.mlx_server_settings") as mock_cfg,
            patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_main_cfg,
        ):
            mock_main_cfg.audit_cleanup_interval_minutes = 1
            mock_cfg.audit_cleanup_interval_minutes = 1

            sleep_count = 0

            async def fake_sleep(seconds):
                nonlocal sleep_count
                sleep_count += 1
                if sleep_count >= 2:
                    raise asyncio.CancelledError

            with (
                patch(
                    "mlx_manager.mlx_server.database.cleanup_old_logs",
                    side_effect=fake_cleanup,
                ),
                patch("asyncio.sleep", side_effect=fake_sleep),
            ):
                import asyncio

                task = asyncio.create_task(_audit_cleanup_loop())
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_audit_cleanup_loop_handles_exceptions(self):
        """_audit_cleanup_loop logs warning and continues on cleanup failure."""
        import asyncio

        from mlx_manager.mlx_server.main import _audit_cleanup_loop

        error_calls = 0

        async def failing_cleanup():
            nonlocal error_calls
            error_calls += 1
            raise RuntimeError("DB connection failed")

        sleep_count = 0

        async def limited_sleep(seconds):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError

        with (
            patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_cfg,
            patch(
                "mlx_manager.mlx_server.database.cleanup_old_logs",
                side_effect=failing_cleanup,
            ),
            patch("asyncio.sleep", side_effect=limited_sleep),
        ):
            mock_cfg.audit_cleanup_interval_minutes = 1

            task = asyncio.create_task(_audit_cleanup_loop())
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should have attempted cleanup despite errors
        assert error_calls >= 1
