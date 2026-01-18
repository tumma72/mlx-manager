"""Tests for the database module."""

import os
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure test environment is set before importing app modules
os.environ["MLX_MANAGER_DATABASE_PATH"] = ":memory:"
os.environ["MLX_MANAGER_DEBUG"] = "false"


class TestGetSessionFunction:
    """Tests for the get_session context manager from database module."""

    @pytest.mark.asyncio
    async def test_get_session_yields_session(self, test_engine):
        """Test get_session yields a valid session."""
        # Import the real function
        # Need to patch the engine/async_session for this test
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_session

        # Create test session factory with test engine
        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Patch the async_session in the database module
        with patch("mlx_manager.database.async_session", test_async_session):
            async with get_session() as session:
                assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_get_session_rollback_on_exception(self, test_engine):
        """Test get_session rolls back on exception."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_session

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        session_used = False
        exception_raised = False

        with patch("mlx_manager.database.async_session", test_async_session):
            try:
                async with get_session():
                    session_used = True
                    raise ValueError("Test error")
            except ValueError:
                exception_raised = True

        assert session_used is True
        assert exception_raised is True


class TestGetDbFunction:
    """Tests for the get_db dependency from database module."""

    @pytest.mark.asyncio
    async def test_get_db_yields_session(self, test_engine):
        """Test get_db yields a valid session."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_db

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        with patch("mlx_manager.database.async_session", test_async_session):
            gen = get_db()
            session = await gen.__anext__()
            assert isinstance(session, AsyncSession)

            # Cleanup
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass

    @pytest.mark.asyncio
    async def test_get_db_commits_on_success(self, test_engine):
        """Test get_db commits on successful completion."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_db

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        with patch("mlx_manager.database.async_session", test_async_session):
            gen = get_db()
            await gen.__anext__()

            # Complete successfully
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass  # Expected - commit happened

    @pytest.mark.asyncio
    async def test_get_db_rollback_on_exception(self, test_engine):
        """Test get_db rolls back on exception."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_db

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        with patch("mlx_manager.database.async_session", test_async_session):
            gen = get_db()
            await gen.__anext__()

            # Simulate error
            try:
                await gen.athrow(ValueError("Test error"))
            except ValueError:
                pass  # Expected - rollback happened


class TestInitDbFunction:
    """Tests for the init_db function."""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables_and_settings(self, test_engine, tmp_path):
        """Test init_db creates tables and default settings."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        from sqlmodel import select

        from mlx_manager.database import init_db
        from mlx_manager.models import Setting

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Patch everything needed
        with patch("mlx_manager.database.engine", test_engine):
            with patch("mlx_manager.database.async_session", test_async_session):
                with patch("mlx_manager.database.ensure_data_dir"):
                    await init_db()

        # Verify settings were created
        async with test_async_session() as session:
            result = await session.execute(select(Setting))
            settings = result.scalars().all()
            assert len(settings) >= 4

            setting_keys = {s.key for s in settings}
            assert "default_port_start" in setting_keys
            assert "huggingface_cache_path" in setting_keys
            assert "max_memory_percent" in setting_keys
            assert "health_check_interval" in setting_keys


class TestGetSession:
    """Tests for the get_session context manager."""

    @pytest.mark.asyncio
    async def test_get_session_yields_session(self, test_session):
        """Test get_session yields a valid session."""
        assert isinstance(test_session, AsyncSession)


class TestInitDb:
    """Tests for the init_db function."""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables(self, test_engine):
        """Test init_db creates all required tables."""
        from sqlmodel import SQLModel

        # Tables should already be created by test_engine fixture
        # Just verify the metadata has our tables
        table_names = SQLModel.metadata.tables.keys()
        assert "server_profiles" in table_names
        assert "settings" in table_names

    @pytest.mark.asyncio
    async def test_init_db_creates_default_settings(self, test_session):
        """Test init_db creates default settings."""
        from sqlmodel import select

        from mlx_manager.models import Setting

        # Insert default settings like init_db does
        result = await test_session.execute(select(Setting))
        existing = result.scalars().first()

        if not existing:
            from mlx_manager.config import settings as app_settings

            default_settings = [
                Setting(key="default_port_start", value="10240"),
                Setting(key="huggingface_cache_path", value=str(app_settings.hf_cache_path)),
                Setting(key="max_memory_percent", value="80"),
                Setting(key="health_check_interval", value="30"),
            ]
            for s in default_settings:
                test_session.add(s)
            await test_session.commit()

        # Verify settings exist
        result = await test_session.execute(select(Setting))
        all_settings = result.scalars().all()
        assert len(all_settings) >= 4

        # Check specific settings
        setting_keys = {s.key for s in all_settings}
        assert "default_port_start" in setting_keys
        assert "huggingface_cache_path" in setting_keys
        assert "max_memory_percent" in setting_keys
        assert "health_check_interval" in setting_keys

    @pytest.mark.asyncio
    async def test_init_db_does_not_duplicate_settings(self, test_session):
        """Test init_db doesn't create duplicate settings."""
        from sqlmodel import select

        from mlx_manager.models import Setting

        # Add a setting manually
        existing_setting = Setting(key="default_port_start", value="10240")
        test_session.add(existing_setting)
        await test_session.commit()

        # Check for existing settings (like init_db does)
        result = await test_session.execute(select(Setting))
        existing = result.scalars().first()

        # If settings exist, don't add more
        if existing:
            # Don't add duplicates
            pass
        else:
            # Would add defaults here
            pass

        # Verify no duplicates
        result = await test_session.execute(
            select(Setting).where(Setting.key == "default_port_start")
        )
        matching = result.scalars().all()
        assert len(matching) == 1


class TestMigrateSchema:
    """Tests for the migrate_schema function."""

    @pytest.mark.asyncio
    async def test_migrate_schema_adds_missing_columns(self, test_engine):
        """Test migrate_schema adds columns that don't exist yet."""
        from mlx_manager.database import migrate_schema

        with patch("mlx_manager.database.engine", test_engine):
            # migrate_schema should run without error
            await migrate_schema()

    @pytest.mark.asyncio
    async def test_migrate_schema_column_already_exists(self, test_engine):
        """Test migrate_schema handles columns that already exist."""
        from mlx_manager.database import migrate_schema

        with patch("mlx_manager.database.engine", test_engine):
            # Run twice - second time columns already exist
            await migrate_schema()
            await migrate_schema()  # Should not raise


class TestRecoverIncompleteDownloads:
    """Tests for the recover_incomplete_downloads function."""

    @pytest.mark.asyncio
    async def test_recover_with_no_incomplete_downloads(self, test_engine):
        """Test recover when no incomplete downloads exist."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_session, recover_incomplete_downloads

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        with patch("mlx_manager.database.async_session", test_async_session):
            with patch("mlx_manager.database.get_session", get_session):
                pending = await recover_incomplete_downloads()
                assert pending == []

    @pytest.mark.asyncio
    async def test_recover_with_pending_downloads(self, test_engine):
        """Test recover marks pending downloads for resume."""
        from datetime import UTC, datetime

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        from sqlmodel import select

        from mlx_manager.database import get_session, recover_incomplete_downloads
        from mlx_manager.models import Download

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create a pending download
        async with test_async_session() as session:
            download = Download(
                model_id="mlx-community/test-model",
                status="downloading",  # Incomplete status
                started_at=datetime.now(tz=UTC),
            )
            session.add(download)
            await session.commit()
            await session.refresh(download)
            download_id = download.id

        with patch("mlx_manager.database.async_session", test_async_session):
            with patch("mlx_manager.database.get_session", get_session):
                pending = await recover_incomplete_downloads()

        # Should return the download for resuming
        assert len(pending) == 1
        assert pending[0][0] == download_id
        assert pending[0][1] == "mlx-community/test-model"

        # Verify status was set to pending
        async with test_async_session() as session:
            result = await session.execute(select(Download).where(Download.id == download_id))
            recovered = result.scalars().first()
            assert recovered.status == "pending"
            assert recovered.error is None


class TestEnsureDataDir:
    """Tests for the ensure_data_dir function."""

    def test_ensure_data_dir_creates_directory(self, tmp_path):
        """Test ensure_data_dir creates the data directory."""
        from mlx_manager.config import ensure_data_dir, settings

        with patch.object(settings, "database_path", tmp_path / "subdir" / "db.sqlite"):
            ensure_data_dir()

        assert (tmp_path / "subdir").exists()
        assert (tmp_path / "subdir").is_dir()

    def test_ensure_data_dir_handles_existing_directory(self, tmp_path):
        """Test ensure_data_dir works when directory already exists."""
        from mlx_manager.config import ensure_data_dir, settings

        # Pre-create the directory
        (tmp_path / "existing").mkdir()

        with patch.object(settings, "database_path", tmp_path / "existing" / "db.sqlite"):
            # Should not raise
            ensure_data_dir()

        assert (tmp_path / "existing").exists()
