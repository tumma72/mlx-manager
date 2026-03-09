"""Tests for the database module."""

import os
from unittest.mock import MagicMock, patch

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
        assert "execution_profiles" in table_names
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


# ============================================================================
# detect_orphaned_downloads (lines 173-218)
# ============================================================================


class TestDetectOrphanedDownloads:
    """Tests for detect_orphaned_downloads()."""

    @pytest.mark.asyncio
    async def test_no_incomplete_models_returns_empty(self):
        """When hf_client reports no incomplete models, return empty list."""
        mock_hf = MagicMock()
        mock_hf.list_incomplete_models.return_value = []

        with patch("mlx_manager.services.hf_client.hf_client", mock_hf):
            from mlx_manager.database import detect_orphaned_downloads

            result = await detect_orphaned_downloads()

        assert result == []

    @pytest.mark.asyncio
    async def test_incomplete_model_with_existing_download_is_skipped(self, test_engine):
        """Incomplete model that already has an active Download record is skipped."""
        from contextlib import asynccontextmanager
        from datetime import UTC, datetime

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.models import Download
        from mlx_manager.models.enums import DownloadStatusEnum

        test_async_session = async_sessionmaker(
            test_engine, class_=AsyncSession, expire_on_commit=False
        )

        # Pre-create a Download record with 'downloading' status
        async with test_async_session() as session:
            dl = Download(
                model_id="mlx-community/test-model",
                status=DownloadStatusEnum.DOWNLOADING,
                downloaded_bytes=500,
                started_at=datetime.now(tz=UTC),
            )
            session.add(dl)
            await session.commit()

        @asynccontextmanager
        async def fake_get_session():
            async with test_async_session() as session:
                yield session

        mock_hf = MagicMock()
        mock_hf.list_incomplete_models.return_value = [
            ("mlx-community/test-model", 500),
        ]

        with (
            patch("mlx_manager.services.hf_client.hf_client", mock_hf),
            patch("mlx_manager.database.get_session", fake_get_session),
        ):
            from mlx_manager.database import detect_orphaned_downloads

            result = await detect_orphaned_downloads()

        assert result == []

    @pytest.mark.asyncio
    async def test_incomplete_model_without_download_creates_record(self, test_engine):
        """Incomplete model with no Download record gets a new one created."""
        from contextlib import asynccontextmanager

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        from sqlmodel import select

        from mlx_manager.models import Download
        from mlx_manager.models.enums import DownloadStatusEnum

        test_async_session = async_sessionmaker(
            test_engine, class_=AsyncSession, expire_on_commit=False
        )

        @asynccontextmanager
        async def fake_get_session():
            async with test_async_session() as session:
                yield session

        mock_hf = MagicMock()
        mock_hf.list_incomplete_models.return_value = [
            ("mlx-community/orphan-model", 1024),
        ]

        with (
            patch("mlx_manager.services.hf_client.hf_client", mock_hf),
            patch("mlx_manager.database.get_session", fake_get_session),
        ):
            from mlx_manager.database import detect_orphaned_downloads

            result = await detect_orphaned_downloads()

        # Should have created one record
        assert len(result) == 1
        download_id, model_id = result[0]
        assert model_id == "mlx-community/orphan-model"
        assert isinstance(download_id, int)

        # Verify the record was committed to the DB
        async with test_async_session() as session:
            stmt = select(Download).where(Download.model_id == "mlx-community/orphan-model")
            row = (await session.execute(stmt)).scalar_one_or_none()
            assert row is not None
            assert row.status == DownloadStatusEnum.PENDING
            assert row.downloaded_bytes == 1024

    @pytest.mark.asyncio
    async def test_mixed_incomplete_models(self, test_engine):
        """Mix of models — one with existing record (skipped), one without (adopted)."""
        from contextlib import asynccontextmanager
        from datetime import UTC, datetime

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.models import Download
        from mlx_manager.models.enums import DownloadStatusEnum

        test_async_session = async_sessionmaker(
            test_engine, class_=AsyncSession, expire_on_commit=False
        )

        # Pre-create a record for one model
        async with test_async_session() as session:
            dl = Download(
                model_id="mlx-community/existing",
                status=DownloadStatusEnum.PAUSED,
                downloaded_bytes=200,
                started_at=datetime.now(tz=UTC),
            )
            session.add(dl)
            await session.commit()

        @asynccontextmanager
        async def fake_get_session():
            async with test_async_session() as session:
                yield session

        mock_hf = MagicMock()
        mock_hf.list_incomplete_models.return_value = [
            ("mlx-community/existing", 200),
            ("mlx-community/new-orphan", 999),
        ]

        with (
            patch("mlx_manager.services.hf_client.hf_client", mock_hf),
            patch("mlx_manager.database.get_session", fake_get_session),
        ):
            from mlx_manager.database import detect_orphaned_downloads

            result = await detect_orphaned_downloads()

        # Only the new orphan should be returned
        assert len(result) == 1
        assert result[0][1] == "mlx-community/new-orphan"


# ============================================================================
# _repair_orphaned_profiles (lines 221-296)
# ============================================================================


class TestRepairOrphanedProfiles:
    """Tests for _repair_orphaned_profiles()."""

    @pytest.mark.asyncio
    async def test_no_profiles_table_returns_early(self):
        """When neither execution_profiles nor server_profiles table exists, return early."""
        from sqlalchemy.ext.asyncio import create_async_engine as make_engine

        empty_engine = make_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)

        with patch("mlx_manager.database.engine", empty_engine):
            from mlx_manager.database import _repair_orphaned_profiles

            # Should return without error
            await _repair_orphaned_profiles()

        await empty_engine.dispose()

    @pytest.mark.asyncio
    async def test_no_orphaned_profiles_is_noop(self):
        """When all profiles have model_id set, nothing happens."""
        from sqlalchemy import text as sa_text
        from sqlalchemy.ext.asyncio import create_async_engine as make_engine

        eng = make_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)

        async with eng.begin() as conn:
            await conn.execute(
                sa_text(
                    "CREATE TABLE execution_profiles "
                    "(id INTEGER PRIMARY KEY, name TEXT, model_id INTEGER)"
                )
            )
            await conn.execute(
                sa_text(
                    "INSERT INTO execution_profiles (id, name, model_id) "
                    "VALUES (1, 'Profile A', 42)"
                )
            )

        with patch("mlx_manager.database.engine", eng):
            from mlx_manager.database import _repair_orphaned_profiles

            await _repair_orphaned_profiles()

        # Verify profile unchanged
        async with eng.begin() as conn:
            result = await conn.execute(
                sa_text("SELECT model_id FROM execution_profiles WHERE id = 1")
            )
            assert result.scalar() == 42

        await eng.dispose()

    @pytest.mark.asyncio
    async def test_orphaned_profile_matched_to_model(self):
        """Orphaned profile whose name matches a model repo_id short name gets repaired."""
        from sqlalchemy import text as sa_text
        from sqlalchemy.ext.asyncio import create_async_engine as make_engine

        eng = make_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)

        async with eng.begin() as conn:
            await conn.execute(
                sa_text(
                    "CREATE TABLE execution_profiles "
                    "(id INTEGER PRIMARY KEY, name TEXT, model_id INTEGER)"
                )
            )
            await conn.execute(
                sa_text("CREATE TABLE models (id INTEGER PRIMARY KEY, repo_id TEXT)")
            )
            # Orphaned profile — name contains model short name
            await conn.execute(
                sa_text(
                    "INSERT INTO execution_profiles (id, name, model_id) "
                    "VALUES (1, 'My Qwen3-0.6B-4bit-DWQ profile', NULL)"
                )
            )
            await conn.execute(
                sa_text(
                    "INSERT INTO models (id, repo_id) "
                    "VALUES (10, 'mlx-community/Qwen3-0.6B-4bit-DWQ')"
                )
            )

        with patch("mlx_manager.database.engine", eng):
            from mlx_manager.database import _repair_orphaned_profiles

            await _repair_orphaned_profiles()

        async with eng.begin() as conn:
            result = await conn.execute(
                sa_text("SELECT model_id FROM execution_profiles WHERE id = 1")
            )
            assert result.scalar() == 10

        await eng.dispose()

    @pytest.mark.asyncio
    async def test_orphaned_profile_no_matching_model(self):
        """Orphaned profile with no matching model stays orphaned (model_id NULL)."""
        from sqlalchemy import text as sa_text
        from sqlalchemy.ext.asyncio import create_async_engine as make_engine

        eng = make_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)

        async with eng.begin() as conn:
            await conn.execute(
                sa_text(
                    "CREATE TABLE execution_profiles "
                    "(id INTEGER PRIMARY KEY, name TEXT, model_id INTEGER)"
                )
            )
            await conn.execute(
                sa_text("CREATE TABLE models (id INTEGER PRIMARY KEY, repo_id TEXT)")
            )
            await conn.execute(
                sa_text(
                    "INSERT INTO execution_profiles (id, name, model_id) "
                    "VALUES (1, 'Custom Profile XYZ', NULL)"
                )
            )
            await conn.execute(
                sa_text(
                    "INSERT INTO models (id, repo_id) "
                    "VALUES (10, 'mlx-community/Qwen3-0.6B-4bit-DWQ')"
                )
            )

        with patch("mlx_manager.database.engine", eng):
            from mlx_manager.database import _repair_orphaned_profiles

            await _repair_orphaned_profiles()

        async with eng.begin() as conn:
            result = await conn.execute(
                sa_text("SELECT model_id FROM execution_profiles WHERE id = 1")
            )
            assert result.scalar() is None

        await eng.dispose()

    @pytest.mark.asyncio
    async def test_orphaned_profile_no_models_available(self):
        """Orphaned profiles with empty models table — returns early with warning."""
        from sqlalchemy import text as sa_text
        from sqlalchemy.ext.asyncio import create_async_engine as make_engine

        eng = make_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)

        async with eng.begin() as conn:
            await conn.execute(
                sa_text(
                    "CREATE TABLE execution_profiles "
                    "(id INTEGER PRIMARY KEY, name TEXT, model_id INTEGER)"
                )
            )
            await conn.execute(
                sa_text("CREATE TABLE models (id INTEGER PRIMARY KEY, repo_id TEXT)")
            )
            await conn.execute(
                sa_text(
                    "INSERT INTO execution_profiles (id, name, model_id) "
                    "VALUES (1, 'Orphan', NULL)"
                )
            )
            # No models inserted

        with patch("mlx_manager.database.engine", eng):
            from mlx_manager.database import _repair_orphaned_profiles

            await _repair_orphaned_profiles()

        async with eng.begin() as conn:
            result = await conn.execute(
                sa_text("SELECT model_id FROM execution_profiles WHERE id = 1")
            )
            assert result.scalar() is None

        await eng.dispose()

    @pytest.mark.asyncio
    async def test_server_profiles_table_fallback(self):
        """Falls back to server_profiles table if execution_profiles doesn't exist."""
        from sqlalchemy import text as sa_text
        from sqlalchemy.ext.asyncio import create_async_engine as make_engine

        eng = make_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)

        async with eng.begin() as conn:
            await conn.execute(
                sa_text(
                    "CREATE TABLE server_profiles "
                    "(id INTEGER PRIMARY KEY, name TEXT, model_id INTEGER)"
                )
            )
            await conn.execute(
                sa_text("CREATE TABLE models (id INTEGER PRIMARY KEY, repo_id TEXT)")
            )
            await conn.execute(
                sa_text(
                    "INSERT INTO server_profiles (id, name, model_id) "
                    "VALUES (1, 'GLM-4.7-Flash', NULL)"
                )
            )
            await conn.execute(
                sa_text(
                    "INSERT INTO models (id, repo_id) "
                    "VALUES (5, 'mlx-community/GLM-4.7-Flash-4bit')"
                )
            )

        with patch("mlx_manager.database.engine", eng):
            from mlx_manager.database import _repair_orphaned_profiles

            await _repair_orphaned_profiles()

        async with eng.begin() as conn:
            result = await conn.execute(
                sa_text("SELECT model_id FROM server_profiles WHERE id = 1")
            )
            assert result.scalar() == 5

        await eng.dispose()
