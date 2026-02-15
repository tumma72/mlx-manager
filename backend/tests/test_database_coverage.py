"""Coverage tests for database.py focusing on uncovered migration paths.

Tests database migration scenarios with REAL async SQLite databases (no mocks).
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


@pytest.fixture
async def migration_engine():
    """Create a fresh in-memory async engine for migration tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    yield engine
    await engine.dispose()


@pytest.fixture
async def migration_session(migration_engine):
    """Create a session factory for migration tests."""
    session_maker = async_sessionmaker(
        migration_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_maker() as session:
        yield session


class TestMigrationColumnAddition:
    """Test migrate_schema adding columns with defaults."""

    @pytest.mark.asyncio
    async def test_migrate_adds_column_with_default(self, migration_engine):
        """Test adding a column with a DEFAULT value."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        # Create a minimal server_profiles table WITHOUT the new columns
        async with migration_engine.begin() as conn:
            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
                """
                )
            )

        # Run migration with patched engine
        with patch("mlx_manager.database.engine", migration_engine):
            await migrate_schema()

        # Verify columns were added with defaults
        async with migration_engine.begin() as conn:
            result = await conn.execute(text("PRAGMA table_info(server_profiles)"))
            columns = {row[1]: row[2] for row in result.fetchall()}

            # Check that new columns exist
            assert "temperature" in columns
            assert "max_tokens" in columns
            assert "top_p" in columns
            assert columns["temperature"] == "REAL"
            assert columns["max_tokens"] == "INTEGER"

    @pytest.mark.asyncio
    async def test_migrate_adds_column_without_default(self, migration_engine):
        """Test adding a column with NULL default."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        # Create minimal table
        async with migration_engine.begin() as conn:
            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await migrate_schema()

        # Verify nullable columns were added
        async with migration_engine.begin() as conn:
            result = await conn.execute(text("PRAGMA table_info(server_profiles)"))
            columns = {row[1] for row in result.fetchall()}

            assert "tool_call_parser" in columns
            assert "system_prompt" in columns
            assert "model_id" in columns

    @pytest.mark.asyncio
    async def test_migrate_skips_nonexistent_table(self, migration_engine):
        """Test migration skips tables that don't exist yet."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        # Don't create any tables - fresh database
        with patch("mlx_manager.database.engine", migration_engine):
            # Should not raise even though tables don't exist
            await migrate_schema()


class TestModelPathMigration:
    """Test model_path → model_id migration logic (lines 108-153)."""

    @pytest.mark.asyncio
    async def test_migrate_model_path_with_existing_model(self, migration_engine):
        """Test migrating profile when model already exists in models table."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        # Setup: create tables with old schema
        async with migration_engine.begin() as conn:
            # Create models table with existing model
            await conn.execute(
                text(
                    """
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    model_type TEXT NOT NULL
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO models (repo_id, model_type)
                VALUES ('mlx-community/Qwen3-0.6B-4bit', 'text-gen')
                """
                )
            )

            # Create profiles table with old model_path column
            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_path TEXT,
                    model_type TEXT,
                    model_id INTEGER
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO server_profiles (name, model_path, model_type, model_id)
                VALUES ('Test Profile', 'mlx-community/Qwen3-0.6B-4bit', 'lm', NULL)
                """
                )
            )

        # Run migration
        with patch("mlx_manager.database.engine", migration_engine):
            await migrate_schema()

        # Verify profile was linked to existing model
        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT model_id FROM server_profiles WHERE name = 'Test Profile'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == 1  # Should be linked to the existing model

    @pytest.mark.asyncio
    async def test_migrate_model_path_creates_new_model(self, migration_engine):
        """Test migrating profile creates new model if it doesn't exist."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        async with migration_engine.begin() as conn:
            # Create models table (empty)
            await conn.execute(
                text(
                    """
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    model_type TEXT NOT NULL
                )
                """
                )
            )

            # Create profile with model_path but no matching model
            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_path TEXT,
                    model_type TEXT,
                    model_id INTEGER
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO server_profiles (name, model_path, model_type, model_id)
                VALUES ('New Profile', 'mlx-community/NewModel', 'lm', NULL)
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await migrate_schema()

        # Verify new model was created
        async with migration_engine.begin() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM models"))
            count = result.fetchone()[0]
            assert count == 1

            result = await conn.execute(text("SELECT repo_id, model_type FROM models"))
            row = result.fetchone()
            assert row[0] == "mlx-community/NewModel"
            assert row[1] == "text-gen"  # Mapped from 'lm'

    @pytest.mark.asyncio
    async def test_migrate_model_type_mapping(self, migration_engine):
        """Test old model_type values are mapped correctly."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        async with migration_engine.begin() as conn:
            await conn.execute(
                text(
                    """
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    model_type TEXT NOT NULL
                )
                """
                )
            )

            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_path TEXT,
                    model_type TEXT,
                    model_id INTEGER
                )
                """
                )
            )

            # Insert profiles with different old model_type values
            await conn.execute(
                text(
                    """
                INSERT INTO server_profiles (name, model_path, model_type, model_id)
                VALUES
                    ('Vision Profile', 'mlx-community/vision-model', 'multimodal', NULL),
                    ('Audio Profile', 'mlx-community/audio-model', 'audio', NULL),
                    ('Embed Profile', 'mlx-community/embed-model', 'embeddings', NULL)
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await migrate_schema()

        # Verify type mapping
        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT repo_id, model_type FROM models ORDER BY repo_id")
            )
            models = result.fetchall()

            model_types = {model[1] for model in models}
            assert "vision" in model_types  # multimodal → vision
            assert "audio" in model_types  # audio → audio
            assert "embeddings" in model_types  # embeddings → embeddings


class TestObsoleteColumnDrop:
    """Test dropping obsolete columns (lines 162-164)."""

    @pytest.mark.asyncio
    async def test_drop_obsolete_columns(self, migration_engine):
        """Test obsolete columns are dropped from server_profiles."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        async with migration_engine.begin() as conn:
            # Create table with obsolete columns
            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    port INTEGER,
                    host TEXT,
                    max_concurrency INTEGER,
                    log_level TEXT
                )
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await migrate_schema()

        # Verify obsolete columns were dropped
        async with migration_engine.begin() as conn:
            result = await conn.execute(text("PRAGMA table_info(server_profiles)"))
            columns = {row[1] for row in result.fetchall()}

            assert "port" not in columns
            assert "host" not in columns
            assert "max_concurrency" not in columns
            assert "log_level" not in columns
            # Original columns should remain
            assert "id" in columns
            assert "name" in columns


class TestLegacyTableDrop:
    """Test dropping legacy tables (lines 172-174)."""

    @pytest.mark.asyncio
    async def test_drop_legacy_downloaded_models_table(self, migration_engine):
        """Test legacy downloaded_models table is dropped."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        async with migration_engine.begin() as conn:
            # Create legacy table
            await conn.execute(
                text(
                    """
                CREATE TABLE downloaded_models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL
                )
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await migrate_schema()

        # Verify table was dropped
        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='downloaded_models'
                """
                )
            )
            assert result.fetchone() is None


class TestModelCapabilitiesMigration:
    """Test dropping old-format model_capabilities table (lines 180-181)."""

    @pytest.mark.asyncio
    async def test_drop_old_model_capabilities_without_capability_type(self, migration_engine):
        """Test old model_capabilities table (missing capability_type) is dropped."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        async with migration_engine.begin() as conn:
            # Create old-format table WITHOUT capability_type column
            await conn.execute(
                text(
                    """
                CREATE TABLE model_capabilities (
                    id INTEGER PRIMARY KEY,
                    model_id INTEGER NOT NULL,
                    supports_tools INTEGER,
                    supports_thinking INTEGER
                )
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await migrate_schema()

        # Verify old table was dropped
        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='model_capabilities'
                """
                )
            )
            assert result.fetchone() is None

    @pytest.mark.asyncio
    async def test_keep_new_model_capabilities_with_capability_type(self, migration_engine):
        """Test new model_capabilities table (with capability_type) is kept."""
        from unittest.mock import patch

        from mlx_manager.database import migrate_schema

        async with migration_engine.begin() as conn:
            # Create new-format table WITH capability_type column
            await conn.execute(
                text(
                    """
                CREATE TABLE model_capabilities (
                    id INTEGER PRIMARY KEY,
                    model_id INTEGER NOT NULL,
                    capability_type TEXT NOT NULL,
                    supports_tools INTEGER,
                    supports_thinking INTEGER
                )
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await migrate_schema()

        # Verify new table was NOT dropped
        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='model_capabilities'
                """
                )
            )
            assert result.fetchone() is not None


class TestRepairOrphanedProfiles:
    """Test _repair_orphaned_profiles function (lines 258-296)."""

    @pytest.mark.asyncio
    async def test_repair_orphaned_profile_with_name_match(self, migration_engine):
        """Test repairing profile by matching name to model repo_id."""
        from unittest.mock import patch

        from mlx_manager.database import _repair_orphaned_profiles

        async with migration_engine.begin() as conn:
            # Create models table with a model
            await conn.execute(
                text(
                    """
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    model_type TEXT NOT NULL
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO models (repo_id, model_type)
                VALUES ('mlx-community/Qwen3-0.6B-4bit-DWQ', 'text-gen')
                """
                )
            )

            # Create profile with NULL model_id but name matches model
            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_id INTEGER
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO server_profiles (name, model_id)
                VALUES ('My Qwen3-0.6B-4bit-DWQ Profile', NULL)
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await _repair_orphaned_profiles()

        # Verify profile was repaired
        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT model_id FROM server_profiles WHERE name LIKE '%Qwen3%'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == 1  # Linked to model

    @pytest.mark.asyncio
    async def test_repair_no_profiles_table_yet(self, migration_engine):
        """Test repair handles missing profiles table (fresh install)."""
        from unittest.mock import patch

        from mlx_manager.database import _repair_orphaned_profiles

        # Don't create any tables - fresh database
        with patch("mlx_manager.database.engine", migration_engine):
            # Should not raise
            await _repair_orphaned_profiles()

    @pytest.mark.asyncio
    async def test_repair_with_execution_profiles_table(self, migration_engine):
        """Test repair works with renamed execution_profiles table."""
        from unittest.mock import patch

        from mlx_manager.database import _repair_orphaned_profiles

        async with migration_engine.begin() as conn:
            # Create models
            await conn.execute(
                text(
                    """
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    model_type TEXT NOT NULL
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO models (repo_id, model_type)
                VALUES ('mlx-community/test-model', 'text-gen')
                """
                )
            )

            # Use new table name
            await conn.execute(
                text(
                    """
                CREATE TABLE execution_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_id INTEGER
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO execution_profiles (name, model_id)
                VALUES ('test-model profile', NULL)
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await _repair_orphaned_profiles()

        # Verify repair worked with new table name
        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT model_id FROM execution_profiles WHERE name = 'test-model profile'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_repair_orphans_no_models_available(self, migration_engine):
        """Test repair when orphaned profiles exist but no models available."""
        from unittest.mock import patch

        from mlx_manager.database import _repair_orphaned_profiles

        async with migration_engine.begin() as conn:
            # Create empty models table
            await conn.execute(
                text(
                    """
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    model_type TEXT NOT NULL
                )
                """
                )
            )

            # Create orphaned profile
            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_id INTEGER
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO server_profiles (name, model_id)
                VALUES ('Orphaned Profile', NULL)
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            # Should not raise, just log warning
            await _repair_orphaned_profiles()

        # Profile should remain orphaned
        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT model_id FROM server_profiles WHERE name = 'Orphaned Profile'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] is None

    @pytest.mark.asyncio
    async def test_repair_orphan_no_match_found(self, migration_engine):
        """Test repair when orphan exists but no name match found."""
        from unittest.mock import patch

        from mlx_manager.database import _repair_orphaned_profiles

        async with migration_engine.begin() as conn:
            await conn.execute(
                text(
                    """
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    model_type TEXT NOT NULL
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO models (repo_id, model_type)
                VALUES ('mlx-community/completely-different-model', 'text-gen')
                """
                )
            )

            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_id INTEGER
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO server_profiles (name, model_id)
                VALUES ('Unrelated Name', NULL)
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await _repair_orphaned_profiles()

        # Profile should remain orphaned (no match)
        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT model_id FROM server_profiles WHERE name = 'Unrelated Name'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] is None

    @pytest.mark.asyncio
    async def test_repair_no_orphans_present(self, migration_engine):
        """Test repair when all profiles have valid model_id."""
        from unittest.mock import patch

        from mlx_manager.database import _repair_orphaned_profiles

        async with migration_engine.begin() as conn:
            await conn.execute(
                text(
                    """
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    model_type TEXT NOT NULL
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO models (repo_id, model_type)
                VALUES ('mlx-community/model', 'text-gen')
                """
                )
            )

            await conn.execute(
                text(
                    """
                CREATE TABLE server_profiles (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_id INTEGER
                )
                """
                )
            )
            await conn.execute(
                text(
                    """
                INSERT INTO server_profiles (name, model_id)
                VALUES ('Valid Profile', 1)
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            # Should complete quickly with no repairs needed
            await _repair_orphaned_profiles()


class TestSessionErrorHandling:
    """Test error handling paths in get_session and get_db."""

    @pytest.mark.asyncio
    async def test_get_session_http_exception_no_log(self, test_engine):
        """Test get_session doesn't log HTTPException errors."""
        from unittest.mock import patch

        from fastapi import HTTPException
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_session

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Mock logger to verify it's NOT called for HTTPException
        with patch("mlx_manager.database.logger") as mock_logger:
            with patch("mlx_manager.database.async_session", test_async_session):
                try:
                    async with get_session():
                        raise HTTPException(status_code=404, detail="Not found")
                except HTTPException:
                    pass

                # Logger.exception should NOT have been called
                mock_logger.exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_session_database_error_logs(self, test_engine):
        """Test get_session logs actual database errors."""
        from unittest.mock import patch

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_session

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        with patch("mlx_manager.database.logger") as mock_logger:
            with patch("mlx_manager.database.async_session", test_async_session):
                try:
                    async with get_session():
                        raise ValueError("Database error")
                except ValueError:
                    pass

                # Logger.exception SHOULD have been called
                mock_logger.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_db_http_exception_no_log(self, test_engine):
        """Test get_db doesn't log HTTPException errors."""
        from unittest.mock import patch

        from fastapi import HTTPException
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_db

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        with patch("mlx_manager.database.logger") as mock_logger:
            with patch("mlx_manager.database.async_session", test_async_session):
                gen = get_db()
                await gen.__anext__()

                try:
                    await gen.athrow(HTTPException(status_code=400, detail="Bad request"))
                except HTTPException:
                    pass

                # Logger.exception should NOT have been called
                mock_logger.exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_db_database_error_logs(self, test_engine):
        """Test get_db logs actual database errors."""
        from unittest.mock import patch

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from mlx_manager.database import get_db

        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        with patch("mlx_manager.database.logger") as mock_logger:
            with patch("mlx_manager.database.async_session", test_async_session):
                gen = get_db()
                await gen.__anext__()

                try:
                    await gen.athrow(RuntimeError("Database failure"))
                except RuntimeError:
                    pass

                # Logger.exception SHOULD have been called
                mock_logger.exception.assert_called_once()
