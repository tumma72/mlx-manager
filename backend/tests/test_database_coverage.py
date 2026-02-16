"""Coverage tests for database.py focusing on uncovered paths.

Tests database scenarios with REAL async SQLite databases (no mocks).
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


class TestRepairOrphanedProfiles:
    """Test _repair_orphaned_profiles function."""

    @pytest.mark.asyncio
    async def test_repair_orphaned_profile_with_name_match(self, migration_engine):
        """Test repairing profile by matching name to model repo_id."""
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
                VALUES ('mlx-community/Qwen3-0.6B-4bit-DWQ', 'text-gen')
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
                VALUES ('My Qwen3-0.6B-4bit-DWQ Profile', NULL)
                """
                )
            )

        with patch("mlx_manager.database.engine", migration_engine):
            await _repair_orphaned_profiles()

        async with migration_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT model_id FROM server_profiles WHERE name LIKE '%Qwen3%'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_repair_no_profiles_table_yet(self, migration_engine):
        """Test repair handles missing profiles table (fresh install)."""
        from unittest.mock import patch

        from mlx_manager.database import _repair_orphaned_profiles

        with patch("mlx_manager.database.engine", migration_engine):
            await _repair_orphaned_profiles()

    @pytest.mark.asyncio
    async def test_repair_with_execution_profiles_table(self, migration_engine):
        """Test repair works with renamed execution_profiles table."""
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
                VALUES ('mlx-community/test-model', 'text-gen')
                """
                )
            )

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
            await _repair_orphaned_profiles()

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

        with patch("mlx_manager.database.logger") as mock_logger:
            with patch("mlx_manager.database.async_session", test_async_session):
                try:
                    async with get_session():
                        raise HTTPException(status_code=404, detail="Not found")
                except HTTPException:
                    pass

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

                mock_logger.exception.assert_called_once()
