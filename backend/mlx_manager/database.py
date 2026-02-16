"""Database setup and session management."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import col

from mlx_manager.config import ensure_data_dir, settings
from mlx_manager.models.enums import DownloadStatusEnum

# SQLAlchemy echo is controlled by a dedicated env var, NOT by settings.debug.
# Debug mode enables application-level debug logging via Loguru, but SQLAlchemy's
# echo produces low-level SQL traces that flood the console and are rarely useful.
# Use MLX_MANAGER_ECHO_SQL=true explicitly when you need raw SQL output.
_echo_sql = os.environ.get("MLX_MANAGER_ECHO_SQL", "").lower() in ("true", "1", "yes")

# Create async engine
engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.database_path}",
    echo=_echo_sql,
    future=True,
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Stamp revision: the last migration before the catch-up migration.
# Pre-Alembic databases (created by create_all) are stamped here so that
# only the catch-up migration (e1a2b3c4d5f6) runs on first upgrade.
_PRE_ALEMBIC_STAMP = "d8e3f5a7b912"


def _run_upgrade(connection, alembic_cfg) -> None:
    """Run alembic upgrade head using the provided synchronous connection."""
    from alembic import command

    alembic_cfg.attributes["connection"] = connection
    command.upgrade(alembic_cfg, "head")


def _stamp_revision(connection, alembic_cfg, revision: str) -> None:
    """Stamp a revision without running migrations."""
    from alembic import command

    alembic_cfg.attributes["connection"] = connection
    command.stamp(alembic_cfg, revision)


async def run_alembic_upgrade() -> None:
    """Run Alembic migrations programmatically.

    Handles three scenarios:
    1. Fresh install (no DB): runs full migration chain from scratch.
    2. Pre-Alembic DB (has tables but no alembic_version): stamps to
       the last known revision, then upgrades to head.
    3. Existing Alembic DB: upgrades from current revision to head.
    """
    from alembic.config import Config

    alembic_ini = Path(__file__).parent.parent / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini))

    async with engine.begin() as conn:
        # Check if this is a pre-Alembic database
        has_alembic = await conn.run_sync(_check_alembic_version_table)
        has_tables = await conn.run_sync(_check_has_existing_tables)

        if has_tables and not has_alembic:
            # Pre-Alembic database: stamp so only the catch-up migration runs
            logger.info(
                f"Pre-Alembic database detected, stamping to {_PRE_ALEMBIC_STAMP}"
            )
            await conn.run_sync(_stamp_revision, alembic_cfg, _PRE_ALEMBIC_STAMP)

        # Run upgrade to head
        await conn.run_sync(_run_upgrade, alembic_cfg)
        logger.info("Database schema is up to date")


def _check_alembic_version_table(connection) -> bool:
    """Check if alembic_version table exists."""
    result = connection.execute(
        text(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='alembic_version'"
        )
    )
    return result.fetchone() is not None


def _check_has_existing_tables(connection) -> bool:
    """Check if the database has any application tables (not just alembic_version)."""
    result = connection.execute(
        text(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "AND name != 'alembic_version' LIMIT 1"
        )
    )
    return result.fetchone() is not None


async def recover_incomplete_downloads() -> list[tuple[int, str]]:
    """Resume any incomplete downloads on startup.

    HuggingFace Hub's snapshot_download automatically resumes partial downloads,
    so we mark actively-downloading ones as pending for retry.

    Paused downloads stay paused - they were intentionally paused by the user
    and should only resume when the user clicks Resume.

    Returns:
        List of (download_id, model_id) tuples for downloads that need auto-resuming.
    """
    pending_downloads: list[tuple[int, str]] = []

    async with get_session() as session:
        from sqlmodel import select

        from mlx_manager.models import Download

        result = await session.execute(
            select(Download).where(col(Download.status).in_(["pending", "downloading"]))
        )
        incomplete = result.scalars().all()

        for download in incomplete:
            # Mark as pending for retry - HF Hub will resume partial downloads
            download.status = DownloadStatusEnum.PENDING
            download.error = None
            session.add(download)
            if download.id is not None:
                pending_downloads.append((download.id, download.model_id))

        if incomplete:
            await session.commit()
            logger.info(f"Marked {len(incomplete)} downloads for auto-resume")

    return pending_downloads


async def _repair_orphaned_profiles() -> None:
    """Fix profiles with NULL model_id after migration from model_path.

    Tries to match orphaned profiles to existing models by checking if
    the profile name contains a model repo_id pattern. If no match is
    found, the profile is left with model_id=NULL (will show as
    'Unknown model' in the UI and can be reassigned manually).
    """
    async with engine.begin() as conn:
        # Check which profile table exists
        result = await conn.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('execution_profiles', 'server_profiles')"
            )
        )
        tables = {row[0] for row in result.fetchall()}

        if "execution_profiles" in tables:
            table_name = "execution_profiles"
        elif "server_profiles" in tables:
            table_name = "server_profiles"
        else:
            return  # No profiles table exists yet (fresh install)

        # Find profiles missing model_id
        sql_orphans = f"SELECT id, name FROM {table_name} WHERE model_id IS NULL"
        rows = await conn.execute(text(sql_orphans))
        orphans = rows.fetchall()
        if not orphans:
            return

        # Get all available models
        model_rows = await conn.execute(text("SELECT id, repo_id FROM models"))
        models = model_rows.fetchall()
        if not models:
            logger.warning(
                f"{len(orphans)} profile(s) have no model assigned and no models "
                "are available. Edit these profiles to assign a model."
            )
            return

        repaired = 0
        for profile_id, profile_name in orphans:
            # Try to match: check if any model repo_id appears in the profile name
            # or if profile name appears in a model repo_id (case-insensitive)
            matched_model_id = None
            name_lower = profile_name.lower()
            for model_id, repo_id in models:
                # Extract short name from repo_id (e.g., "Qwen3-0.6B-4bit-DWQ" from
                # "mlx-community/Qwen3-0.6B-4bit-DWQ")
                short_name = repo_id.split("/")[-1].lower()
                if short_name in name_lower or name_lower in short_name:
                    matched_model_id = model_id
                    break

            if matched_model_id:
                await conn.execute(
                    text(f"UPDATE {table_name} SET model_id = :model_id WHERE id = :pid"),
                    {"model_id": matched_model_id, "pid": profile_id},
                )
                repaired += 1
                logger.info(
                    f"Repaired profile '{profile_name}' (id={profile_id}) "
                    f"→ model_id={matched_model_id}"
                )

        if repaired:
            logger.info(f"Repaired {repaired}/{len(orphans)} orphaned profiles")
        remaining = len(orphans) - repaired
        if remaining:
            logger.warning(
                f"{remaining} profile(s) still have no model assigned. "
                "Edit these profiles in the UI to assign a model."
            )


async def init_db() -> None:
    """Initialize the database and run migrations."""
    ensure_data_dir()

    # Single schema management system: Alembic handles both fresh installs
    # and upgrades. No more create_all() or hand-rolled ALTER TABLE statements.
    await run_alembic_upgrade()

    # Recover any incomplete downloads from previous runs
    await recover_incomplete_downloads()

    # Sync models from HuggingFace cache into models table
    from mlx_manager.services.model_registry import sync_models_from_cache

    await sync_models_from_cache()

    # Fix orphaned profiles (model_id is NULL after migration from model_path)
    await _repair_orphaned_profiles()

    # Insert default settings if not present
    async with get_session() as session:
        from sqlmodel import select

        from mlx_manager.models import Setting

        # Check if settings exist
        result = await session.execute(select(Setting))
        if not result.scalars().first():
            default_settings = [
                Setting(key="default_port_start", value="10240"),
                Setting(key="huggingface_cache_path", value=str(settings.hf_cache_path)),
                Setting(key="max_memory_percent", value="80"),
                Setting(key="health_check_interval", value="30"),
            ]
            for s in default_settings:
                session.add(s)
            await session.commit()


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            # Only log actual database errors, not HTTP exceptions from endpoints
            from fastapi import HTTPException

            if not isinstance(e, HTTPException):
                logger.exception(f"Database session error, rolling back: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting a database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            # Only log actual database errors, not HTTP exceptions from endpoints
            from fastapi import HTTPException

            if not isinstance(e, HTTPException):
                logger.exception(f"Database session error, rolling back: {e}")
            await session.rollback()
            raise
