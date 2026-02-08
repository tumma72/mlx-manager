"""Database setup and session management."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, col

from mlx_manager.config import ensure_data_dir, settings

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


async def migrate_schema() -> None:
    """Add missing columns to existing tables.

    SQLite doesn't support adding columns in CREATE TABLE IF NOT EXISTS,
    so we need to manually add new columns to existing databases.
    """
    # Define columns that may be missing from existing databases
    # Format: (table_name, column_name, column_type, default_value)
    migrations: list[tuple[str, str, str, str | None]] = [
        ("server_profiles", "tool_call_parser", "TEXT", None),
        ("server_profiles", "reasoning_parser", "TEXT", None),
        ("server_profiles", "message_converter", "TEXT", None),
        ("server_profiles", "system_prompt", "TEXT", None),
        # Generation parameters (Phase 15-08: Profile model cleanup)
        ("server_profiles", "temperature", "REAL", "0.7"),
        ("server_profiles", "max_tokens", "INTEGER", "4096"),
        ("server_profiles", "top_p", "REAL", "1.0"),
        # CloudCredential columns for provider configuration (Phase 14 bug fix)
        ("cloud_credentials", "api_type", "TEXT", "'openai'"),
        ("cloud_credentials", "name", "TEXT", "''"),
        # Tool calling settings
        ("server_profiles", "force_tool_injection", "INTEGER", "0"),
        # Model capabilities: type-specific probe fields (probe v2)
        ("model_capabilities", "model_type", "TEXT", None),
        ("model_capabilities", "supports_multi_image", "INTEGER", None),
        ("model_capabilities", "supports_video", "INTEGER", None),
        ("model_capabilities", "embedding_dimensions", "INTEGER", None),
        ("model_capabilities", "max_sequence_length", "INTEGER", None),
        ("model_capabilities", "is_normalized", "INTEGER", None),
        ("model_capabilities", "supports_tts", "INTEGER", None),
        ("model_capabilities", "supports_stt", "INTEGER", None),
        ("model_capabilities", "tool_format", "TEXT", None),
    ]

    # Obsolete columns to drop from server_profiles (unified server approach)
    obsolete_columns = [
        "port",
        "host",
        "max_concurrency",
        "queue_timeout",
        "queue_size",
        "enable_auto_tool_choice",
        "trust_remote_code",
        "chat_template_file",
        "log_level",
        "log_file",
        "no_log_file",
    ]

    async with engine.begin() as conn:
        for table, column, col_type, default in migrations:
            # Check if table and column exist
            result = await conn.execute(text(f"PRAGMA table_info({table})"))
            columns = [row[1] for row in result.fetchall()]

            # Skip if table doesn't exist (no columns returned)
            # Fresh databases will have the column from CREATE TABLE
            if not columns:
                continue

            if column not in columns:
                # Add the column
                default_clause = f" DEFAULT {default}" if default is not None else ""
                sql = f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}"
                logger.info(f"Migrating database: {sql}")
                await conn.execute(text(sql))

        # Drop obsolete columns from server_profiles (SQLite 3.35+ supports DROP COLUMN)
        result = await conn.execute(text("PRAGMA table_info(server_profiles)"))
        existing_columns = [row[1] for row in result.fetchall()]
        for column in obsolete_columns:
            if column in existing_columns:
                sql = f"ALTER TABLE server_profiles DROP COLUMN {column}"
                logger.info(f"Migrating database: {sql}")
                await conn.execute(text(sql))


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
            download.status = "pending"
            download.error = None
            session.add(download)
            if download.id is not None:
                pending_downloads.append((download.id, download.model_id))

        if incomplete:
            await session.commit()
            logger.info(f"Marked {len(incomplete)} downloads for auto-resume")

    return pending_downloads


async def init_db() -> None:
    """Initialize the database and create tables."""
    ensure_data_dir()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Run schema migrations for existing databases
    await migrate_schema()

    # Recover any incomplete downloads from previous runs
    await recover_incomplete_downloads()

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
