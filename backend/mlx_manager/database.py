"""Database setup and session management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from mlx_manager.config import ensure_data_dir, settings

# Create async engine
engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.database_path}",
    echo=settings.debug,
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
    ]
    # NOTE: Obsolete columns in server_profiles are NOT dropped - SQLite limitations
    # and backward compatibility. They will be ignored by the new model:
    # port, host, max_concurrency, queue_timeout, queue_size, tool_call_parser,
    # reasoning_parser, message_converter, enable_auto_tool_choice,
    # trust_remote_code, chat_template_file, log_level, log_file, no_log_file

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


async def recover_incomplete_downloads() -> list[tuple[int, str]]:
    """Resume any incomplete downloads on startup.

    HuggingFace Hub's snapshot_download automatically resumes partial downloads,
    so we mark them as pending for retry.

    Returns:
        List of (download_id, model_id) tuples for downloads that need resuming.
    """
    pending_downloads: list[tuple[int, str]] = []

    async with get_session() as session:
        from sqlalchemy import or_
        from sqlmodel import select

        from mlx_manager.models import Download

        result = await session.execute(
            select(Download).where(
                or_(Download.status == "pending", Download.status == "downloading")  # type: ignore[arg-type]
            )
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
            logger.info(f"Marked {len(incomplete)} downloads for resume")

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
