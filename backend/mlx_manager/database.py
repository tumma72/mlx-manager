"""Database setup and session management."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from mlx_manager.config import ensure_data_dir, settings

logger = logging.getLogger(__name__)

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
    ]

    async with engine.begin() as conn:
        for table, column, col_type, default in migrations:
            # Check if column exists
            result = await conn.execute(text(f"PRAGMA table_info({table})"))
            columns = [row[1] for row in result.fetchall()]

            if column not in columns:
                # Add the column
                default_clause = f" DEFAULT {default}" if default is not None else ""
                sql = f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}"
                logger.info(f"Migrating database: {sql}")
                await conn.execute(text(sql))


async def init_db() -> None:
    """Initialize the database and create tables."""
    ensure_data_dir()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Run schema migrations for existing databases
    await migrate_schema()

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
        except Exception:
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
        except Exception:
            await session.rollback()
            raise
