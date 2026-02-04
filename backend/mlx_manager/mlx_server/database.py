"""Database setup for MLX Server audit logging."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from mlx_manager.mlx_server.config import get_settings

# Lazily initialized engine and session factory
_engine = None
_async_session = None


def _get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_path = Path(settings.database_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
            future=True,
        )
    return _engine


def _get_session_factory():
    """Get or create the session factory."""
    global _async_session
    if _async_session is None:
        _async_session = async_sessionmaker(
            _get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session


async def init_db() -> None:
    """Initialize the database and create tables."""
    # Import models to register them
    from mlx_manager.mlx_server.models.audit import AuditLog  # noqa: F401

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("MLX Server database initialized")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    session_factory = _get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def cleanup_old_logs() -> int:
    """Delete audit logs older than retention period.

    Returns:
        Number of deleted records.
    """
    from mlx_manager.mlx_server.models.audit import AuditLog

    settings = get_settings()
    cutoff = datetime.now(UTC) - timedelta(days=settings.audit_retention_days)

    async with get_session() as session:
        result = await session.execute(delete(AuditLog).where(AuditLog.timestamp < cutoff))
        deleted = result.rowcount
        if deleted > 0:
            logger.info(
                f"Cleaned up {deleted} audit logs older than {settings.audit_retention_days} days"
            )
        return deleted


def reset_for_testing() -> None:
    """Reset database state for testing."""
    global _engine, _async_session
    _engine = None
    _async_session = None
