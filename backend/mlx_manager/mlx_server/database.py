"""Database setup for MLX Server audit logging."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from loguru import logger
from sqlalchemy import CursorResult, delete, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, col

from mlx_manager.mlx_server.config import get_settings

# Lazily initialized engine and session factory
_engine = None
_async_session = None

# Shared engine injected from the host application in embedded mode.
# When set, _get_engine() and _get_session_factory() return these instead of
# creating their own, so both components share a single connection pool.
_shared_engine = None
_shared_session_factory = None


def set_shared_engine(engine, session_factory=None) -> None:
    """Inject a shared engine for embedded mode.

    When MLX Server runs embedded inside MLX Manager, both components point at
    the same SQLite file.  By sharing the *engine* they also share the
    connection pool, which avoids duplicate connections and WAL-mode conflicts.

    Args:
        engine: An AsyncEngine instance created by the host application.
        session_factory: Optional async_sessionmaker bound to *engine*.  If
            omitted a new sessionmaker is created from the provided engine.
    """
    global _shared_engine, _shared_session_factory, _engine, _async_session
    _shared_engine = engine
    _engine = engine
    if session_factory is not None:
        _shared_session_factory = session_factory
        _async_session = session_factory
    else:
        # Build a compatible session factory from the provided engine
        _shared_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        _async_session = _shared_session_factory


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
    """Initialize the database and create tables.

    Creates only the MLX Server's own tables (audit_logs).  Passing the
    explicit ``tables`` list to ``create_all`` ensures we never touch tables
    that belong to the host application when running in embedded mode.
    """
    # Import so the model registers its table with SQLModel.metadata
    from mlx_manager.mlx_server.models.audit import AuditLog  # noqa: F401

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[AuditLog.__table__],  # type: ignore[attr-defined]
        )
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
    """Delete audit logs older than retention period, then run size-based cleanup.

    Returns:
        Total number of deleted records (time-based + size-based).
    """
    from mlx_manager.mlx_server.models.audit import AuditLog

    settings = get_settings()
    cutoff = datetime.now(UTC) - timedelta(days=settings.audit_retention_days)

    async with get_session() as session:
        # DML statements return CursorResult which has rowcount
        cursor = cast(
            CursorResult,
            await session.execute(delete(AuditLog).where(col(AuditLog.timestamp) < cutoff)),
        )
        deleted = cursor.rowcount or 0
        if deleted > 0:
            logger.info(
                f"Cleaned up {deleted} audit logs older than {settings.audit_retention_days} days"
            )

    # Run size-based cleanup after time-based cleanup
    size_deleted = await cleanup_by_size()
    return deleted + size_deleted


async def cleanup_by_size() -> int:
    """Delete oldest audit log records until the DB file is under the configured size limit.

    Checks the DB file size against ``audit_max_mb``. If over the limit, deletes
    records in batches of 1000 (oldest first) until the size is within bounds, then
    runs VACUUM to reclaim disk space.

    Returns:
        Total number of deleted records. 0 if already under the size limit.
    """
    from mlx_manager.mlx_server.models.audit import AuditLog

    settings = get_settings()
    db_path = settings.get_database_path()
    limit_bytes = settings.audit_max_mb * 1024 * 1024

    # If the database file does not exist yet, nothing to do
    if not db_path.exists():
        return 0

    current_size = os.path.getsize(db_path)
    if current_size <= limit_bytes:
        return 0

    logger.info(
        f"Audit DB size {current_size / 1024 / 1024:.1f} MB exceeds limit "
        f"{settings.audit_max_mb} MB — purging oldest records"
    )

    total_deleted = 0
    batch_size = 1000

    while current_size > limit_bytes:
        async with get_session() as session:
            # Find oldest batch of records by timestamp
            from sqlmodel import select

            subq_result = await session.execute(
                select(AuditLog.id).order_by(col(AuditLog.timestamp).asc()).limit(batch_size)
            )
            ids_to_delete = [row[0] for row in subq_result.all()]

            if not ids_to_delete:
                # No more records to delete
                break

            cursor = cast(
                CursorResult,
                await session.execute(delete(AuditLog).where(col(AuditLog.id).in_(ids_to_delete))),
            )
            batch_deleted = cursor.rowcount or 0
            total_deleted += batch_deleted

        if batch_deleted == 0:
            break

        # Re-check size after each batch
        current_size = os.path.getsize(db_path) if db_path.exists() else 0

    if total_deleted > 0:
        # VACUUM to reclaim freed disk space — only run after actual deletes
        async with get_session() as session:
            await session.execute(text("VACUUM"))
        logger.info(
            f"Size-based cleanup purged {total_deleted} audit records; "
            f"DB size now {os.path.getsize(db_path) / 1024 / 1024:.1f} MB"
            if db_path.exists()
            else f"Size-based cleanup purged {total_deleted} audit records"
        )

    return total_deleted


def reset_for_testing() -> None:
    """Reset database state for testing."""
    global _engine, _async_session, _shared_engine, _shared_session_factory
    _engine = None
    _async_session = None
    _shared_engine = None
    _shared_session_factory = None
