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
        # Audio params on server_profiles
        ("server_profiles", "tts_default_voice", "TEXT", None),
        ("server_profiles", "tts_default_speed", "REAL", None),
        ("server_profiles", "tts_sample_rate", "INTEGER", None),
        ("server_profiles", "stt_default_language", "TEXT", None),
        # FK to models table
        ("server_profiles", "model_id", "INTEGER", None),
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
        "model_path",
        "model_type",
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

        # Migrate model_path → model_id for existing profiles
        result = await conn.execute(text("PRAGMA table_info(server_profiles)"))
        sp_columns = [row[1] for row in result.fetchall()]

        if "model_path" in sp_columns and "model_id" in sp_columns:
            # Read profiles that need migration (have model_path but no model_id)
            rows = await conn.execute(
                text(
                    "SELECT id, model_path, model_type FROM server_profiles "
                    "WHERE model_path IS NOT NULL AND (model_id IS NULL OR model_id = 0)"
                )
            )
            profiles_to_migrate = rows.fetchall()

            for profile_id, model_path, model_type in profiles_to_migrate:
                # Find existing Model record by repo_id
                model_row = await conn.execute(
                    text("SELECT id FROM models WHERE repo_id = :repo_id"),
                    {"repo_id": model_path},
                )
                model = model_row.fetchone()

                if model:
                    model_id = model[0]
                else:
                    # Create a Model record for this profile's model
                    # Map old model_type to new: "lm" -> "text-gen", "multimodal" -> "vision"
                    mapped_type = "text-gen"
                    if model_type == "multimodal":
                        mapped_type = "vision"
                    elif model_type in ("text-gen", "vision", "embeddings", "audio"):
                        mapped_type = model_type

                    await conn.execute(
                        text(
                            "INSERT INTO models (repo_id, model_type) "
                            "VALUES (:repo_id, :model_type)"
                        ),
                        {"repo_id": model_path, "model_type": mapped_type},
                    )

                    # Get the ID of the just-inserted model
                    id_row = await conn.execute(text("SELECT last_insert_rowid()"))
                    row = id_row.fetchone()
                    model_id = row[0] if row else None

                # Update the profile's model_id
                await conn.execute(
                    text("UPDATE server_profiles SET model_id = :model_id WHERE id = :profile_id"),
                    {"model_id": model_id, "profile_id": profile_id},
                )
                logger.info(
                    f"Migrated profile {profile_id}: model_path={model_path} → model_id={model_id}"
                )

        # Drop obsolete columns from server_profiles (SQLite 3.35+ supports DROP COLUMN)
        result = await conn.execute(text("PRAGMA table_info(server_profiles)"))
        existing_columns = [row[1] for row in result.fetchall()]
        for column in obsolete_columns:
            if column in existing_columns:
                sql = f"ALTER TABLE server_profiles DROP COLUMN {column}"
                logger.info(f"Migrating database: {sql}")
                await conn.execute(text(sql))

        # Drop legacy tables replaced by unified 'models' table
        # Note: model_capabilities is now a real JTI table — only drop the
        # old legacy table if it has the wrong schema (no capability_type column).
        for table in ("downloaded_models",):
            result = await conn.execute(text(f"PRAGMA table_info({table})"))
            if result.fetchall():  # Table exists
                sql = f"DROP TABLE {table}"
                logger.info(f"Migrating database: {sql}")
                await conn.execute(text(sql))

        # Drop old-format model_capabilities table (lacks capability_type column)
        result = await conn.execute(text("PRAGMA table_info(model_capabilities)"))
        mc_columns = {row[1] for row in result.fetchall()}
        if mc_columns and "capability_type" not in mc_columns:
            logger.info("Migrating database: DROP TABLE model_capabilities (legacy format)")
            await conn.execute(text("DROP TABLE model_capabilities"))


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


async def _repair_orphaned_profiles() -> None:
    """Fix profiles with NULL model_id after migration from model_path.

    Tries to match orphaned profiles to existing models by checking if
    the profile name contains a model repo_id pattern. If no match is
    found, the profile is left with model_id=NULL (will show as
    'Unknown model' in the UI and can be reassigned manually).
    """
    async with engine.begin() as conn:
        # Check if server_profiles table exists (legacy) or execution_profiles (new)
        sql_check_server = (
            "SELECT name FROM sqlite_master WHERE type='table' AND name='server_profiles'"
        )
        result = await conn.execute(text(sql_check_server))
        has_server_profiles = result.fetchone() is not None

        sql_check_execution = (
            "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_profiles'"
        )
        result = await conn.execute(text(sql_check_execution))
        has_execution_profiles = result.fetchone() is not None

        # Determine which table to query
        table_name = "server_profiles" if has_server_profiles else "execution_profiles"
        if not (has_server_profiles or has_execution_profiles):
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
    """Initialize the database and create tables."""
    ensure_data_dir()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Run schema migrations for existing databases
    await migrate_schema()

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
