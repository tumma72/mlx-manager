"""Add missing tables for existing users

Revision ID: e1a2b3c4d5f6
Revises: d8e3f5a7b912
Create Date: 2026-02-16

For users who upgraded through the Alembic chain (initial migration only
created server_profiles, running_instances, settings), this migration adds
all tables that were previously only created by SQLModel.metadata.create_all().

Uses table-existence checks so it's safe for both:
- Users who ran all prior migrations (tables missing)
- Pre-Alembic users stamped at d8e3f5a7b912 (tables already exist via create_all)
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1a2b3c4d5f6"
down_revision: str | Sequence[str] | None = "d8e3f5a7b912"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_exists(table_name: str) -> bool:
    """Check if a table exists in the current database."""
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=:name"
        ),
        {"name": table_name},
    )
    return result.fetchone() is not None


def upgrade() -> None:
    """Create tables that were previously only created by create_all()."""
    # --- users ---
    if not _table_exists("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("email", sa.String(), nullable=False),
            sa.Column("hashed_password", sa.String(), nullable=False),
            sa.Column("is_admin", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column(
                "status", sa.String(), nullable=False, server_default="'pending'"
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("approved_at", sa.DateTime(), nullable=True),
            sa.Column("approved_by", sa.Integer(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(["approved_by"], ["users.id"]),
            sa.UniqueConstraint("email"),
        )
        op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    # --- models ---
    if not _table_exists("models"):
        op.create_table(
            "models",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("repo_id", sa.String(), nullable=False),
            sa.Column("model_type", sa.String(), nullable=True),
            sa.Column("local_path", sa.String(), nullable=True),
            sa.Column("size_bytes", sa.Integer(), nullable=True),
            sa.Column("downloaded_at", sa.DateTime(), nullable=False),
            sa.Column("last_used_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("repo_id"),
        )
        op.create_index(
            op.f("ix_models_repo_id"), "models", ["repo_id"], unique=True
        )

    # --- downloads ---
    if not _table_exists("downloads"):
        op.create_table(
            "downloads",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("model_id", sa.String(), nullable=False),
            sa.Column(
                "status", sa.String(), nullable=False, server_default="'pending'"
            ),
            sa.Column("total_bytes", sa.Integer(), nullable=True),
            sa.Column(
                "downloaded_bytes", sa.Integer(), nullable=False, server_default="0"
            ),
            sa.Column("error", sa.String(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=False),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_downloads_model_id"), "downloads", ["model_id"])

    # --- server_config ---
    if not _table_exists("server_config"):
        op.create_table(
            "server_config",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column(
                "memory_limit_mode",
                sa.String(),
                nullable=False,
                server_default="'percent'",
            ),
            sa.Column(
                "memory_limit_value",
                sa.Integer(),
                nullable=False,
                server_default="80",
            ),
            sa.Column(
                "eviction_policy",
                sa.String(),
                nullable=False,
                server_default="'lru'",
            ),
            sa.Column(
                "preload_models",
                sa.String(),
                nullable=False,
                server_default="'[]'",
            ),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    # --- backend_mappings ---
    if not _table_exists("backend_mappings"):
        op.create_table(
            "backend_mappings",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("model_pattern", sa.String(), nullable=False),
            sa.Column(
                "pattern_type",
                sa.String(),
                nullable=False,
                server_default="'exact'",
            ),
            sa.Column("backend_type", sa.String(), nullable=False),
            sa.Column("backend_model", sa.String(), nullable=True),
            sa.Column("fallback_backend", sa.String(), nullable=True),
            sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default="1"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_backend_mappings_model_pattern"),
            "backend_mappings",
            ["model_pattern"],
        )

    # --- cloud_credentials ---
    if not _table_exists("cloud_credentials"):
        op.create_table(
            "cloud_credentials",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("backend_type", sa.String(), nullable=False),
            sa.Column(
                "api_type", sa.String(), nullable=False, server_default="'openai'"
            ),
            sa.Column("name", sa.String(), nullable=False, server_default="''"),
            sa.Column("encrypted_api_key", sa.String(), nullable=False),
            sa.Column("base_url", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    # --- audit_logs ---
    if not _table_exists("audit_logs"):
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("request_id", sa.String(), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("model", sa.String(), nullable=False),
            sa.Column("backend_type", sa.String(), nullable=False),
            sa.Column("endpoint", sa.String(), nullable=False),
            sa.Column("duration_ms", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("prompt_tokens", sa.Integer(), nullable=True),
            sa.Column("completion_tokens", sa.Integer(), nullable=True),
            sa.Column("total_tokens", sa.Integer(), nullable=True),
            sa.Column("error_type", sa.String(), nullable=True),
            sa.Column("error_message", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_audit_logs_request_id"), "audit_logs", ["request_id"]
        )
        op.create_index(
            op.f("ix_audit_logs_timestamp"), "audit_logs", ["timestamp"]
        )
        op.create_index(op.f("ix_audit_logs_model"), "audit_logs", ["model"])
        op.create_index(
            op.f("ix_audit_logs_backend_type"), "audit_logs", ["backend_type"]
        )
        op.create_index(op.f("ix_audit_logs_status"), "audit_logs", ["status"])

    # --- Drop running_instances if it exists (unused legacy table) ---
    if _table_exists("running_instances"):
        op.drop_table("running_instances")


def downgrade() -> None:
    """Reverse: drop tables added by this migration.

    Note: We can't know which tables were actually created by this migration
    vs. already existed, so downgrade drops all of them. The previous
    create_all() mechanism would recreate them on next startup anyway.
    """
    # Re-create running_instances (was in original migration)
    op.create_table(
        "running_instances",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("health_status", sa.String(), nullable=False),
        sa.Column("last_health_check", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Drop tables in reverse order of creation
    for table in [
        "audit_logs",
        "cloud_credentials",
        "backend_mappings",
        "server_config",
        "downloads",
    ]:
        if _table_exists(table):
            op.drop_table(table)

    # Don't drop models/users — they may have data from other migrations
