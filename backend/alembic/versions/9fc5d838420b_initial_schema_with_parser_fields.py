"""Initial schema with parser fields

Revision ID: 9fc5d838420b
Revises:
Create Date: 2026-01-15 14:18:45.034720

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9fc5d838420b"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create initial database schema (all tables)."""
    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(), nullable=False, server_default="'pending'"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("approved_by", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["approved_by"], ["users.id"]),
        sa.UniqueConstraint("email"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    # --- models ---
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
    op.create_index(op.f("ix_models_repo_id"), "models", ["repo_id"], unique=True)

    # --- server_profiles (later renamed to execution_profiles by c7e4a1b2d3f5) ---
    op.create_table(
        "server_profiles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("model_path", sa.String(), nullable=False),
        sa.Column("model_type", sa.String(), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("host", sa.String(), nullable=False),
        sa.Column("context_length", sa.Integer(), nullable=True),
        sa.Column("max_concurrency", sa.Integer(), nullable=False),
        sa.Column("queue_timeout", sa.Integer(), nullable=False),
        sa.Column("queue_size", sa.Integer(), nullable=False),
        sa.Column("tool_call_parser", sa.String(), nullable=True),
        sa.Column("reasoning_parser", sa.String(), nullable=True),
        sa.Column("message_converter", sa.String(), nullable=True),
        sa.Column("enable_auto_tool_choice", sa.Boolean(), nullable=False),
        sa.Column("trust_remote_code", sa.Boolean(), nullable=False),
        sa.Column("chat_template_file", sa.String(), nullable=True),
        sa.Column("log_level", sa.String(), nullable=False),
        sa.Column("log_file", sa.String(), nullable=True),
        sa.Column("no_log_file", sa.Boolean(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("max_tokens", sa.Integer(), nullable=True),
        sa.Column("top_p", sa.Float(), nullable=True),
        sa.Column("force_tool_injection", sa.Boolean(), nullable=True, server_default="0"),
        sa.Column("model_id", sa.Integer(), nullable=True),
        sa.Column("tts_default_voice", sa.String(), nullable=True),
        sa.Column("tts_default_speed", sa.Float(), nullable=True),
        sa.Column("tts_sample_rate", sa.Integer(), nullable=True),
        sa.Column("stt_default_language", sa.String(), nullable=True),
        sa.Column("auto_start", sa.Boolean(), nullable=False),
        sa.Column("launchd_installed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_server_profiles_name"), "server_profiles", ["name"], unique=False
    )

    # --- settings ---
    op.create_table(
        "settings",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("key"),
    )

    # --- downloads ---
    op.create_table(
        "downloads",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="'pending'"),
        sa.Column("total_bytes", sa.Integer(), nullable=True),
        sa.Column("downloaded_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_downloads_model_id"), "downloads", ["model_id"])

    # --- server_config ---
    op.create_table(
        "server_config",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "memory_limit_mode",
            sa.String(),
            nullable=False,
            server_default="'percent'",
        ),
        sa.Column("memory_limit_value", sa.Integer(), nullable=False, server_default="80"),
        sa.Column(
            "eviction_policy", sa.String(), nullable=False, server_default="'lru'"
        ),
        sa.Column("preload_models", sa.String(), nullable=False, server_default="'[]'"),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- backend_mappings ---
    op.create_table(
        "backend_mappings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_pattern", sa.String(), nullable=False),
        sa.Column(
            "pattern_type", sa.String(), nullable=False, server_default="'exact'"
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


def downgrade() -> None:
    """Drop all tables."""
    op.drop_index(op.f("ix_audit_logs_status"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_backend_type"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_model"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_timestamp"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_request_id"), table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_table("cloud_credentials")
    op.drop_index(
        op.f("ix_backend_mappings_model_pattern"), table_name="backend_mappings"
    )
    op.drop_table("backend_mappings")
    op.drop_table("server_config")
    op.drop_index(op.f("ix_downloads_model_id"), table_name="downloads")
    op.drop_table("downloads")
    op.drop_table("settings")
    op.drop_index(
        op.f("ix_server_profiles_name"), table_name="server_profiles"
    )
    op.drop_table("server_profiles")
    op.drop_index(op.f("ix_models_repo_id"), table_name="models")
    op.drop_table("models")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
