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
    """Create initial database schema."""
    # Create server_profiles table
    op.create_table(
        "server_profiles",
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
        sa.Column("auto_start", sa.Boolean(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("launchd_installed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_server_profiles_name"), "server_profiles", ["name"], unique=False)

    # Create running_instances table
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

    # Create settings table
    op.create_table(
        "settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_settings_key"), "settings", ["key"], unique=True)


def downgrade() -> None:
    """Drop all tables."""
    op.drop_index(op.f("ix_settings_key"), table_name="settings")
    op.drop_table("settings")
    op.drop_table("running_instances")
    op.drop_index(op.f("ix_server_profiles_name"), table_name="server_profiles")
    op.drop_table("server_profiles")
