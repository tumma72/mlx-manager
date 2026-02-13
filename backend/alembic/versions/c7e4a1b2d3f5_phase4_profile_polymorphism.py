"""Phase 4: Profile polymorphism (single table)

Revision ID: c7e4a1b2d3f5
Revises: b3f7a2c91d4e
Create Date: 2026-02-13

Renames ``server_profiles`` to ``execution_profiles``, adds a
``profile_type`` discriminator, renames columns with ``default_`` prefix,
and drops legacy columns.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7e4a1b2d3f5"
down_revision: str | Sequence[str] | None = "b3f7a2c91d4e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Rename server_profiles -> execution_profiles with column renames."""
    # Step 1: Rename the table
    op.rename_table("server_profiles", "execution_profiles")

    # Step 2: Add profile_type column (nullable initially for data migration)
    op.add_column(
        "execution_profiles",
        sa.Column("profile_type", sa.String(), nullable=True),
    )

    # Step 3: Populate profile_type from joined models.model_type
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "UPDATE execution_profiles SET profile_type = "
            "CASE "
            "  WHEN model_id IS NOT NULL AND EXISTS ("
            "    SELECT 1 FROM models WHERE models.id = execution_profiles.model_id "
            "    AND models.model_type = 'audio'"
            "  ) THEN 'audio' "
            "  WHEN model_id IS NOT NULL AND EXISTS ("
            "    SELECT 1 FROM models WHERE models.id = execution_profiles.model_id "
            "    AND models.model_type = 'embeddings'"
            "  ) THEN 'base' "
            "  ELSE 'inference' "
            "END"
        )
    )

    # Step 4: Use batch mode to rename columns, drop legacy columns,
    # and set profile_type NOT NULL
    with op.batch_alter_table("execution_profiles", recreate="always") as batch_op:
        # Drop legacy columns
        batch_op.drop_column("tool_call_parser")
        batch_op.drop_column("reasoning_parser")
        batch_op.drop_column("message_converter")

        # Rename inference default columns (add default_ prefix)
        batch_op.alter_column("temperature", new_column_name="default_temperature")
        batch_op.alter_column("max_tokens", new_column_name="default_max_tokens")
        batch_op.alter_column("top_p", new_column_name="default_top_p")
        batch_op.alter_column("context_length", new_column_name="default_context_length")
        batch_op.alter_column("system_prompt", new_column_name="default_system_prompt")
        batch_op.alter_column(
            "force_tool_injection", new_column_name="default_enable_tool_injection"
        )

        # Rename audio default columns (normalize naming)
        batch_op.alter_column("tts_default_voice", new_column_name="default_tts_voice")
        batch_op.alter_column("tts_default_speed", new_column_name="default_tts_speed")
        batch_op.alter_column("tts_sample_rate", new_column_name="default_tts_sample_rate")
        batch_op.alter_column("stt_default_language", new_column_name="default_stt_language")

        # Make profile_type NOT NULL
        batch_op.alter_column("profile_type", nullable=False)


def downgrade() -> None:
    """Reverse: rename back to server_profiles with original columns."""
    with op.batch_alter_table("execution_profiles", recreate="always") as batch_op:
        # Reverse column renames
        batch_op.alter_column("default_temperature", new_column_name="temperature")
        batch_op.alter_column("default_max_tokens", new_column_name="max_tokens")
        batch_op.alter_column("default_top_p", new_column_name="top_p")
        batch_op.alter_column("default_context_length", new_column_name="context_length")
        batch_op.alter_column("default_system_prompt", new_column_name="system_prompt")
        batch_op.alter_column(
            "default_enable_tool_injection", new_column_name="force_tool_injection"
        )
        batch_op.alter_column("default_tts_voice", new_column_name="tts_default_voice")
        batch_op.alter_column("default_tts_speed", new_column_name="tts_default_speed")
        batch_op.alter_column("default_tts_sample_rate", new_column_name="tts_sample_rate")
        batch_op.alter_column("default_stt_language", new_column_name="stt_default_language")

        # Re-add legacy columns
        batch_op.add_column(sa.Column("tool_call_parser", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("reasoning_parser", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("message_converter", sa.Text(), nullable=True))

        # Drop profile_type
        batch_op.drop_column("profile_type")

    op.rename_table("execution_profiles", "server_profiles")
