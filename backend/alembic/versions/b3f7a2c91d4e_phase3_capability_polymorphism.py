"""Phase 3: Capability polymorphism (single table)

Revision ID: b3f7a2c91d4e
Revises: 9fc5d838420b
Create Date: 2026-02-12

Migrates capability fields from the flat ``models`` table into a dedicated
``model_capabilities`` table with a ``capability_type`` discriminator.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3f7a2c91d4e"
down_revision: str | Sequence[str] | None = "9fc5d838420b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create model_capabilities table and migrate data from models."""
    op.create_table(
        "model_capabilities",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "model_id",
            sa.Integer(),
            sa.ForeignKey("models.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column("capability_type", sa.String(), nullable=False),
        sa.Column("probed_at", sa.DateTime(), nullable=True),
        sa.Column("probe_version", sa.Integer(), nullable=True),
        sa.Column("model_family", sa.String(), nullable=True),
        # Text-gen fields
        sa.Column("supports_native_tools", sa.Boolean(), nullable=True),
        sa.Column("supports_thinking", sa.Boolean(), nullable=True),
        sa.Column("tool_format", sa.String(), nullable=True),
        sa.Column("practical_max_tokens", sa.Integer(), nullable=True),
        sa.Column("tool_parser_id", sa.String(), nullable=True),
        sa.Column("thinking_parser_id", sa.String(), nullable=True),
        # Vision fields
        sa.Column("supports_multi_image", sa.Boolean(), nullable=True),
        sa.Column("supports_video", sa.Boolean(), nullable=True),
        # Embedding fields
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
        sa.Column("max_sequence_length", sa.Integer(), nullable=True),
        sa.Column("is_normalized", sa.Boolean(), nullable=True),
        # Audio fields
        sa.Column("supports_tts", sa.Boolean(), nullable=True),
        sa.Column("supports_stt", sa.Boolean(), nullable=True),
    )

    # Data migration: copy probed model data into capabilities table
    conn = op.get_bind()

    # Check if models table has the old columns (if not, this is a fresh DB)
    result = conn.execute(sa.text("PRAGMA table_info(models)"))
    columns = {row[1] for row in result.fetchall()}

    if "probed_at" in columns:
        # Fetch all probed models
        rows = conn.execute(
            sa.text(
                "SELECT id, model_type, probed_at, probe_version, model_family, "
                "supports_native_tools, supports_thinking, tool_format, "
                "practical_max_tokens, tool_parser_id, thinking_parser_id, "
                "supports_multi_image, supports_video, "
                "embedding_dimensions, max_sequence_length, is_normalized, "
                "supports_tts, supports_stt "
                "FROM models WHERE probed_at IS NOT NULL"
            )
        ).fetchall()

        for row in rows:
            (
                model_id,
                model_type,
                probed_at,
                probe_version,
                model_family,
                supports_native_tools,
                supports_thinking,
                tool_format,
                practical_max_tokens,
                tool_parser_id,
                thinking_parser_id,
                supports_multi_image,
                supports_video,
                embedding_dimensions,
                max_sequence_length,
                is_normalized,
                supports_tts,
                supports_stt,
            ) = row

            cap_type = model_type or "text-gen"

            conn.execute(
                sa.text(
                    "INSERT INTO model_capabilities "
                    "(model_id, capability_type, probed_at, probe_version, "
                    "model_family, supports_native_tools, supports_thinking, "
                    "tool_format, practical_max_tokens, tool_parser_id, "
                    "thinking_parser_id, supports_multi_image, supports_video, "
                    "embedding_dimensions, max_sequence_length, is_normalized, "
                    "supports_tts, supports_stt) "
                    "VALUES (:model_id, :cap_type, :probed_at, :probe_version, "
                    ":model_family, :snt, :st, :tf, :pmt, :tpi, :thi, "
                    ":smi, :sv, :ed, :msl, :isn, :tts, :stt)"
                ),
                {
                    "model_id": model_id,
                    "cap_type": cap_type,
                    "probed_at": probed_at,
                    "probe_version": probe_version,
                    "model_family": model_family,
                    "snt": supports_native_tools,
                    "st": supports_thinking,
                    "tf": tool_format,
                    "pmt": practical_max_tokens,
                    "tpi": tool_parser_id,
                    "thi": thinking_parser_id,
                    "smi": supports_multi_image,
                    "sv": supports_video,
                    "ed": embedding_dimensions,
                    "msl": max_sequence_length,
                    "isn": is_normalized,
                    "tts": supports_tts,
                    "stt": supports_stt,
                },
            )

        # Drop old capability columns from models
        capability_columns = [
            "probed_at",
            "probe_version",
            "model_family",
            "supports_native_tools",
            "supports_thinking",
            "tool_format",
            "practical_max_tokens",
            "tool_parser_id",
            "thinking_parser_id",
            "supports_multi_image",
            "supports_video",
            "embedding_dimensions",
            "max_sequence_length",
            "is_normalized",
            "supports_tts",
            "supports_stt",
        ]
        for col in capability_columns:
            if col in columns:
                try:
                    op.drop_column("models", col)
                except Exception:
                    pass  # Column may not exist in all DB versions


def downgrade() -> None:
    """Reverse: re-add columns to models, migrate data back, drop table."""
    # Re-add capability columns to models
    with op.batch_alter_table("models") as batch_op:
        batch_op.add_column(sa.Column("probed_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("probe_version", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("model_family", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("supports_native_tools", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("supports_thinking", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("tool_format", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("practical_max_tokens", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("tool_parser_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("thinking_parser_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("supports_multi_image", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("supports_video", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("embedding_dimensions", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("max_sequence_length", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("is_normalized", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("supports_tts", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("supports_stt", sa.Boolean(), nullable=True))

    # Migrate data back from capabilities table
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT * FROM model_capabilities")).fetchall()

    for row in rows:
        conn.execute(
            sa.text(
                "UPDATE models SET "
                "probed_at = :probed_at, probe_version = :probe_version, "
                "model_family = :model_family, "
                "supports_native_tools = :snt, supports_thinking = :st, "
                "tool_format = :tf, practical_max_tokens = :pmt, "
                "tool_parser_id = :tpi, thinking_parser_id = :thi, "
                "supports_multi_image = :smi, supports_video = :sv, "
                "embedding_dimensions = :ed, max_sequence_length = :msl, "
                "is_normalized = :isn, "
                "supports_tts = :tts, supports_stt = :stt "
                "WHERE id = :model_id"
            ),
            {
                "model_id": row.model_id,
                "probed_at": row.probed_at,
                "probe_version": row.probe_version,
                "model_family": row.model_family,
                "snt": row.supports_native_tools,
                "st": row.supports_thinking,
                "tf": row.tool_format,
                "pmt": row.practical_max_tokens,
                "tpi": row.tool_parser_id,
                "thi": row.thinking_parser_id,
                "smi": row.supports_multi_image,
                "sv": row.supports_video,
                "ed": row.embedding_dimensions,
                "msl": row.max_sequence_length,
                "isn": row.is_normalized,
                "tts": row.supports_tts,
                "stt": row.supports_stt,
            },
        )

    op.drop_table("model_capabilities")
