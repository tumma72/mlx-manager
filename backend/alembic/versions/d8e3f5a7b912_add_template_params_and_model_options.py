"""Add template_params and model_options columns.

Revision ID: d8e3f5a7b912
Revises: c7e4a1b2d3f5
Create Date: 2026-02-15

Adds:
- model_capabilities.template_params (TEXT, nullable) — JSON-serialized
  template parameter metadata discovered during probing
- execution_profiles.model_options (TEXT, nullable) — JSON-serialized
  model-specific option overrides per profile
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "d8e3f5a7b912"
down_revision = "c7e4a1b2d3f5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_capabilities",
        sa.Column("template_params", sa.Text(), nullable=True),
    )
    op.add_column(
        "execution_profiles",
        sa.Column("model_options", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("execution_profiles", "model_options")
    op.drop_column("model_capabilities", "template_params")
