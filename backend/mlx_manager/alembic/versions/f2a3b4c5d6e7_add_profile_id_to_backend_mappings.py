"""Add profile_id to backend_mappings.

Revision ID: f2a3b4c5d6e7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-02

Adds a nullable profile_id foreign key column to backend_mappings so that
a routing rule can target a specific local ExecutionProfile instead of (or
in addition to) a cloud backend.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f2a3b4c5d6e7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add profile_id column to backend_mappings."""
    op.add_column(
        "backend_mappings",
        sa.Column("profile_id", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    """Remove profile_id column from backend_mappings."""
    op.drop_column("backend_mappings", "profile_id")
