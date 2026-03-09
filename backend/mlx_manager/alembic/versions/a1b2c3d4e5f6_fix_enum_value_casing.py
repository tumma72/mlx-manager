"""Fix enum value casing for SQLAlchemy compatibility.

Revision ID: a1b2c3d4e5f6
Revises: d8e3f5a7b912
Create Date: 2026-02-16

SQLAlchemy uses enum member NAMES (uppercase) for lookups, but the database
has lowercase VALUES. This migration updates all enum columns to use uppercase
values to match the Python enum member names.

Affected columns:
- cloud_credentials.api_type: 'openai' → 'OPENAI', 'anthropic' → 'ANTHROPIC'
- backend_mappings.pattern_type: 'exact' → 'EXACT', 'prefix' → 'PREFIX', 'regex' → 'REGEX'
- server_config.memory_limit_mode: 'percent' → 'PERCENT', 'gb' → 'GB'
- server_config.eviction_policy: 'lru' → 'LRU', 'lfu' → 'LFU', 'ttl' → 'TTL'
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "e1a2b3c4d5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Convert enum values from lowercase to uppercase."""
    conn = op.get_bind()

    # Check if cloud_credentials table exists and has data
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='cloud_credentials'")
    )
    if result.fetchone():
        # Update cloud_credentials.api_type
        conn.execute(
            sa.text(
                """
                UPDATE cloud_credentials
                SET api_type = CASE api_type
                    WHEN 'openai' THEN 'OPENAI'
                    WHEN 'anthropic' THEN 'ANTHROPIC'
                    ELSE UPPER(api_type)
                END
                WHERE api_type IS NOT NULL
                """
            )
        )

    # Check if backend_mappings table exists and has data
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='backend_mappings'")
    )
    if result.fetchone():
        # Update backend_mappings.pattern_type
        conn.execute(
            sa.text(
                """
                UPDATE backend_mappings
                SET pattern_type = CASE pattern_type
                    WHEN 'exact' THEN 'EXACT'
                    WHEN 'prefix' THEN 'PREFIX'
                    WHEN 'regex' THEN 'REGEX'
                    ELSE UPPER(pattern_type)
                END
                WHERE pattern_type IS NOT NULL
                """
            )
        )

    # Check if server_config table exists and has data
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='server_config'")
    )
    if result.fetchone():
        # Update server_config.memory_limit_mode
        conn.execute(
            sa.text(
                """
                UPDATE server_config
                SET memory_limit_mode = CASE memory_limit_mode
                    WHEN 'percent' THEN 'PERCENT'
                    WHEN 'gb' THEN 'GB'
                    ELSE UPPER(memory_limit_mode)
                END
                WHERE memory_limit_mode IS NOT NULL
                """
            )
        )

        # Update server_config.eviction_policy
        conn.execute(
            sa.text(
                """
                UPDATE server_config
                SET eviction_policy = CASE eviction_policy
                    WHEN 'lru' THEN 'LRU'
                    WHEN 'lfu' THEN 'LFU'
                    WHEN 'ttl' THEN 'TTL'
                    ELSE UPPER(eviction_policy)
                END
                WHERE eviction_policy IS NOT NULL
                """
            )
        )


def downgrade() -> None:
    """Revert enum values from uppercase to lowercase."""
    conn = op.get_bind()

    # Check if cloud_credentials table exists and has data
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='cloud_credentials'")
    )
    if result.fetchone():
        # Revert cloud_credentials.api_type
        conn.execute(
            sa.text(
                """
                UPDATE cloud_credentials
                SET api_type = CASE api_type
                    WHEN 'OPENAI' THEN 'openai'
                    WHEN 'ANTHROPIC' THEN 'anthropic'
                    ELSE LOWER(api_type)
                END
                WHERE api_type IS NOT NULL
                """
            )
        )

    # Check if backend_mappings table exists and has data
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='backend_mappings'")
    )
    if result.fetchone():
        # Revert backend_mappings.pattern_type
        conn.execute(
            sa.text(
                """
                UPDATE backend_mappings
                SET pattern_type = CASE pattern_type
                    WHEN 'EXACT' THEN 'exact'
                    WHEN 'PREFIX' THEN 'prefix'
                    WHEN 'REGEX' THEN 'regex'
                    ELSE LOWER(pattern_type)
                END
                WHERE pattern_type IS NOT NULL
                """
            )
        )

    # Check if server_config table exists and has data
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='server_config'")
    )
    if result.fetchone():
        # Revert server_config.memory_limit_mode
        conn.execute(
            sa.text(
                """
                UPDATE server_config
                SET memory_limit_mode = CASE memory_limit_mode
                    WHEN 'PERCENT' THEN 'percent'
                    WHEN 'GB' THEN 'gb'
                    ELSE LOWER(memory_limit_mode)
                END
                WHERE memory_limit_mode IS NOT NULL
                """
            )
        )

        # Revert server_config.eviction_policy
        conn.execute(
            sa.text(
                """
                UPDATE server_config
                SET eviction_policy = CASE eviction_policy
                    WHEN 'LRU' THEN 'lru'
                    WHEN 'LFU' THEN 'lfu'
                    WHEN 'TTL' THEN 'ttl'
                    ELSE LOWER(eviction_policy)
                END
                WHERE eviction_policy IS NOT NULL
                """
            )
        )
