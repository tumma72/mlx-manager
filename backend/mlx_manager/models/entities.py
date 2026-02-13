"""SQLModel database entities.

This module contains ONLY table=True entities and shared base classes.
DTOs (request/response models) live in the dto/ package.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional

from sqlmodel import Field, Relationship, SQLModel

from mlx_manager.models.enums import (
    DownloadStatusEnum,
    EvictionPolicy,
    MemoryLimitMode,
    UserStatus,
)

if TYPE_CHECKING:
    from mlx_manager.models.capabilities import ModelCapabilities
    from mlx_manager.models.profiles import ExecutionProfile


class UserBase(SQLModel):
    """Base model for users. Shared by User entity and UserPublic DTO."""

    email: str = Field(unique=True, index=True)


class User(UserBase, table=True):
    """User database model."""

    __tablename__ = "users"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    hashed_password: str
    is_admin: bool = Field(default=False)
    status: UserStatus = Field(default=UserStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    approved_at: datetime | None = None
    approved_by: int | None = Field(default=None, foreign_key="users.id")


class Model(SQLModel, table=True):
    """Unified model entity for catalog metadata.

    Created when a model is discovered in HuggingFace cache or downloaded.
    Capability data lives in the ``ModelCapabilities`` table (STI),
    linked via the ``capabilities`` relationship.
    """

    __tablename__ = "models"

    id: int | None = Field(default=None, primary_key=True)
    repo_id: str = Field(unique=True, index=True)
    model_type: str | None = Field(default=None)

    # Download info
    local_path: str | None = None
    size_bytes: int | None = None
    downloaded_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    last_used_at: datetime | None = None

    # Relationships
    profiles: list["ExecutionProfile"] = Relationship(back_populates="model")
    capabilities: Optional["ModelCapabilities"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan"},
    )


class Setting(SQLModel, table=True):
    """Application settings."""

    __tablename__ = "settings"  # type: ignore

    key: str = Field(primary_key=True)
    value: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class Download(SQLModel, table=True):
    """Active download tracking."""

    __tablename__ = "downloads"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(index=True)
    # Status flow: pending -> downloading -> completed/failed
    #              downloading -> paused -> downloading (resume)
    #              downloading/paused/pending -> cancelled
    status: DownloadStatusEnum = Field(default=DownloadStatusEnum.PENDING)
    total_bytes: int | None = None
    downloaded_bytes: int = Field(default=0)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    completed_at: datetime | None = None


# ============================================================================
# Model Pool Configuration Entity (Phase 11 - Configuration UI)
# ============================================================================


class ServerConfig(SQLModel, table=True):
    """Global server configuration (singleton - only id=1 used)."""

    __tablename__ = "server_config"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    # Model pool settings
    memory_limit_mode: MemoryLimitMode = Field(default=MemoryLimitMode.PERCENT)  # "percent" or "gb"
    memory_limit_value: int = Field(default=80)  # % or GB depending on mode
    eviction_policy: EvictionPolicy = Field(default=EvictionPolicy.LRU)  # "lru", "lfu", "ttl"
    preload_models: str = Field(default="[]")  # JSON array of model paths
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
