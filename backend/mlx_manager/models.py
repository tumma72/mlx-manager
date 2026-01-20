"""SQLModel database models."""

from datetime import UTC, datetime
from enum import Enum

from sqlmodel import Field, SQLModel


class UserStatus(str, Enum):
    """User account status."""

    PENDING = "pending"
    APPROVED = "approved"
    DISABLED = "disabled"


class UserBase(SQLModel):
    """Base model for users."""

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


class UserCreate(SQLModel):
    """Schema for creating a user (registration)."""

    email: str
    password: str


class UserPublic(UserBase):
    """Public response model for user (no password)."""

    id: int
    is_admin: bool
    status: UserStatus
    created_at: datetime


class UserLogin(SQLModel):
    """Schema for login request."""

    email: str
    password: str


class UserUpdate(SQLModel):
    """Schema for admin user updates."""

    email: str | None = None
    is_admin: bool | None = None
    status: UserStatus | None = None


class Token(SQLModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


class PasswordReset(SQLModel):
    """Schema for admin password reset."""

    password: str


class ServerProfileBase(SQLModel):
    """Base model for server profiles."""

    name: str = Field(index=True)
    description: str | None = None
    model_path: str
    model_type: str = Field(default="lm")
    port: int
    host: str = Field(default="127.0.0.1")
    context_length: int | None = None
    max_concurrency: int = Field(default=1)
    queue_timeout: int = Field(default=300)
    queue_size: int = Field(default=100)
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    message_converter: str | None = None
    enable_auto_tool_choice: bool = Field(default=False)
    trust_remote_code: bool = Field(default=False)
    chat_template_file: str | None = None
    log_level: str = Field(default="INFO")
    log_file: str | None = None
    no_log_file: bool = Field(default=False)
    auto_start: bool = Field(default=False)


class ServerProfile(ServerProfileBase, table=True):
    """Server profile database model."""

    __tablename__ = "server_profiles"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    launchd_installed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ServerProfileCreate(ServerProfileBase):
    """Schema for creating a server profile."""

    pass


class ServerProfileUpdate(SQLModel):
    """Schema for updating a server profile."""

    name: str | None = None
    description: str | None = None
    model_path: str | None = None
    model_type: str | None = None
    port: int | None = None
    host: str | None = None
    context_length: int | None = None
    max_concurrency: int | None = None
    queue_timeout: int | None = None
    queue_size: int | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    message_converter: str | None = None
    enable_auto_tool_choice: bool | None = None
    trust_remote_code: bool | None = None
    chat_template_file: str | None = None
    log_level: str | None = None
    log_file: str | None = None
    no_log_file: bool | None = None
    auto_start: bool | None = None


class RunningInstance(SQLModel, table=True):
    """Running server instance tracking."""

    __tablename__ = "running_instances"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="server_profiles.id", unique=True)
    pid: int
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    health_status: str = Field(default="starting")
    last_health_check: datetime | None = None


class DownloadedModel(SQLModel, table=True):
    """Downloaded models cache tracking."""

    __tablename__ = "downloaded_models"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(unique=True, index=True)
    local_path: str
    size_bytes: int | None = None
    downloaded_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    last_used_at: datetime | None = None


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
    status: str = Field(default="pending")  # pending, downloading, completed, failed
    total_bytes: int | None = None
    downloaded_bytes: int = Field(default=0)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    completed_at: datetime | None = None


# Response models for API
class ServerProfileResponse(ServerProfileBase):
    """Response model for server profile."""

    id: int
    launchd_installed: bool
    created_at: datetime
    updated_at: datetime


class RunningServerResponse(SQLModel):
    """Response model for running server status."""

    profile_id: int
    profile_name: str
    pid: int
    port: int
    health_status: str
    uptime_seconds: float
    memory_mb: float
    memory_percent: float = 0.0
    cpu_percent: float = 0.0


class ModelSearchResult(SQLModel):
    """Model search result from HuggingFace."""

    model_id: str
    author: str
    downloads: int
    likes: int
    estimated_size_gb: float
    tags: list[str]
    is_downloaded: bool
    last_modified: str | None = None


class LocalModel(SQLModel):
    """Locally downloaded model."""

    model_id: str
    local_path: str
    size_bytes: int
    size_gb: float
    characteristics: dict | None = None  # ModelCharacteristics from config.json


class SystemMemory(SQLModel):
    """System memory information."""

    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    mlx_recommended_gb: float


class SystemInfo(SQLModel):
    """System information."""

    os_version: str
    chip: str
    memory_gb: float
    python_version: str
    mlx_version: str | None = None
    mlx_openai_server_version: str | None = None


class HealthStatus(SQLModel):
    """Server health status."""

    status: str
    response_time_ms: float | None = None
    model_loaded: bool | None = None
    error: str | None = None


class LaunchdStatus(SQLModel):
    """Launchd service status."""

    installed: bool
    running: bool
    pid: int | None = None
    label: str


class ServerStatus(SQLModel):
    """Detailed server process status."""

    profile_id: int
    running: bool
    pid: int | None = None
    exit_code: int | None = None
    failed: bool = False
    error_message: str | None = None
