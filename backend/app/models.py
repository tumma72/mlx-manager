"""SQLModel database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class ServerProfileBase(SQLModel):
    """Base model for server profiles."""

    name: str = Field(index=True)
    description: Optional[str] = None
    model_path: str
    model_type: str = Field(default="lm")
    port: int
    host: str = Field(default="127.0.0.1")
    context_length: Optional[int] = None
    max_concurrency: int = Field(default=1)
    queue_timeout: int = Field(default=300)
    queue_size: int = Field(default=100)
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    enable_auto_tool_choice: bool = Field(default=False)
    trust_remote_code: bool = Field(default=False)
    chat_template_file: Optional[str] = None
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = None
    no_log_file: bool = Field(default=False)
    auto_start: bool = Field(default=False)


class ServerProfile(ServerProfileBase, table=True):
    """Server profile database model."""

    __tablename__ = "server_profiles"

    id: Optional[int] = Field(default=None, primary_key=True)
    launchd_installed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ServerProfileCreate(ServerProfileBase):
    """Schema for creating a server profile."""

    pass


class ServerProfileUpdate(SQLModel):
    """Schema for updating a server profile."""

    name: Optional[str] = None
    description: Optional[str] = None
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    port: Optional[int] = None
    host: Optional[str] = None
    context_length: Optional[int] = None
    max_concurrency: Optional[int] = None
    queue_timeout: Optional[int] = None
    queue_size: Optional[int] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    enable_auto_tool_choice: Optional[bool] = None
    trust_remote_code: Optional[bool] = None
    chat_template_file: Optional[str] = None
    log_level: Optional[str] = None
    log_file: Optional[str] = None
    no_log_file: Optional[bool] = None
    auto_start: Optional[bool] = None


class RunningInstance(SQLModel, table=True):
    """Running server instance tracking."""

    __tablename__ = "running_instances"

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="server_profiles.id", unique=True)
    pid: int
    started_at: datetime = Field(default_factory=datetime.utcnow)
    health_status: str = Field(default="starting")
    last_health_check: Optional[datetime] = None


class DownloadedModel(SQLModel, table=True):
    """Downloaded models cache tracking."""

    __tablename__ = "downloaded_models"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: str = Field(unique=True, index=True)
    local_path: str
    size_bytes: Optional[int] = None
    downloaded_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None


class Setting(SQLModel, table=True):
    """Application settings."""

    __tablename__ = "settings"

    key: str = Field(primary_key=True)
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


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


class ModelSearchResult(SQLModel):
    """Model search result from HuggingFace."""

    model_id: str
    author: str
    downloads: int
    likes: int
    estimated_size_gb: float
    tags: list[str]
    is_downloaded: bool
    last_modified: Optional[str] = None


class LocalModel(SQLModel):
    """Locally downloaded model."""

    model_id: str
    local_path: str
    size_bytes: int
    size_gb: float


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
    mlx_version: Optional[str] = None
    mlx_openai_server_version: Optional[str] = None


class HealthStatus(SQLModel):
    """Server health status."""

    status: str
    response_time_ms: Optional[float] = None
    model_loaded: Optional[bool] = None
    error: Optional[str] = None


class LaunchdStatus(SQLModel):
    """Launchd service status."""

    installed: bool
    running: bool
    pid: Optional[int] = None
    label: str
