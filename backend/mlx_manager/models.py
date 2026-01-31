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
    system_prompt: str | None = None


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
    system_prompt: str | None = None


# NOTE: RunningInstance model removed - no longer needed with embedded MLX Server.
# The running_instances table can be manually dropped if it exists.


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


# ============================================================================
# Backend Routing Models (Phase 10 - Cloud Fallback)
# ============================================================================


class BackendType(str, Enum):
    """Backend types for routing."""

    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class BackendMapping(SQLModel, table=True):
    """Maps model patterns to backends with fallback configuration."""

    __tablename__ = "backend_mappings"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    model_pattern: str = Field(index=True)  # e.g., "gpt-*" or exact model name
    pattern_type: str = Field(default="exact")  # "exact", "prefix", or "regex"
    backend_type: BackendType
    backend_model: str | None = None  # Override model name for cloud
    fallback_backend: BackendType | None = None  # Optional fallback on failure
    priority: int = Field(default=0)  # Higher = checked first for pattern matching
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class CloudCredential(SQLModel, table=True):
    """Encrypted cloud API credentials."""

    __tablename__ = "cloud_credentials"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    backend_type: BackendType = Field(unique=True)  # One credential per backend type
    encrypted_api_key: str  # Encrypted with AuthLib (Phase 11 will add encryption)
    base_url: str | None = None  # Override default API URL (Azure OpenAI, proxies)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class BackendMappingCreate(SQLModel):
    """Schema for creating a backend mapping."""

    model_pattern: str
    pattern_type: str = "exact"  # "exact", "prefix", or "regex"
    backend_type: BackendType
    backend_model: str | None = None
    fallback_backend: BackendType | None = None
    priority: int = 0


class BackendMappingUpdate(SQLModel):
    """Schema for updating a backend mapping."""

    model_pattern: str | None = None
    pattern_type: str | None = None
    backend_type: BackendType | None = None
    backend_model: str | None = None
    fallback_backend: BackendType | None = None
    priority: int | None = None
    enabled: bool | None = None


class BackendMappingResponse(SQLModel):
    """Response model for backend mapping."""

    id: int
    model_pattern: str
    pattern_type: str
    backend_type: BackendType
    backend_model: str | None
    fallback_backend: BackendType | None
    priority: int
    enabled: bool


class CloudCredentialCreate(SQLModel):
    """Schema for creating cloud credentials."""

    backend_type: BackendType
    api_key: str  # Plain text - will be encrypted before storage
    base_url: str | None = None


class CloudCredentialResponse(SQLModel):
    """Response model (no API key exposed)."""

    id: int
    backend_type: BackendType
    base_url: str | None
    created_at: datetime


# ============================================================================
# Model Pool Configuration Models (Phase 11 - Configuration UI)
# ============================================================================


class ServerConfig(SQLModel, table=True):
    """Global server configuration (singleton - only id=1 used)."""

    __tablename__ = "server_config"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    # Model pool settings
    memory_limit_mode: str = Field(default="percent")  # "percent" or "gb"
    memory_limit_value: int = Field(default=80)  # % or GB depending on mode
    eviction_policy: str = Field(default="lru")  # "lru", "lfu", "ttl"
    preload_models: str = Field(default="[]")  # JSON array of model paths
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ServerConfigUpdate(SQLModel):
    """Schema for updating server configuration."""

    memory_limit_mode: str | None = None
    memory_limit_value: int | None = None
    eviction_policy: str | None = None
    preload_models: list[str] | None = None


class ServerConfigResponse(SQLModel):
    """Response model for server configuration."""

    memory_limit_mode: str
    memory_limit_value: int
    eviction_policy: str
    preload_models: list[str]  # Parsed from JSON


# ============================================================================
# Settings Router Helper Models (Phase 11 - Configuration UI)
# ============================================================================


class RulePriorityUpdate(SQLModel):
    """Schema for batch updating rule priorities."""

    id: int
    priority: int


class RuleMatchResult(SQLModel):
    """Result of testing which rule matches a model name."""

    matched_rule_id: int | None
    backend_type: BackendType | None
    pattern_matched: str | None = None
