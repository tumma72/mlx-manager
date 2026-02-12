"""SQLModel database models."""

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, computed_field
from sqlalchemy import Boolean, Column
from sqlmodel import Field, Relationship, SQLModel

from mlx_manager.models.enums import ApiType, BackendType, UserStatus


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


class UserCreate(BaseModel):
    """Schema for creating a user (registration)."""

    email: str
    password: str


class UserPublic(UserBase):
    """Public response model for user (no password)."""

    id: int
    is_admin: bool
    status: UserStatus
    created_at: datetime


class UserLogin(BaseModel):
    """Schema for login request."""

    email: str
    password: str


class UserUpdate(BaseModel):
    """Schema for admin user updates."""

    email: str | None = None
    is_admin: bool | None = None
    status: UserStatus | None = None


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


class PasswordReset(BaseModel):
    """Schema for admin password reset."""

    password: str


class ServerProfileBase(SQLModel):
    """Base model for server profiles.

    With the embedded MLX server, many previous fields are no longer needed:
    - port, host: No longer running separate servers
    - max_concurrency, queue_timeout, queue_size: Handled by model pool
    - tool_call_parser, reasoning_parser, message_converter: Handled by adapters
    - enable_auto_tool_choice, trust_remote_code: Not applicable
    - chat_template_file: Handled by tokenizer
    - log_level, log_file, no_log_file: Server-level settings
    """

    name: str = Field(index=True)
    description: str | None = None
    model_id: int | None = Field(default=None, foreign_key="models.id")
    context_length: int | None = None
    auto_start: bool = Field(default=False)
    system_prompt: str | None = None
    # Generation parameters (nullable — only meaningful for text/vision)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    # Tool calling
    enable_prompt_injection: bool = Field(
        default=False,
        sa_column=Column("force_tool_injection", Boolean, default=False),
    )
    # Audio parameters (nullable — only meaningful for audio models)
    tts_default_voice: str | None = None
    tts_default_speed: float | None = Field(default=None, ge=0.25, le=4.0)
    tts_sample_rate: int | None = None
    stt_default_language: str | None = None


class ServerProfile(ServerProfileBase, table=True):
    """Server profile database model."""

    __tablename__ = "server_profiles"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    launchd_installed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    # Relationships
    model: Optional["Model"] = Relationship(back_populates="profiles")


class ServerProfileCreate(ServerProfileBase):
    """Schema for creating a server profile."""

    model_id: int = Field(foreign_key="models.id")  # type: ignore[assignment]


class ServerProfileUpdate(BaseModel):
    """Schema for updating a server profile."""

    name: str | None = None
    description: str | None = None
    model_id: int | None = None
    context_length: int | None = None
    auto_start: bool | None = None
    system_prompt: str | None = None
    # Generation parameters
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    # Tool calling
    enable_prompt_injection: bool | None = None
    # Audio parameters
    tts_default_voice: str | None = None
    tts_default_speed: float | None = Field(default=None, ge=0.25, le=4.0)
    tts_sample_rate: int | None = None
    stt_default_language: str | None = None


class Model(SQLModel, table=True):
    """Unified model entity combining download info and probed capabilities.

    Created when a model is discovered in HuggingFace cache or downloaded.
    Capability fields are populated when the model is probed.
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

    # Probe info
    probed_at: datetime | None = None
    probe_version: int | None = None

    # Text-gen capabilities
    supports_native_tools: bool | None = Field(default=None)
    supports_thinking: bool | None = Field(default=None)
    tool_format: str | None = Field(default=None)
    practical_max_tokens: int | None = Field(default=None)

    # Composable adapter fields
    model_family: str | None = Field(default=None)
    tool_parser_id: str | None = Field(default=None)
    thinking_parser_id: str | None = Field(default=None)

    # Vision capabilities
    supports_multi_image: bool | None = Field(default=None)
    supports_video: bool | None = Field(default=None)

    # Embeddings capabilities
    embedding_dimensions: int | None = Field(default=None)
    max_sequence_length: int | None = Field(default=None)
    is_normalized: bool | None = Field(default=None)

    # Audio capabilities
    supports_tts: bool | None = Field(default=None)
    supports_stt: bool | None = Field(default=None)

    # Relationships
    profiles: list["ServerProfile"] = Relationship(back_populates="model")


class ModelResponse(BaseModel):
    """Response model for Model entity."""

    id: int
    repo_id: str
    model_type: str | None = None
    local_path: str | None = None
    size_bytes: int | None = None
    size_gb: float | None = None
    downloaded_at: datetime | None = None
    last_used_at: datetime | None = None
    probed_at: datetime | None = None
    probe_version: int | None = None
    supports_native_tools: bool | None = None
    supports_thinking: bool | None = None
    tool_format: str | None = None
    practical_max_tokens: int | None = None
    model_family: str | None = None
    tool_parser_id: str | None = None
    thinking_parser_id: str | None = None
    supports_multi_image: bool | None = None
    supports_video: bool | None = None
    embedding_dimensions: int | None = None
    max_sequence_length: int | None = None
    is_normalized: bool | None = None
    supports_tts: bool | None = None
    supports_stt: bool | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_id(self) -> str:
        """Alias for repo_id used by the frontend capabilities mapping."""
        return self.repo_id


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
    status: str = Field(default="pending")
    total_bytes: int | None = None
    downloaded_bytes: int = Field(default=0)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    completed_at: datetime | None = None


# Response models for API
class ServerProfileResponse(BaseModel):
    """Response model for server profile."""

    id: int
    name: str
    description: str | None = None
    model_id: int | None = None
    model_repo_id: str | None = None
    model_type: str | None = None
    context_length: int | None = None
    auto_start: bool
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    enable_prompt_injection: bool
    tts_default_voice: str | None = None
    tts_default_speed: float | None = None
    tts_sample_rate: int | None = None
    stt_default_language: str | None = None
    launchd_installed: bool
    created_at: datetime
    updated_at: datetime


class RunningServerResponse(BaseModel):
    """Response model for running server status."""

    profile_id: int
    profile_name: str
    pid: int
    health_status: str
    uptime_seconds: float
    memory_mb: float
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    port: int = 0
    memory_limit_percent: float = 0.0


class ModelSearchResult(BaseModel):
    """Model search result from HuggingFace."""

    model_id: str
    author: str
    downloads: int
    likes: int
    estimated_size_gb: float
    tags: list[str]
    is_downloaded: bool
    last_modified: str | None = None


class LocalModel(BaseModel):
    """Locally downloaded model."""

    model_id: str
    local_path: str
    size_bytes: int
    size_gb: float
    characteristics: dict | None = None  # ModelCharacteristics from config.json


class SystemMemory(BaseModel):
    """System memory information."""

    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    mlx_recommended_gb: float


class SystemInfo(BaseModel):
    """System information."""

    os_version: str
    chip: str
    memory_gb: float
    python_version: str
    mlx_version: str | None = None


class HealthStatus(BaseModel):
    """Server health status."""

    status: str
    response_time_ms: float | None = None
    model_loaded: bool | None = None
    error: str | None = None


class LaunchdStatus(BaseModel):
    """Launchd service status."""

    installed: bool
    running: bool
    pid: int | None = None
    label: str
    plist_path: str | None = None


class ServerStatus(BaseModel):
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


# Default base URLs for known providers
DEFAULT_BASE_URLS: dict[BackendType, str] = {
    BackendType.OPENAI: "https://api.openai.com",
    BackendType.ANTHROPIC: "https://api.anthropic.com",
    BackendType.TOGETHER: "https://api.together.xyz",
    BackendType.GROQ: "https://api.groq.com/openai",
    BackendType.FIREWORKS: "https://api.fireworks.ai/inference",
    BackendType.MISTRAL: "https://api.mistral.ai",
    BackendType.DEEPSEEK: "https://api.deepseek.com",
}

# API type mapping for each backend type
API_TYPE_FOR_BACKEND: dict[BackendType, ApiType] = {
    BackendType.OPENAI: ApiType.OPENAI,
    BackendType.ANTHROPIC: ApiType.ANTHROPIC,
    BackendType.TOGETHER: ApiType.OPENAI,
    BackendType.GROQ: ApiType.OPENAI,
    BackendType.FIREWORKS: ApiType.OPENAI,
    BackendType.MISTRAL: ApiType.OPENAI,
    BackendType.DEEPSEEK: ApiType.OPENAI,
    BackendType.OPENAI_COMPATIBLE: ApiType.OPENAI,
    BackendType.ANTHROPIC_COMPATIBLE: ApiType.ANTHROPIC,
}


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
    backend_type: BackendType  # No longer unique - allows multiple providers of same API type
    api_type: ApiType = Field(default=ApiType.OPENAI)  # Which API protocol to use
    name: str = Field(default="")  # Display name (e.g., "Groq", "Together", "My Custom API")
    encrypted_api_key: str  # Encrypted with AuthLib
    base_url: str | None = None  # Override default API URL (Azure OpenAI, proxies)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class BackendMappingCreate(BaseModel):
    """Schema for creating a backend mapping."""

    model_pattern: str
    pattern_type: str = "exact"  # "exact", "prefix", or "regex"
    backend_type: BackendType
    backend_model: str | None = None
    fallback_backend: BackendType | None = None
    priority: int = 0


class BackendMappingUpdate(BaseModel):
    """Schema for updating a backend mapping."""

    model_pattern: str | None = None
    pattern_type: str | None = None
    backend_type: BackendType | None = None
    backend_model: str | None = None
    fallback_backend: BackendType | None = None
    priority: int | None = None
    enabled: bool | None = None


class BackendMappingResponse(BaseModel):
    """Response model for backend mapping."""

    id: int
    model_pattern: str
    pattern_type: str
    backend_type: BackendType
    backend_model: str | None
    fallback_backend: BackendType | None
    priority: int
    enabled: bool


class CloudCredentialCreate(BaseModel):
    """Schema for creating cloud credentials."""

    backend_type: BackendType
    api_type: ApiType = ApiType.OPENAI
    name: str = ""
    api_key: str  # Plain text - will be encrypted before storage
    base_url: str | None = None


class CloudCredentialResponse(BaseModel):
    """Response model (no API key exposed)."""

    id: int
    backend_type: BackendType
    api_type: ApiType = ApiType.OPENAI  # Default for backwards compatibility
    name: str = ""  # Default for backwards compatibility
    base_url: str | None = None
    created_at: datetime | None = None


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


class ServerConfigUpdate(BaseModel):
    """Schema for updating server configuration."""

    memory_limit_mode: str | None = None
    memory_limit_value: int | None = None
    eviction_policy: str | None = None
    preload_models: list[str] | None = None


class ServerConfigResponse(BaseModel):
    """Response model for server configuration."""

    memory_limit_mode: str
    memory_limit_value: int
    eviction_policy: str
    preload_models: list[str]  # Parsed from JSON


# ============================================================================
# Settings Router Helper Models (Phase 11 - Configuration UI)
# ============================================================================


class RulePriorityUpdate(BaseModel):
    """Schema for batch updating rule priorities."""

    id: int
    priority: int


class RuleMatchResult(BaseModel):
    """Result of testing which rule matches a model name."""

    matched_rule_id: int | None
    backend_type: BackendType | None
    pattern_matched: str | None = None
