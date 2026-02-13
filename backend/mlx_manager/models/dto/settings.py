"""Settings DTOs - backend mappings, cloud credentials, server config, timeouts."""

from datetime import datetime

from pydantic import BaseModel, Field

from mlx_manager.models.enums import (
    ApiType,
    BackendType,
    EvictionPolicy,
    MemoryLimitMode,
    PatternType,
)

__all__ = [
    "BackendMappingCreate",
    "BackendMappingUpdate",
    "BackendMappingResponse",
    "CloudCredentialCreate",
    "CloudCredentialResponse",
    "ServerConfigUpdate",
    "ServerConfigResponse",
    "RulePriorityUpdate",
    "RuleMatchResult",
    "TimeoutSettings",
    "TimeoutSettingsUpdate",
]


class BackendMappingCreate(BaseModel):
    """Schema for creating a backend mapping."""

    model_pattern: str
    pattern_type: PatternType = PatternType.EXACT  # "exact", "prefix", or "regex"
    backend_type: BackendType
    backend_model: str | None = None
    fallback_backend: BackendType | None = None
    priority: int = 0


class BackendMappingUpdate(BaseModel):
    """Schema for updating a backend mapping."""

    model_pattern: str | None = None
    pattern_type: PatternType | None = None
    backend_type: BackendType | None = None
    backend_model: str | None = None
    fallback_backend: BackendType | None = None
    priority: int | None = None
    enabled: bool | None = None


class BackendMappingResponse(BaseModel):
    """Response model for backend mapping."""

    id: int
    model_pattern: str
    pattern_type: PatternType
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


class ServerConfigUpdate(BaseModel):
    """Schema for updating server configuration."""

    memory_limit_mode: MemoryLimitMode | None = None
    memory_limit_value: int | None = None
    eviction_policy: EvictionPolicy | None = None
    preload_models: list[str] | None = None


class ServerConfigResponse(BaseModel):
    """Response model for server configuration."""

    memory_limit_mode: MemoryLimitMode
    memory_limit_value: int
    eviction_policy: EvictionPolicy
    preload_models: list[str]  # Parsed from JSON


class RulePriorityUpdate(BaseModel):
    """Schema for batch updating rule priorities."""

    id: int
    priority: int


class RuleMatchResult(BaseModel):
    """Result of testing which rule matches a model name."""

    matched_rule_id: int | None
    backend_type: BackendType | None
    pattern_matched: str | None = None


class TimeoutSettings(BaseModel):
    """Timeout configuration for MLX Server endpoints."""

    chat_seconds: float = Field(default=900.0, ge=60, le=7200)
    completions_seconds: float = Field(default=600.0, ge=60, le=7200)
    embeddings_seconds: float = Field(default=120.0, ge=30, le=600)


class TimeoutSettingsUpdate(BaseModel):
    """Update model for timeout settings."""

    chat_seconds: float | None = Field(default=None, ge=60, le=7200)
    completions_seconds: float | None = Field(default=None, ge=60, le=7200)
    embeddings_seconds: float | None = Field(default=None, ge=30, le=600)
