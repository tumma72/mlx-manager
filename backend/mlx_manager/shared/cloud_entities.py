"""Cloud backend entities - shared between mlx_manager and mlx_server.

These entities are in a shared package to avoid circular dependencies
between the main app (mlx_manager) and the embedded server (mlx_server).
"""

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel

from mlx_manager.models.enums import ApiType, BackendType

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
