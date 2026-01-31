"""MLX Server configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLXServerSettings(BaseSettings):
    """MLX Inference Server settings.

    All settings can be configured via environment variables with MLX_SERVER_ prefix.
    Example: MLX_SERVER_PORT=10242
    """

    model_config = SettingsConfigDict(
        env_prefix="MLX_SERVER_",
        env_file=".env",
        extra="ignore",
    )

    # Server binding
    host: str = Field(default="127.0.0.1", description="Host to bind the server to")
    port: int = Field(default=10242, description="Port to bind the server to")

    # Available models (loadable via /v1/chat/completions)
    # These are model IDs that the server is configured to serve.
    # Models are loaded on-demand when requested.
    # Format: HuggingFace model ID (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")
    available_models: list[str] = Field(
        default_factory=lambda: [
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
        ],
        description="List of model IDs available for loading",
    )

    # Default model for inference (if not specified in request)
    default_model: str | None = Field(
        default=None,
        description="Default model ID to use when not specified in request",
    )

    # Model pool settings
    max_memory_gb: float = Field(
        default=48.0,
        description="Maximum memory (GB) for model pool",
    )
    max_models: int = Field(
        default=4,
        description="Maximum number of hot models in memory",
    )

    # Memory settings
    max_cache_size_gb: float = Field(
        default=8.0,
        ge=1.0,
        le=128.0,
        description="Maximum GPU cache size in GB",
    )

    # Generation defaults
    default_max_tokens: int = Field(
        default=4096,
        description="Default maximum tokens for generation",
    )

    # Observability
    logfire_enabled: bool = Field(
        default=True,
        description="Enable LogFire instrumentation",
    )
    logfire_token: str | None = Field(
        default=None,
        description="LogFire API token (optional, uses LOGFIRE_TOKEN env var if not set)",
    )

    # Continuous batching
    enable_batching: bool = Field(
        default=False,
        description="Enable continuous batching for text requests (experimental)",
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment (development/production)",
    )

    # Cloud routing
    enable_cloud_routing: bool = Field(
        default=False,
        description="Enable backend router for cloud fallback",
    )
    batch_block_pool_size: int = Field(
        default=1000,
        description="Number of KV cache blocks per model for batching",
    )
    batch_max_batch_size: int = Field(
        default=8,
        description="Maximum concurrent requests per batch",
    )


@lru_cache
def get_settings() -> MLXServerSettings:
    """Get cached settings instance."""
    return MLXServerSettings()


# For backward compatibility
mlx_server_settings = get_settings()
