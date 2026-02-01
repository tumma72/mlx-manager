"""MLX Server configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default database paths
DEFAULT_MLX_MANAGER_DB = "~/.mlx-manager/mlx-manager.db"
DEFAULT_MLX_SERVER_DB = "~/.mlx-manager/mlx-server.db"


class MLXServerSettings(BaseSettings):
    """MLX Inference Server settings.

    All settings can be configured via environment variables with MLX_SERVER_ prefix.
    Example: MLX_SERVER_PORT=10242

    When embedded_mode=True, the MLX Server runs within MLX Manager:
    - Uses MLX Manager's database for shared audit logs
    - Skips LogFire configuration (parent app handles it)
    - Skips lifespan handler (parent app handles model pool init)
    """

    model_config = SettingsConfigDict(
        env_prefix="MLX_SERVER_",
        env_file=".env",
        extra="ignore",
    )

    # Embedded mode (set to True when running within MLX Manager)
    embedded_mode: bool = Field(
        default=False,
        description="Whether running embedded within MLX Manager",
    )

    # Server binding (standalone mode only)
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

    # Timeout settings (seconds) - per CONTEXT.md decisions
    # Same timeouts apply to both local and cloud backends
    timeout_chat_seconds: float = Field(
        default=900.0,  # 15 minutes
        description="Timeout for /v1/chat/completions endpoint",
    )
    timeout_completions_seconds: float = Field(
        default=600.0,  # 10 minutes
        description="Timeout for /v1/completions endpoint",
    )
    timeout_embeddings_seconds: float = Field(
        default=120.0,  # 2 minutes
        description="Timeout for /v1/embeddings endpoint",
    )

    # Audit logging
    database_path: str = Field(
        default=DEFAULT_MLX_SERVER_DB,
        description="Path to audit log database (uses MLX Manager's DB when embedded)",
    )
    audit_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days to retain audit logs before cleanup",
    )

    def get_database_path(self) -> Path:
        """Get the resolved database path.

        When embedded_mode is True, uses MLX Manager's shared database.
        Otherwise uses the configured database_path.

        Returns:
            Resolved Path to the database file
        """
        if self.embedded_mode:
            # Use MLX Manager's database for shared audit logs
            return Path(DEFAULT_MLX_MANAGER_DB).expanduser()
        return Path(self.database_path).expanduser()


@lru_cache
def get_settings() -> MLXServerSettings:
    """Get cached settings instance."""
    return MLXServerSettings()


def is_embedded() -> bool:
    """Check if running in embedded mode within MLX Manager.

    Returns:
        True if embedded_mode is enabled, False otherwise
    """
    return get_settings().embedded_mode


# For backward compatibility
mlx_server_settings = get_settings()
