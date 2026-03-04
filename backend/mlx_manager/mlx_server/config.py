"""MLX Server configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlx_manager.config import DEFAULT_PORT

# Default database paths
DEFAULT_MLX_MANAGER_DB = "~/.mlx-manager/mlx-manager.db"
DEFAULT_MLX_SERVER_DB = "~/.mlx-manager/mlx-server.db"

# Settings that require a server restart to take effect — cannot be hot-reloaded.
IMMUTABLE_SETTINGS: frozenset[str] = frozenset({"database_path", "host", "port"})


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
    port: int = Field(default=DEFAULT_PORT, description="Port to bind the server to")

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
        default=0.0,
        description="Maximum memory (GB) for model pool (0 = auto-detect 75% of device memory)",
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

    # Image validation
    max_image_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum decoded base64 image size in MB",
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

    # Batching
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
    audit_max_mb: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum audit log database size in MB before oldest records are purged",
    )
    audit_cleanup_interval_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="How often to run audit log cleanup in minutes",
    )

    # Rate limiting
    rate_limit_rpm: int = Field(
        default=0,
        ge=0,
        description="Rate limit: requests per minute per IP. 0 = disabled.",
    )

    # Graceful shutdown
    drain_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Seconds to wait for in-flight requests to complete during shutdown",
    )

    # Prometheus metrics
    metrics_enabled: bool = Field(
        default=False,
        description="Enable Prometheus metrics endpoint at /v1/admin/metrics",
    )

    # Admin authentication
    admin_token: str | None = Field(
        default=None,
        description="Bearer token for /v1/admin/* endpoints. When None, admin endpoints are open.",
    )

    # Startup preloading
    preload_models: list[str] = Field(
        default_factory=list,
        description="Model IDs to preload at server startup for reduced cold-start latency",
    )
    warmup_prompt: str = Field(
        default="Hello",
        description="Prompt to run after preloading to warm up the model",
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


def reload_settings() -> dict[str, Any]:
    """Clear the settings cache and reload from environment/files.

    Compares old and new settings, returning a dict that describes what
    changed.  Immutable settings (database_path, host, port) that have
    changed are recorded under a separate ``warnings`` key so callers can
    inform operators that a restart is required.

    Returns:
        A dict with two keys:
        - ``"changes"``: mapping of setting name ->
          ``{"old": <old_value>, "new": <new_value>}`` for every field
          whose value changed.
        - ``"warnings"``: list of human-readable strings for immutable
          settings that changed but cannot take effect without a restart.
    """
    old_settings = get_settings()
    old_values: dict[str, Any] = old_settings.model_dump()

    # Invalidate the lru_cache so the next call loads fresh values
    get_settings.cache_clear()
    new_settings = get_settings()
    new_values: dict[str, Any] = new_settings.model_dump()

    changes: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for key, old_val in old_values.items():
        new_val = new_values.get(key)
        if old_val != new_val:
            changes[key] = {"old": old_val, "new": new_val}
            if key in IMMUTABLE_SETTINGS:
                warnings.append(
                    f"{key} changed from {old_val!r} to {new_val!r}"
                    " but requires a server restart to take effect"
                )

    return {"changes": changes, "warnings": warnings}


# For backward compatibility
mlx_server_settings = get_settings()
