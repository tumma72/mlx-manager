"""MLX Server configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLXServerSettings(BaseSettings):
    """MLX Inference Server settings.

    All settings can be configured via environment variables with MLX_SERVER_ prefix.
    Example: MLX_SERVER_PORT=8000
    """

    model_config = SettingsConfigDict(env_prefix="MLX_SERVER_")

    # Server binding
    host: str = Field(default="127.0.0.1", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")

    # Model pool settings
    max_memory_gb: float = Field(
        default=48.0,
        description="Maximum memory (GB) for model pool",
    )
    max_models: int = Field(
        default=4,
        description="Maximum number of hot models in memory",
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


mlx_server_settings = MLXServerSettings()
