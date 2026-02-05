"""Application configuration.

Environment Variables:
    MLX_MANAGER_JWT_SECRET: JWT signing secret (required for production)
    MLX_MANAGER_DATABASE_PATH: SQLite database location (default: ~/.mlx-manager/mlx-manager.db)
    MLX_MANAGER_HF_CACHE_PATH: HuggingFace cache directory
    MLX_MANAGER_OFFLINE_MODE: Disable HuggingFace API calls (default: false)
    MLX_MANAGER_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR). Default: INFO
    MLX_MANAGER_LOG_DIR: Log directory path (default: ~/.mlx-manager/logs/)
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All settings can be configured via environment variables with the
    MLX_MANAGER_ prefix. For example, MLX_MANAGER_DEBUG=true enables debug mode.

    Logging is configured separately via MLX_MANAGER_LOG_LEVEL and
    MLX_MANAGER_LOG_DIR environment variables (see logging_config.py).
    """

    model_config = SettingsConfigDict(env_prefix="MLX_MANAGER_")

    # JWT Authentication
    # Default secret is 32+ bytes (256+ bits) to meet RFC 7518 Section 3.2 requirements for HS256
    jwt_secret: str = Field(default="CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR")
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = 7

    # Database
    database_path: Path = Path.home() / ".mlx-manager" / "mlx-manager.db"

    # Server
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False

    # HuggingFace
    hf_cache_path: Path = Path.home() / ".cache" / "huggingface" / "hub"
    # Filter by author/organization (None = search all MLX models from any author)
    # Set to specific org like "mlx-community" to filter by that author only
    hf_organization: str | None = None

    # Offline mode - set to True to disable HuggingFace dependencies
    offline_mode: bool = False

    # Server defaults
    default_port_start: int = 10240
    max_memory_percent: int = 80
    health_check_interval: int = 30

    # Allowed model directories
    allowed_model_dirs: list[str] = [
        str(Path.home() / ".cache" / "huggingface"),
        str(Path.home() / "models"),
    ]


settings = Settings()


def ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
