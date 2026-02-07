"""Application configuration.

Environment Variables:
    MLX_MANAGER_JWT_SECRET: JWT signing secret (auto-generated if not set)
    MLX_MANAGER_DATABASE_PATH: SQLite database location (default: ~/.mlx-manager/mlx-manager.db)
    MLX_MANAGER_HF_CACHE_PATH: HuggingFace cache directory
    MLX_MANAGER_OFFLINE_MODE: Disable HuggingFace API calls (default: false)
    MLX_MANAGER_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR). Default: INFO
    MLX_MANAGER_LOG_DIR: Log directory path (default: ~/.mlx-manager/logs/)
"""

import os
import secrets
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Port constants — single source of truth
DEFAULT_PORT = 10242  # Production / installed version
DEV_PORT = 10241  # Development / testing (make dev)

_JWT_SECRET_PATH = Path.home() / ".mlx-manager" / ".jwt_secret"


def _resolve_jwt_secret() -> str:
    """Resolve the JWT secret, auto-generating and persisting one if needed.

    Priority: MLX_MANAGER_JWT_SECRET env var > persisted file > generate new.
    """
    if _JWT_SECRET_PATH.exists():
        return _JWT_SECRET_PATH.read_text().strip()

    secret = secrets.token_urlsafe(32)
    _JWT_SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    _JWT_SECRET_PATH.write_text(secret)
    os.chmod(_JWT_SECRET_PATH, 0o600)
    return secret


class Settings(BaseSettings):
    """Application settings.

    All settings can be configured via environment variables with the
    MLX_MANAGER_ prefix. For example, MLX_MANAGER_DEBUG=true enables debug mode.

    Logging is configured separately via MLX_MANAGER_LOG_LEVEL and
    MLX_MANAGER_LOG_DIR environment variables (see logging_config.py).
    """

    model_config = SettingsConfigDict(env_prefix="MLX_MANAGER_")

    # JWT Authentication
    jwt_secret: str = Field(default_factory=lambda: _resolve_jwt_secret())
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = 7

    # Database
    database_path: Path = Path.home() / ".mlx-manager" / "mlx-manager.db"

    # Server
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
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
