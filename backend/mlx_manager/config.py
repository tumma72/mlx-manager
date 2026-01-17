"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="MLX_MANAGER_")

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
