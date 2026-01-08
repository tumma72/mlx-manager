"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_path: Path = Path.home() / ".mlx-manager" / "mlx-manager.db"

    # Server
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False

    # HuggingFace
    hf_cache_path: Path = Path.home() / ".cache" / "huggingface" / "hub"
    hf_organization: str = "mlx-community"

    # Server defaults
    default_port_start: int = 10240
    max_memory_percent: int = 80
    health_check_interval: int = 30

    # Allowed model directories
    allowed_model_dirs: list[str] = [
        str(Path.home() / ".cache" / "huggingface"),
        str(Path.home() / "models"),
    ]

    class Config:
        env_prefix = "MLX_MANAGER_"


settings = Settings()


def ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
