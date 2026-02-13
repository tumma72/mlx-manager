"""Model types for the MLX server."""

from pydantic import BaseModel

from mlx_manager.models.enums import ModelType

# Re-export ModelType for backward compatibility
__all__ = ["AdapterInfo", "ModelType"]


class AdapterInfo(BaseModel):
    """Information about a loaded LoRA adapter."""

    adapter_path: str
    base_model: str | None = None  # From adapter_config.json if available
    description: str | None = None
