"""Model types for the MLX server."""

from dataclasses import dataclass

from mlx_manager.models.enums import ModelType

# Re-export ModelType for backward compatibility
__all__ = ["AdapterInfo", "ModelType"]


@dataclass
class AdapterInfo:
    """Information about a loaded LoRA adapter."""

    adapter_path: str
    base_model: str | None = None  # From adapter_config.json if available
    description: str | None = None
