"""Model adapters for family-specific handling."""

from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter, ModelAdapter
from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter
from mlx_manager.mlx_server.models.adapters.registry import (
    detect_model_family,
    get_adapter,
    get_supported_families,
    register_adapter,
)

__all__ = [
    "ModelAdapter",
    "DefaultAdapter",
    "LlamaAdapter",
    "get_adapter",
    "detect_model_family",
    "register_adapter",
    "get_supported_families",
]
