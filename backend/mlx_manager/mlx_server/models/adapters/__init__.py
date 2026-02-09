"""Model adapters for family-specific handling."""

from mlx_manager.mlx_server.models.adapters.composable import (
    FAMILY_REGISTRY,
    DefaultAdapter,
    GemmaAdapter,
    GLM4Adapter,
    LlamaAdapter,
    MistralAdapter,
    ModelAdapter,
    QwenAdapter,
    create_adapter,
)
from mlx_manager.mlx_server.models.adapters.registry import (
    FAMILY_PATTERNS,
    detect_model_family,
)

__all__ = [
    # Protocol and base
    "ModelAdapter",
    "DefaultAdapter",
    # Family-specific adapters
    "QwenAdapter",
    "GLM4Adapter",
    "LlamaAdapter",
    "GemmaAdapter",
    "MistralAdapter",
    # Factory and registry
    "create_adapter",
    "FAMILY_REGISTRY",
    # Family detection
    "detect_model_family",
    "FAMILY_PATTERNS",
]
