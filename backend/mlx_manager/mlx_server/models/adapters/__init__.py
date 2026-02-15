"""Model adapters for family-specific handling."""

from mlx_manager.mlx_server.models.adapters.composable import (
    FAMILY_REGISTRY,
    DefaultAdapter,
    ModelAdapter,
    create_adapter,
)
from mlx_manager.mlx_server.models.adapters.configs import (
    FAMILY_CONFIGS,
    FamilyConfig,
)
from mlx_manager.mlx_server.models.adapters.registry import (
    FAMILY_PATTERNS,
    detect_model_family,
)

__all__ = [
    # Core adapter
    "ModelAdapter",
    "DefaultAdapter",
    # Config-driven families
    "FamilyConfig",
    "FAMILY_CONFIGS",
    # Factory and registry
    "create_adapter",
    "FAMILY_REGISTRY",
    # Family detection
    "detect_model_family",
    "FAMILY_PATTERNS",
]
