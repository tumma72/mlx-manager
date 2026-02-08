"""Model probe package.

Provides type-specific probing of model capabilities using
a strategy pattern. Each model type (text-gen, vision, embeddings,
audio) has its own probe strategy that knows which capabilities
to test and how.

Usage:
    from mlx_manager.services.probe import probe_model, ProbeStep

    async for step in probe_model(model_id):
        print(step.to_sse())
"""

from .service import probe_model
from .steps import ProbeResult, ProbeStep

# Register all built-in strategies on import
from .strategy import (
    _register_all,
    get_probe_strategy,
    has_probe_strategy,
    registered_model_types,
)

_register_all()

__all__ = [
    "ProbeResult",
    "ProbeStep",
    "get_probe_strategy",
    "has_probe_strategy",
    "probe_model",
    "registered_model_types",
]
