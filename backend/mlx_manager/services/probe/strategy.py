"""Probe strategy protocol and registry.

Defines the interface that all type-specific probes implement,
and provides the registry for looking up strategies by ModelType.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mlx_manager.mlx_server.models.types import ModelType

from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel


@runtime_checkable
class ProbeStrategy(Protocol):
    """Strategy for probing capabilities of a specific model type.

    Each model type (text-gen, vision, embeddings, audio) has a concrete
    implementation that knows which capabilities to test and how.
    """

    @property
    def model_type(self) -> ModelType:
        """The model type this strategy handles."""
        ...

    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        """Run type-specific probe steps.

        Yields ProbeStep objects for SSE streaming and populates
        the shared ProbeResult with discovered capabilities.

        Args:
            model_id: HuggingFace model path
            loaded: The loaded model from the pool
            result: Shared result object to populate with capabilities
        """
        ...  # pragma: no cover
        # Make this a valid async generator for type checking
        if False:  # pragma: no cover
            yield ProbeStep(step="", status="")


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_strategies: dict[ModelType, ProbeStrategy] = {}


def register_strategy(strategy: ProbeStrategy) -> None:
    """Register a probe strategy for a model type."""
    _strategies[strategy.model_type] = strategy


def get_probe_strategy(model_type: ModelType) -> ProbeStrategy | None:
    """Look up the probe strategy for a model type.

    Returns None if no strategy is registered for the type.
    """
    return _strategies.get(model_type)


def has_probe_strategy(model_type: ModelType) -> bool:
    """Check whether a probe strategy is registered for a model type."""
    return model_type in _strategies


def registered_model_types() -> list[ModelType]:
    """Return the list of model types that have probe strategies."""
    return list(_strategies.keys())


def _register_all() -> None:
    """Register all built-in probe strategies.

    Called once at import time from __init__.py.
    """
    from .audio import AudioProbe
    from .embeddings import EmbeddingsProbe
    from .text_gen import TextGenProbe
    from .vision import VisionProbe

    register_strategy(TextGenProbe())
    register_strategy(VisionProbe())
    register_strategy(EmbeddingsProbe())
    register_strategy(AudioProbe())
