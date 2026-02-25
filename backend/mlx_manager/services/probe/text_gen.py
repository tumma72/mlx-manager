"""Text generation model probe strategy.

Tests thinking support, native tool support, and estimates
practical context window based on KV cache memory requirements.

Tool verification uses a 2-attempt, adapter-driven approach:
1. Template delivery: adapter passes tools= to tokenizer natively
2. Adapter delivery: adapter injects tool prompt into messages

Thinking verification uses generation-based validation with
adapter's thinking_parser, then sweeps all registered parsers.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from mlx_manager.mlx_server.models.types import ModelType

from .base import GenerativeProbe
from .steps import ProbeResult, ProbeStep, probe_step

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel

__all__ = [
    "TextGenProbe",
]


class TextGenProbe(GenerativeProbe):
    """Probe strategy for text generation models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.TEXT_GEN

    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        """Type-specific static checks for text-gen models.

        Generative capability probing (thinking, tools) is handled by
        the ProbingCoordinator which calls this strategy first for
        static checks, then runs its own parser sweep.
        """
        result.model_type = ModelType.TEXT_GEN

        # Detect model family (used by coordinator for adapter config)
        from mlx_manager.mlx_server.models.adapters.registry import (
            detect_model_family,
        )

        result.model_family = detect_model_family(model_id)

        # Step 1: Estimate practical context window
        async with probe_step("check_context", "practical_max_tokens") as ctx:
            yield ctx.running
            practical_max = _estimate_practical_max_tokens(model_id, loaded)
            result.practical_max_tokens = practical_max
            ctx.value = practical_max
        yield ctx.result


def _estimate_practical_max_tokens(model_id: str, loaded: Any) -> int | None:
    """Estimate practical max tokens based on model config and available memory.

    Delegates to the shared KV cache estimation utility.
    """
    from mlx_manager.mlx_server.utils.kv_cache import estimate_practical_max_tokens

    return estimate_practical_max_tokens(model_id, loaded.size_gb)
