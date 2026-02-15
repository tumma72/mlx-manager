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

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .base import (
    GenerativeProbe,
    _detect_unknown_xml_tags,
    _find_matching_parser,
    _validate_tool_output,
)
from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel

# Re-export helpers so existing test patches continue to work
__all__ = [
    "TextGenProbe",
    "_detect_unknown_xml_tags",
    "_find_matching_parser",
    "_validate_tool_output",
]


class TextGenProbe(GenerativeProbe):
    """Probe strategy for text generation models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.TEXT_GEN

    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        result.model_type = ModelType.TEXT_GEN

        # Detect model family
        from mlx_manager.mlx_server.models.adapters.registry import (
            detect_model_family,
        )

        result.model_family = detect_model_family(model_id)

        # Step 1: Estimate practical context window
        yield ProbeStep(step="check_context", status="running")
        try:
            practical_max = _estimate_practical_max_tokens(model_id, loaded)
            result.practical_max_tokens = practical_max
            yield ProbeStep(
                step="check_context",
                status="completed",
                capability="practical_max_tokens",
                value=practical_max,
            )
        except Exception as e:
            logger.warning("Context check failed for {}: {}", model_id, e)
            yield ProbeStep(step="check_context", status="failed", error=str(e))

        # Steps 2-3: Thinking and tool verification (shared with VisionProbe)
        async for step in self._probe_generative_capabilities(model_id, loaded, result):
            yield step


def _estimate_practical_max_tokens(model_id: str, loaded: Any) -> int | None:
    """Estimate practical max tokens based on model config and available memory.

    Delegates to the shared KV cache estimation utility.
    """
    from mlx_manager.mlx_server.utils.kv_cache import estimate_practical_max_tokens

    return estimate_practical_max_tokens(model_id, loaded.size_gb)
