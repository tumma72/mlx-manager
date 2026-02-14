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

import json
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

    async def _generate(
        self,
        loaded: LoadedModel,
        messages: list[dict],
        tools: list[dict] | None = None,
        enable_thinking: bool = False,
        max_tokens: int = 800,
    ) -> str:
        """Generate a response using mlx-lm via the model's adapter."""
        import asyncio

        from mlx_lm import generate as mlx_generate

        adapter = loaded.adapter
        if adapter is None:
            msg = "No adapter available for generation"
            raise RuntimeError(msg)

        prompt = adapter.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        output = await asyncio.to_thread(
            mlx_generate,
            loaded.model,
            loaded.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return output

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

    Uses the formula:
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  (bytes, fp16)
    available_bytes = (device_memory * 0.75 - model_size_gb - 1.0) * 1e9
    practical_max = min(max_position_embeddings, available_bytes / kv_per_token)
    """
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        max_pos = config.get(
            "max_position_embeddings",
            config.get("max_sequence_length", config.get("seq_length")),
        )
        if max_pos is None:
            return None

        num_layers = config.get(
            "num_hidden_layers", config.get("n_layer", config.get("num_layers"))
        )
        num_kv_heads = config.get(
            "num_key_value_heads",
            config.get("num_attention_heads", config.get("n_head")),
        )
        head_dim = config.get("head_dim")
        if head_dim is None:
            hidden_size = config.get("hidden_size", config.get("d_model"))
            num_heads = config.get("num_attention_heads", config.get("n_head"))
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads

        if not all([num_layers, num_kv_heads, head_dim]):
            return int(max_pos)

        # KV cache cost per token in bytes (fp16)
        kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2

        # Available memory
        from mlx_manager.mlx_server.utils.memory import get_device_memory_gb

        device_memory_gb = get_device_memory_gb()
        model_size_gb = loaded.size_gb

        available_gb = (device_memory_gb * 0.75) - model_size_gb - 1.0
        if available_gb <= 0:
            return int(min(max_pos, 2048))  # Minimum practical context

        available_bytes = available_gb * 1e9
        practical_max = int(min(max_pos, available_bytes / kv_per_token))

        return max(practical_max, 512)  # At least 512 tokens

    except Exception as e:
        logger.debug("Could not estimate practical max tokens for {}: {}", model_id, e)
        return None
