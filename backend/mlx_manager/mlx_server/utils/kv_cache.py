"""KV cache estimation utility.

Shared formula for estimating practical max tokens based on
device memory, model size, and KV cache requirements.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


def estimate_practical_max_tokens(
    model_id: str,
    model_size_gb: float,
    *,
    config_override: dict[str, Any] | None = None,
) -> int | None:
    """Estimate practical max tokens based on KV cache memory requirements.

    Works for both text and vision models. For vision models, the function
    automatically checks text_config for nested LLM backbone parameters.

    Args:
        model_id: HuggingFace model ID (used to read config.json)
        model_size_gb: Size of the loaded model in GB
        config_override: Optional pre-loaded config dict (skips file read)

    Returns:
        Estimated practical max tokens, or None if estimation fails
    """
    try:
        if config_override is not None:
            config = config_override
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(model_id, "config.json")
            with open(config_path) as f:
                config = json.load(f)

        # Vision models often nest LLM params under text_config
        text_config = config.get("text_config", config)

        max_pos = text_config.get(
            "max_position_embeddings",
            text_config.get("max_sequence_length", text_config.get("seq_length")),
        )
        if max_pos is None:
            return None

        num_layers = text_config.get(
            "num_hidden_layers",
            text_config.get("n_layer", text_config.get("num_layers")),
        )
        num_kv_heads = text_config.get(
            "num_key_value_heads",
            text_config.get("num_attention_heads", text_config.get("n_head")),
        )
        head_dim = text_config.get("head_dim")
        if head_dim is None:
            hidden_size = text_config.get("hidden_size", text_config.get("d_model"))
            num_heads = text_config.get("num_attention_heads", text_config.get("n_head"))
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads

        if not all([num_layers, num_kv_heads, head_dim]):
            return int(max_pos)

        # KV cache cost per token in bytes (fp16)
        kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2

        # Available memory
        from mlx_manager.mlx_server.utils.memory import get_device_memory_gb

        device_memory_gb = get_device_memory_gb()

        available_gb = (device_memory_gb * 0.75) - model_size_gb - 1.0
        if available_gb <= 0:
            return int(min(max_pos, 2048))  # Minimum practical context

        available_bytes = available_gb * 1e9
        practical_max = int(min(max_pos, available_bytes / kv_per_token))

        return max(practical_max, 512)  # At least 512 tokens

    except Exception as e:
        logger.debug("Could not estimate max tokens for {}: {}", model_id, e)
        return None
