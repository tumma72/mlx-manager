"""Vision model probe strategy.

Tests image processing, multi-image support, video support,
and estimates practical context window.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel


class VisionProbe:
    """Probe strategy for vision-language models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.VISION

    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        result.model_type = ModelType.VISION

        # Step 1: Validate image processor works with a synthetic image
        yield ProbeStep(step="check_processor", status="running")
        try:
            processor_ok = _check_processor(loaded)
            if processor_ok:
                yield ProbeStep(step="check_processor", status="completed")
            else:
                yield ProbeStep(
                    step="check_processor",
                    status="failed",
                    error="Processor could not handle test image",
                )
        except Exception as e:
            logger.warning(f"Processor check failed for {model_id}: {e}")
            yield ProbeStep(step="check_processor", status="failed", error=str(e))

        # Step 2: Check multi-image support from config
        yield ProbeStep(step="check_multi_image", status="running")
        try:
            multi_image = _check_multi_image(model_id)
            result.supports_multi_image = multi_image
            yield ProbeStep(
                step="check_multi_image",
                status="completed",
                capability="supports_multi_image",
                value=multi_image,
            )
        except Exception as e:
            logger.warning(f"Multi-image check failed for {model_id}: {e}")
            yield ProbeStep(step="check_multi_image", status="failed", error=str(e))

        # Step 3: Check video support from config
        yield ProbeStep(step="check_video", status="running")
        try:
            video = _check_video_support(model_id)
            result.supports_video = video
            yield ProbeStep(
                step="check_video",
                status="completed",
                capability="supports_video",
                value=video,
            )
        except Exception as e:
            logger.warning(f"Video check failed for {model_id}: {e}")
            yield ProbeStep(step="check_video", status="failed", error=str(e))

        # Step 4: Estimate practical context window
        yield ProbeStep(step="check_context", status="running")
        try:
            practical_max = _estimate_vision_max_tokens(model_id, loaded)
            result.practical_max_tokens = practical_max
            yield ProbeStep(
                step="check_context",
                status="completed",
                capability="practical_max_tokens",
                value=practical_max,
            )
        except Exception as e:
            logger.warning(f"Context check failed for {model_id}: {e}")
            yield ProbeStep(step="check_context", status="failed", error=str(e))


def _check_processor(loaded: LoadedModel) -> bool:
    """Verify the vision processor can handle a synthetic test image."""
    try:
        from PIL import Image

        # Create a small synthetic image (64x64 RGB)
        test_image = Image.new("RGB", (64, 64), color=(128, 128, 128))

        processor = loaded.tokenizer  # Vision models store processor as tokenizer
        if processor is None:
            return False

        # Try to process the image — different processors have different APIs
        # Most accept images via __call__ or process_images
        if hasattr(processor, "image_processor") and processor.image_processor is not None:
            processor.image_processor(test_image)
            return True

        return True  # Processor loaded successfully even if we can't test directly
    except Exception as e:
        logger.debug(f"Processor check error: {e}")
        raise


def _check_multi_image(model_id: str) -> bool:
    """Check if the model supports multiple images per request.

    Models with image_token_id in config typically support multiple images
    by inserting the token at each image position.
    """
    from mlx_manager.utils.model_detection import read_model_config

    config = read_model_config(model_id)
    if not config:
        return False

    # Models with explicit image token IDs can handle multiple images
    return any(key in config for key in ("image_token_id", "image_token_index"))


def _check_video_support(model_id: str) -> bool:
    """Check if the model supports video input."""
    from mlx_manager.utils.model_detection import read_model_config

    config = read_model_config(model_id)
    if not config:
        return False

    return "video_token_id" in config


def _estimate_vision_max_tokens(model_id: str, loaded: LoadedModel) -> int | None:
    """Estimate practical max tokens for vision models.

    Vision models share the same KV cache formula as text models
    but may use text_config for the LLM backbone parameters.
    """
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Vision models often nest LLM params under text_config
        text_config = config.get("text_config", config)

        max_pos = text_config.get(
            "max_position_embeddings",
            text_config.get("max_sequence_length"),
        )
        if max_pos is None:
            return None

        num_layers = text_config.get("num_hidden_layers", text_config.get("num_layers"))
        num_kv_heads = text_config.get(
            "num_key_value_heads",
            text_config.get("num_attention_heads"),
        )
        head_dim = text_config.get("head_dim")
        if head_dim is None:
            hidden_size = text_config.get("hidden_size")
            num_heads = text_config.get("num_attention_heads")
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads

        if not all([num_layers, num_kv_heads, head_dim]):
            return int(max_pos)

        kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2

        from mlx_manager.mlx_server.utils.memory import get_device_memory_gb

        device_memory_gb = get_device_memory_gb()
        model_size_gb = loaded.size_gb

        available_gb = (device_memory_gb * 0.75) - model_size_gb - 1.0
        if available_gb <= 0:
            return int(min(max_pos, 2048))

        available_bytes = available_gb * 1e9
        practical_max = int(min(max_pos, available_bytes / kv_per_token))
        return max(practical_max, 512)

    except Exception as e:
        logger.debug(f"Could not estimate vision max tokens for {model_id}: {e}")
        return None
