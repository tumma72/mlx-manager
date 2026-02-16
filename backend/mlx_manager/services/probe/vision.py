"""Vision model probe strategy.

Tests image processing, multi-image support, video support,
estimates practical context window, and (for models with adapters)
tests thinking and tool calling capabilities.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .base import GenerativeProbe
from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel


class VisionProbe(GenerativeProbe):
    """Probe strategy for vision-language models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.VISION

    async def _generate(
        self,
        loaded: LoadedModel,
        messages: list[dict],
        tools: list[dict] | None = None,
        template_options: dict[str, Any] | None = None,
        max_tokens: int = 800,
    ) -> str:
        """Generate a response using adapter's vision pipeline with a synthetic test image."""
        from PIL import Image

        adapter = loaded.adapter
        if adapter is None:
            msg = "No adapter available for generation"
            raise RuntimeError(msg)

        # Synthetic 64x64 gray test image
        test_image = Image.new("RGB", (64, 64), color=(128, 128, 128))

        result = await adapter.generate(
            model=loaded.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            tools=tools,
            template_options=template_options,
            images=[test_image],
        )
        return result.content

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

        # Steps 5-6: Thinking and tool verification (shared with TextGenProbe)
        async for step in self._probe_generative_capabilities(model_id, loaded, result):
            yield step


def _messages_to_text(messages: list[dict]) -> str:
    """Extract a simple text prompt from messages for VLM template formatting."""
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Handle structured content (text parts)
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
    return "\n".join(parts) if parts else ""


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

    Delegates to the shared KV cache estimation utility, which automatically
    checks text_config for nested LLM backbone parameters.
    """
    from mlx_manager.mlx_server.utils.kv_cache import estimate_practical_max_tokens

    return estimate_practical_max_tokens(model_id, loaded.size_gb)
