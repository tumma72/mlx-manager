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

from .base import GenerativeProbe, estimate_context_window
from .steps import ProbeResult, ProbeStep, probe_step

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.ir import TextResult
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
    ) -> TextResult:
        """Generate a response using adapter's vision pipeline with a synthetic test image."""
        from PIL import Image

        adapter = loaded.adapter
        if adapter is None:
            msg = "No adapter available for generation"
            raise RuntimeError(msg)

        # Synthetic 64x64 gray test image
        test_image = Image.new("RGB", (64, 64), color=(128, 128, 128))

        # Temporarily configure adapter with probe-specific template options
        if template_options is not None:
            adapter.configure(template_options=template_options)

        try:
            return await adapter.generate(
                model=loaded.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                tools=tools,
                images=[test_image],
            )
        finally:
            if template_options is not None:
                adapter.configure(template_options=None)

    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        """Type-specific static checks for vision models.

        Generative capability probing (thinking, tools) is handled by
        the ProbingCoordinator which calls this strategy first for
        static checks, then runs its own parser sweep.
        """
        result.model_type = ModelType.VISION

        # Step 1: Validate image processor works with a synthetic image
        async with probe_step("check_processor") as ctx:
            yield ctx.running
            processor_ok = _check_processor(loaded)
            if not processor_ok:
                ctx.fail("Processor could not handle test image")
        yield ctx.result

        # Step 2: Check multi-image support from config
        async with probe_step("check_multi_image", "supports_multi_image") as ctx:
            yield ctx.running
            multi_image = _check_multi_image(model_id)
            result.supports_multi_image = multi_image
            ctx.value = multi_image
        yield ctx.result

        # Step 3: Check video support from config
        async with probe_step("check_video", "supports_video") as ctx:
            yield ctx.running
            video = _check_video_support(model_id)
            result.supports_video = video
            ctx.value = video
        yield ctx.result

        # Step 4: Estimate practical context window
        async with probe_step("check_context", "practical_max_tokens") as ctx:
            yield ctx.running
            practical_max = estimate_context_window(model_id, loaded.size_gb)
            result.practical_max_tokens = practical_max
            ctx.value = practical_max
        yield ctx.result


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
