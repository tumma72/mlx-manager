"""Vision model inference service.

Handles vision-language model generation using mlx-vlm.

CRITICAL: This module uses the same queue-based threading pattern as inference.py
to respect MLX Metal thread affinity requirements.
"""

import asyncio
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from queue import Queue

from loguru import logger
from PIL import Image

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


async def generate_vision_completion(
    model_id: str,
    text_prompt: str,
    images: list[Image.Image],
    max_tokens: int = 4096,
    temperature: float = 0.7,
    stream: bool = False,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a vision completion from text and images.

    Args:
        model_id: HuggingFace model ID (must be a vision model)
        text_prompt: Text portion of the prompt
        images: List of PIL Image objects (preprocessed)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stream: If True, yield chunks; if False, return complete response

    Yields/Returns:
        Streaming: yields chunk dicts
        Non-streaming: returns complete response dict

    Raises:
        RuntimeError: If model loading or generation fails
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool
    from mlx_manager.mlx_server.models.types import ModelType

    # Get model from pool
    pool = get_model_pool()
    loaded = await pool.get_model(model_id)

    # Check if model was actually loaded as vision type
    # This can happen if detection was wrong at load time (e.g., config not available)
    if loaded.model_type != ModelType.VISION.value:
        raise RuntimeError(
            f"Model {model_id} was loaded as {loaded.model_type}, "
            f"but image input requires a vision model. "
            f"Please unload and reload the model, or use a vision-capable model."
        )

    model = loaded.model
    processor = loaded.tokenizer  # VLM stores processor in tokenizer field

    # Generate unique ID for this completion
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    logger.info(
        f"Starting vision generation: {completion_id}, model={model_id}, "
        f"images={len(images)}, max_tokens={max_tokens}"
    )

    # LogFire span
    span_context = None
    if LOGFIRE_AVAILABLE:
        span_context = logfire.span(
            "vision_completion",
            model=model_id,
            num_images=len(images),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        span_context.__enter__()

    try:
        if stream:
            return _stream_vision_generate(
                model=model,
                processor=processor,
                text_prompt=text_prompt,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
                completion_id=completion_id,
                created=created,
                model_id=model_id,
            )
        else:
            return await _generate_vision_complete(
                model=model,
                processor=processor,
                text_prompt=text_prompt,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
                completion_id=completion_id,
                created=created,
                model_id=model_id,
            )
    finally:
        if span_context:
            span_context.__exit__(None, None, None)


async def _stream_vision_generate(
    model,
    processor,
    text_prompt: str,
    images: list[Image.Image],
    max_tokens: int,
    temperature: float,
    completion_id: str,
    created: int,
    model_id: str,
) -> AsyncGenerator[dict, None]:
    """Generate vision completion with streaming.

    Note: mlx-vlm's generate() is non-streaming. We simulate streaming by
    running generation in a thread and yielding the complete response as
    a single chunk, then sending the finish chunk.

    TODO: Investigate mlx-vlm internals for true token-by-token streaming.
    """
    from mlx_manager.mlx_server.utils.memory import clear_cache

    # Queue for passing result from generation thread
    result_queue: Queue[str | Exception] = Queue()

    def run_generation() -> None:
        """Run vision generation in dedicated thread (owns Metal context)."""
        try:
            from mlx_vlm import generate as vlm_generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config

            # Load config for chat template
            config = load_config(model_id)

            # Apply chat template with image count
            formatted_prompt = apply_chat_template(
                processor, config, text_prompt, num_images=len(images)
            )

            # Generate response
            response = vlm_generate(
                model,
                processor,
                formatted_prompt,
                images,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )

            # mlx_vlm.generate returns GenerationResult, extract text
            result_queue.put(response.text)
        except Exception as e:
            result_queue.put(e)

    try:
        # First chunk with role
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }

        # Start generation thread
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # Wait for result (with 10 minute timeout for vision models)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: result_queue.get(timeout=600))

        gen_thread.join(timeout=1.0)

        # Check for exception
        if isinstance(result, Exception):
            raise result

        response_text = result

        # Yield content chunk
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": response_text},
                    "finish_reason": None,
                }
            ],
        }

        # Final chunk with finish_reason
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        logger.info(f"Vision stream complete: {completion_id}")

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "vision_stream_finished",
                completion_id=completion_id,
                response_length=len(response_text),
            )

    finally:
        clear_cache()


async def _generate_vision_complete(
    model,
    processor,
    text_prompt: str,
    images: list[Image.Image],
    max_tokens: int,
    temperature: float,
    completion_id: str,
    created: int,
    model_id: str,
) -> dict:
    """Generate complete vision response (non-streaming)."""
    from mlx_manager.mlx_server.utils.memory import clear_cache

    # Queue for passing result from generation thread
    result_queue: Queue[str | Exception] = Queue()

    def run_generation() -> None:
        """Run vision generation in dedicated thread (owns Metal context)."""
        try:
            from mlx_vlm import generate as vlm_generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config

            # Load config for chat template
            config = load_config(model_id)

            # Apply chat template with image count
            formatted_prompt = apply_chat_template(
                processor, config, text_prompt, num_images=len(images)
            )

            # Generate response
            response = vlm_generate(
                model,
                processor,
                formatted_prompt,
                images,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )

            # mlx_vlm.generate returns GenerationResult, extract text
            result_queue.put(response.text)
        except Exception as e:
            result_queue.put(e)

    try:
        # Start generation thread
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # Wait for result (with 10 minute timeout)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: result_queue.get(timeout=600))

        gen_thread.join(timeout=1.0)

        # Check for exception
        if isinstance(result, Exception):
            raise result

        response_text = result

        # Estimate tokens (rough approximation)
        completion_tokens = len(response_text.split())
        prompt_tokens = len(text_prompt.split()) + (len(images) * 256)  # ~256 tokens per image

        logger.info(f"Vision complete: {completion_id}, response_length={len(response_text)}")

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "vision_completion_finished",
                completion_id=completion_id,
                response_length=len(response_text),
            )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    finally:
        clear_cache()
