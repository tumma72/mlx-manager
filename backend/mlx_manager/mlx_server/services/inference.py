"""Inference service for MLX model generation.

CRITICAL: This module implements stop token detection in the generation loop.
mlx_lm's stream_generate() doesn't accept stop_tokens directly - we must
check each generated token against the model's terminators to prevent
Llama 3.x models from generating indefinitely.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

logger = logging.getLogger(__name__)


async def generate_chat_completion(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    stream: bool = False,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a chat completion.

    Args:
        model_id: HuggingFace model ID
        messages: List of {"role": str, "content": str}
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        stop: Additional stop strings (user-provided)
        stream: If True, yield chunks; if False, return complete response

    Yields/Returns:
        Streaming: yields chunk dicts
        Non-streaming: returns complete response dict
    """
    from mlx_manager.mlx_server.models.adapters import get_adapter
    from mlx_manager.mlx_server.models.pool import get_model_pool

    # Get model from pool
    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    model = loaded.model
    tokenizer = loaded.tokenizer

    # Get adapter for model family
    adapter = get_adapter(model_id)

    # Apply chat template
    prompt = adapter.apply_chat_template(tokenizer, messages, add_generation_prompt=True)

    # CRITICAL: Get stop token IDs from adapter
    # Llama 3.x requires BOTH eos_token_id AND <|eot_id|> (end of turn)
    # Without this, models will generate past the assistant's response
    stop_token_ids: set[int] = set(adapter.get_stop_tokens(tokenizer))
    logger.debug(f"Stop token IDs for {model_id}: {stop_token_ids}")

    # Generate unique ID for this completion
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    logger.info(f"Starting generation: {completion_id}, model={model_id}, max_tokens={max_tokens}")

    # LogFire span for observability
    span_context = None
    if LOGFIRE_AVAILABLE:
        span_context = logfire.span(
            "chat_completion",
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        span_context.__enter__()

    try:
        if stream:
            return _stream_chat_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                completion_id=completion_id,
                created=created,
                model_id=model_id,
            )
        else:
            return await _generate_chat_complete(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                completion_id=completion_id,
                created=created,
                model_id=model_id,
            )
    finally:
        if span_context:
            span_context.__exit__(None, None, None)


async def _stream_chat_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_ids: set[int],
    completion_id: str,
    created: int,
    model_id: str,
) -> AsyncGenerator[dict, None]:
    """Generate tokens with streaming for chat completion.

    CRITICAL: This function checks each token against stop_token_ids
    to halt generation when a terminator is encountered.
    """
    from mlx_lm import stream_generate

    from mlx_manager.mlx_server.utils.memory import clear_cache

    completion_tokens = 0

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

        finish_reason = "length"  # Default if we hit max_tokens

        # Stream generate and check tokens
        # mlx_lm stream_generate is blocking - run in thread pool
        def generate_with_stop_detection():
            """Generator that yields tokens until stop token or max_tokens."""
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            ):
                # Get token ID from response
                # Note: stream_generate response has .token attribute (int)
                # and .text attribute (str) for the decoded token
                token_id = getattr(response, "token", None)
                token_text = getattr(response, "text", str(response))

                # CRITICAL: Check for stop token BEFORE yielding
                if token_id is not None and token_id in stop_token_ids:
                    # Found stop token - signal completion
                    yield (token_text, token_id, True)  # is_stop=True
                    return

                yield (token_text, token_id, False)  # is_stop=False

        # Process tokens from generator
        loop = asyncio.get_event_loop()
        generator = generate_with_stop_detection()

        while True:
            try:
                # Get next token in thread pool
                result = await loop.run_in_executor(None, next, generator)
                token_text, token_id, is_stop = result

                completion_tokens += 1

                if is_stop:
                    # Stop token found - don't yield the stop token text
                    finish_reason = "stop"
                    break

                # Yield content chunk
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None,
                        }
                    ],
                }

            except StopIteration:
                # Generator exhausted (max_tokens reached)
                finish_reason = "length"
                break

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
                    "finish_reason": finish_reason,
                }
            ],
        }

        logger.info(
            f"Chat stream complete: {completion_id}, "
            f"tokens={completion_tokens}, reason={finish_reason}"
        )

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "stream_completion_finished",
                completion_id=completion_id,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )

    finally:
        # Always cleanup
        clear_cache()


async def _generate_chat_complete(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_ids: set[int],
    completion_id: str,
    created: int,
    model_id: str,
) -> dict:
    """Generate complete response (non-streaming) for chat completion.

    CRITICAL: Uses streaming internally to implement stop token detection,
    since mlx_lm.generate() doesn't support stop_tokens parameter.
    """
    from mlx_lm import stream_generate

    from mlx_manager.mlx_server.utils.memory import clear_cache

    prompt_tokens = len(tokenizer.encode(prompt))

    try:
        # Collect all generated text with stop token detection
        response_text = ""
        finish_reason = "length"

        def generate_with_stop_detection():
            """Generate tokens until stop token or max_tokens."""
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            ):
                token_id = getattr(response, "token", None)
                token_text = getattr(response, "text", str(response))

                # CRITICAL: Check for stop token
                if token_id is not None and token_id in stop_token_ids:
                    yield (token_text, True)  # is_stop=True
                    return

                yield (token_text, False)

        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        generator = generate_with_stop_detection()

        while True:
            try:
                result = await loop.run_in_executor(None, next, generator)
                token_text, is_stop = result

                if is_stop:
                    finish_reason = "stop"
                    break

                response_text += token_text

            except StopIteration:
                finish_reason = "length"
                break

        completion_tokens = len(tokenizer.encode(response_text))

        logger.info(
            f"Chat complete: {completion_id}, tokens={completion_tokens}, reason={finish_reason}"
        )

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "completion_finished",
                completion_id=completion_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason=finish_reason,
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
                    "finish_reason": finish_reason,
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


# --- Completion API (legacy) ---


async def generate_completion(
    model_id: str,
    prompt: str | list[str],
    max_tokens: int = 16,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    stream: bool = False,
    echo: bool = False,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a raw text completion (legacy API).

    Args:
        model_id: HuggingFace model ID
        prompt: Text prompt or list of prompts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        stop: Additional stop strings
        stream: If True, yield chunks; if False, return complete response
        echo: If True, include prompt in response

    Yields/Returns:
        Streaming: yields chunk dicts
        Non-streaming: returns complete response dict
    """
    from mlx_manager.mlx_server.models.adapters import get_adapter
    from mlx_manager.mlx_server.models.pool import get_model_pool

    # Handle list of prompts (use first for now, batch in Phase 9)
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    # Get model from pool
    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    model = loaded.model
    tokenizer = loaded.tokenizer

    # Get stop tokens (still use adapter for this)
    adapter = get_adapter(model_id)
    stop_tokens = set(adapter.get_stop_tokens(tokenizer))

    # Generate unique ID
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    logger.info(f"Starting completion: {completion_id}, model={model_id}, max_tokens={max_tokens}")

    if stream:
        return _stream_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_tokens=stop_tokens,
            completion_id=completion_id,
            created=created,
            model_id=model_id,
            echo=echo,
        )
    else:
        return await _generate_raw_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_tokens=stop_tokens,
            completion_id=completion_id,
            created=created,
            model_id=model_id,
            echo=echo,
        )


async def _stream_completion(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_tokens: set[int],
    completion_id: str,
    created: int,
    model_id: str,
    echo: bool,
) -> AsyncGenerator[dict, None]:
    """Stream raw completion tokens."""
    from mlx_lm import stream_generate

    from mlx_manager.mlx_server.utils.memory import clear_cache

    try:
        # Echo prompt if requested
        if echo:
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": prompt,
                        "finish_reason": None,
                    }
                ],
            }

        finish_reason = "length"

        # Generate tokens with stop detection
        def generate_with_stop_detection():
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            ):
                token_id = getattr(response, "token", None)
                token_text = getattr(response, "text", str(response))

                if token_id is not None and token_id in stop_tokens:
                    yield (token_text, token_id, True)
                    return

                yield (token_text, token_id, False)

        loop = asyncio.get_event_loop()
        generator = generate_with_stop_detection()

        while True:
            try:
                result = await loop.run_in_executor(None, next, generator)
                token_text, token_id, is_stop = result

                if is_stop:
                    finish_reason = "stop"
                    break

                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "text": token_text,
                            "finish_reason": None,
                        }
                    ],
                }

            except StopIteration:
                finish_reason = "length"
                break

        # Final chunk
        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": finish_reason,
                }
            ],
        }

    finally:
        clear_cache()


async def _generate_raw_completion(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_tokens: set[int],
    completion_id: str,
    created: int,
    model_id: str,
    echo: bool,
) -> dict:
    """Generate complete raw completion."""
    from mlx_lm import stream_generate

    from mlx_manager.mlx_server.utils.memory import clear_cache

    prompt_tokens = len(tokenizer.encode(prompt))

    try:
        response_text = ""
        finish_reason = "length"

        def generate_with_stop_detection():
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            ):
                token_id = getattr(response, "token", None)
                token_text = getattr(response, "text", str(response))

                if token_id is not None and token_id in stop_tokens:
                    yield (token_text, True)
                    return

                yield (token_text, False)

        loop = asyncio.get_event_loop()
        generator = generate_with_stop_detection()

        while True:
            try:
                result = await loop.run_in_executor(None, next, generator)
                token_text, is_stop = result

                if is_stop:
                    finish_reason = "stop"
                    break

                response_text += token_text

            except StopIteration:
                finish_reason = "length"
                break

        # Prepend prompt if echo requested
        text = (prompt + response_text) if echo else response_text
        completion_tokens = len(tokenizer.encode(response_text))

        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": finish_reason,
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
