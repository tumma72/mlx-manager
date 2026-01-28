"""Inference service for MLX model generation.

CRITICAL: This module implements stop token detection in the generation loop.
mlx_lm's stream_generate() doesn't accept stop_tokens directly - we must
check each generated token against the model's terminators to prevent
Llama 3.x models from generating indefinitely.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from queue import Empty, Queue

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

    CRITICAL: This function uses a dedicated thread for MLX generation to
    respect Metal GPU thread affinity. Tokens are passed via Queue to the
    async generator. Using run_in_executor with next(generator) does NOT work
    because MLX Metal operations have thread affinity requirements.
    """
    from mlx_lm import stream_generate

    from mlx_manager.mlx_server.utils.memory import clear_cache

    completion_tokens = 0

    # Queue for passing tokens from generation thread to async generator
    # Format: (token_text, token_id, is_stop) or Exception or None (completion signal)
    token_queue: Queue[tuple[str, int | None, bool] | Exception | None] = Queue()

    def run_generation() -> None:
        """Run MLX generation in dedicated thread (owns Metal context)."""
        try:
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

                # CRITICAL: Check for stop token
                is_stop = token_id is not None and token_id in stop_token_ids
                token_queue.put((token_text, token_id, is_stop))

                if is_stop:
                    return
        except Exception as e:
            # Put exception marker for async side to handle
            token_queue.put(e)
        finally:
            # Signal completion
            token_queue.put(None)

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

        # Start generation thread (daemon=True so it doesn't block process exit)
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        loop = asyncio.get_running_loop()

        while True:
            # Poll queue without blocking event loop (use run_in_executor for queue.get)
            try:
                result = await loop.run_in_executor(
                    None, lambda: token_queue.get(timeout=0.1)
                )
            except Empty:
                continue

            # Check for completion signal
            if result is None:
                break

            # Check for exception from generation thread
            if isinstance(result, Exception):
                raise result

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

        # Wait for thread to finish
        gen_thread.join(timeout=1.0)

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

    CRITICAL: Uses a dedicated thread for MLX generation to respect Metal GPU
    thread affinity. Results are passed via Queue. Using run_in_executor with
    next(generator) does NOT work because MLX Metal has thread affinity.
    """
    from mlx_lm import stream_generate

    from mlx_manager.mlx_server.utils.memory import clear_cache

    prompt_tokens = len(tokenizer.encode(prompt))

    # Queue for passing result from generation thread
    # Format: (response_text, finish_reason) or Exception
    result_queue: Queue[tuple[str, str] | Exception] = Queue()

    def run_generation() -> None:
        """Run complete generation in dedicated thread (owns Metal context)."""
        try:
            response_text = ""
            finish_reason = "length"

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
                    finish_reason = "stop"
                    break

                response_text += token_text

            result_queue.put((response_text, finish_reason))
        except Exception as e:
            result_queue.put(e)

    try:
        # Start generation thread
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # Wait for result (with timeout to not block forever - 5 min max)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: result_queue.get(timeout=300)
        )

        gen_thread.join(timeout=1.0)

        # Check for exception from generation thread
        if isinstance(result, Exception):
            raise result

        response_text, finish_reason = result
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
    """Stream raw completion tokens.

    CRITICAL: Uses a dedicated thread for MLX generation to respect Metal GPU
    thread affinity. Tokens are passed via Queue to the async generator.
    """
    from mlx_lm import stream_generate

    from mlx_manager.mlx_server.utils.memory import clear_cache

    # Queue for passing tokens from generation thread to async generator
    token_queue: Queue[tuple[str, int | None, bool] | Exception | None] = Queue()

    def run_generation() -> None:
        """Run MLX generation in dedicated thread (owns Metal context)."""
        try:
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

                is_stop = token_id is not None and token_id in stop_tokens
                token_queue.put((token_text, token_id, is_stop))

                if is_stop:
                    return
        except Exception as e:
            token_queue.put(e)
        finally:
            token_queue.put(None)

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

        # Start generation thread
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        loop = asyncio.get_running_loop()

        while True:
            try:
                result = await loop.run_in_executor(
                    None, lambda: token_queue.get(timeout=0.1)
                )
            except Empty:
                continue

            if result is None:
                break

            if isinstance(result, Exception):
                raise result

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

        # Wait for thread to finish
        gen_thread.join(timeout=1.0)

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
    """Generate complete raw completion.

    CRITICAL: Uses a dedicated thread for MLX generation to respect Metal GPU
    thread affinity. Results are passed via Queue.
    """
    from mlx_lm import stream_generate

    from mlx_manager.mlx_server.utils.memory import clear_cache

    prompt_tokens = len(tokenizer.encode(prompt))

    # Queue for passing result from generation thread
    result_queue: Queue[tuple[str, str] | Exception] = Queue()

    def run_generation() -> None:
        """Run complete generation in dedicated thread (owns Metal context)."""
        try:
            response_text = ""
            finish_reason = "length"

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
                    finish_reason = "stop"
                    break

                response_text += token_text

            result_queue.put((response_text, finish_reason))
        except Exception as e:
            result_queue.put(e)

    try:
        # Start generation thread
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # Wait for result (with timeout - 5 min max)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: result_queue.get(timeout=300)
        )

        gen_thread.join(timeout=1.0)

        # Check for exception from generation thread
        if isinstance(result, Exception):
            raise result

        response_text, finish_reason = result

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
