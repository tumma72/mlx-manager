"""Inference service for MLX model generation.

CRITICAL: This module implements stop token detection in the generation loop.
mlx_lm's stream_generate() doesn't accept stop_tokens directly - we must
check each generated token against the model's terminators to prevent
Llama 3.x models from generating indefinitely.

Extended capabilities:
- Tool calling: Tools injected into prompt, tool calls parsed from output
- Reasoning extraction: Chain-of-thought content in <think> tags extracted
- Structured output: JSON schema validation (handled at API layer)
"""

import time
import uuid
from collections.abc import AsyncGenerator, Iterator
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult

try:  # pragma: no cover
    import logfire  # pragma: no cover

    LOGFIRE_AVAILABLE = True  # pragma: no cover
except ImportError:  # pragma: no cover
    LOGFIRE_AVAILABLE = False  # pragma: no cover


class InferenceResult(BaseModel):
    """Non-streaming inference result with token counts."""

    result: TextResult
    prompt_tokens: int
    completion_tokens: int


class _GenContext(BaseModel):
    """Shared generation context prepared once per request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    tokenizer: Any
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    stop_token_ids: set[int]
    adapter: Any
    model_id: str
    completion_id: str
    created: int
    tools: list[dict[str, Any]] | None
    pixel_values: Any | None = None  # Vision: preprocessed images from PreparedInput


async def _prepare_generation(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    enable_prompt_injection: bool = False,
    images: list[Any] | None = None,
) -> _GenContext:
    """Prepare shared context for chat generation (model, adapter, prompt, stop tokens)."""
    from mlx_manager.mlx_server.models.pool import get_model_pool

    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    model = loaded.model
    tokenizer = loaded.tokenizer

    adapter = loaded.adapter
    if adapter is None:
        from mlx_manager.mlx_server.models.adapters.composable import DefaultAdapter

        adapter = DefaultAdapter(tokenizer)

    # Use adapter.prepare_input() for message conversion, template, and stop tokens
    prepared = adapter.prepare_input(
        messages,
        tools=tools,
        enable_prompt_injection=enable_prompt_injection,
        images=images,
    )

    if tools and not (adapter.supports_tool_calling() or enable_prompt_injection):
        logger.info(
            "Tools requested but model {} does not support "
            "tool calling and prompt injection not enabled",
            model_id,
        )

    if tools:
        logger.debug(f"=== PROMPT WITH TOOLS (last 1500 chars) ===\n{prepared.prompt[-1500:]}")

    stop_token_ids = set(prepared.stop_token_ids or [])
    logger.debug(f"Stop token IDs for {model_id}: {stop_token_ids}")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    logger.info(
        f"Starting generation: {completion_id}, model={model_id}, "
        f"max_tokens={max_tokens}, tools={'yes' if tools else 'no'}"
    )

    return _GenContext(
        model=model,
        tokenizer=tokenizer,
        prompt=prepared.prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
        adapter=adapter,
        model_id=model_id,
        completion_id=completion_id,
        created=created,
        tools=tools,
        pixel_values=prepared.pixel_values,
    )


# ── Public IR-returning functions ─────────────────────────────────


async def generate_chat_stream(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    enable_prompt_injection: bool = False,
    images: list[Any] | None = None,
) -> AsyncGenerator[StreamEvent | TextResult, None]:
    """Streaming chat generation returning IR events.

    Awaiting this coroutine prepares the generation context (model loading,
    template application, etc.) and returns an async generator.  The generator
    yields StreamEvent objects during generation, then a final TextResult
    with finish_reason, tool_calls, and reasoning_content.

    Callers apply a ProtocolFormatter to convert IR to protocol-specific SSE events.
    """
    ctx = await _prepare_generation(
        model_id,
        messages,
        max_tokens,
        temperature,
        top_p,
        stop,
        tools,
        enable_prompt_injection,
        images,
    )

    span_context = None
    if LOGFIRE_AVAILABLE:
        span_context = logfire.span(
            "chat_completion",
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            has_tools=tools is not None,
        )
        span_context.__enter__()

    async def _stream() -> AsyncGenerator[StreamEvent | TextResult, None]:
        try:
            async for item in _stream_chat_ir(ctx):
                yield item
        finally:
            if span_context:
                span_context.__exit__(None, None, None)

    return _stream()


async def generate_chat_complete_response(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    enable_prompt_injection: bool = False,
    images: list[Any] | None = None,
) -> InferenceResult:
    """Non-streaming chat generation returning IR result.

    Returns InferenceResult with TextResult and token counts.
    Callers apply a ProtocolFormatter to convert IR to protocol-specific response.
    """
    ctx = await _prepare_generation(
        model_id,
        messages,
        max_tokens,
        temperature,
        top_p,
        stop,
        tools,
        enable_prompt_injection,
        images,
    )

    span_context = None
    if LOGFIRE_AVAILABLE:
        span_context = logfire.span(
            "chat_completion",
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            has_tools=tools is not None,
        )
        span_context.__enter__()

    try:
        return await _complete_chat_ir(ctx)
    finally:
        if span_context:
            span_context.__exit__(None, None, None)


# ── Backward-compat wrapper (returns OpenAI dicts) ───────────────


async def generate_chat_completion(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
    enable_prompt_injection: bool = False,
    images: list[Any] | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a chat completion in OpenAI format.

    Backward-compatible wrapper. New code should prefer
    ``generate_chat_stream`` / ``generate_chat_complete_response``.

    Returns:
        Streaming: yields OpenAI chunk dicts
        Non-streaming: returns OpenAI ChatCompletion dict
    """
    ctx = await _prepare_generation(
        model_id,
        messages,
        max_tokens,
        temperature,
        top_p,
        stop,
        tools,
        enable_prompt_injection,
        images,
    )

    span_context = None
    if LOGFIRE_AVAILABLE:
        span_context = logfire.span(
            "chat_completion",
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            has_tools=tools is not None,
        )
        span_context.__enter__()

    try:
        if stream:
            return _stream_chat_generate(
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                prompt=ctx.prompt,
                max_tokens=ctx.max_tokens,
                temperature=ctx.temperature,
                top_p=ctx.top_p,
                stop_token_ids=ctx.stop_token_ids,
                completion_id=ctx.completion_id,
                created=ctx.created,
                model_id=ctx.model_id,
                adapter=ctx.adapter,
                tools=ctx.tools,
            )
        else:
            return await _generate_chat_complete(
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                prompt=ctx.prompt,
                max_tokens=ctx.max_tokens,
                temperature=ctx.temperature,
                top_p=ctx.top_p,
                stop_token_ids=ctx.stop_token_ids,
                completion_id=ctx.completion_id,
                created=ctx.created,
                model_id=ctx.model_id,
                adapter=ctx.adapter,
                tools=ctx.tools,
            )
    finally:
        if span_context:
            span_context.__exit__(None, None, None)


async def _stream_chat_ir(
    ctx: _GenContext,
) -> AsyncGenerator[StreamEvent | TextResult, None]:
    """Streaming generation yielding IR events.

    Yields StreamEvent for each token, then a final TextResult with
    finish_reason, tool_calls, and reasoning_content.
    """
    if ctx.pixel_values is not None:
        # Vision path: non-streaming generation, simulate streaming
        from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

        def run_vision_generation() -> str:
            """Run vision generation in dedicated thread (owns Metal context)."""
            from mlx_vlm import generate as vlm_generate

            response = vlm_generate(
                ctx.model,
                ctx.tokenizer,
                ctx.prompt,
                ctx.pixel_values,
                max_tokens=ctx.max_tokens,
                temp=ctx.temperature,
                verbose=False,
            )
            return str(response.text)

        response_text = await run_on_metal_thread(run_vision_generation, timeout=600.0)

        # Yield the full response as a single content event
        yield StreamEvent(type="content", content=response_text)

        # Process with adapter for tool/thinking extraction
        result = ctx.adapter.process_complete(response_text, "stop")
        yield result
        return

    # Text path: existing streaming logic
    from mlx_manager.mlx_server.utils.metal import stream_from_metal_thread

    completion_tokens = 0
    stream_processor = ctx.adapter.create_stream_processor(prompt=ctx.prompt)

    if ctx.tools:
        logger.debug(
            f"Generation config: model={ctx.model_id}, max_tokens={ctx.max_tokens}, "
            f"stop_tokens={ctx.stop_token_ids}, tools_count={len(ctx.tools)}"
        )

    def produce_tokens() -> Iterator[tuple[str, int | None, bool]]:
        """Yield (token_text, token_id, is_stop) tuples on Metal thread."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=ctx.temperature, top_p=ctx.top_p)

        for response in stream_generate(
            ctx.model,
            ctx.tokenizer,
            ctx.prompt,
            max_tokens=ctx.max_tokens,
            sampler=sampler,
        ):
            token_id = getattr(response, "token", None)
            token_text = getattr(response, "text", str(response))

            is_stop = token_id is not None and token_id in ctx.stop_token_ids
            yield (token_text, token_id, is_stop)

            if is_stop:
                return

    finish_reason = "length"

    async for token_text, token_id, is_stop in stream_from_metal_thread(produce_tokens):
        completion_tokens += 1

        if is_stop:
            finish_reason = "stop"
            stream_processor.feed(token_text)
            logger.debug(
                f"Stop token encountered: token_id={token_id}, "
                f"token_text={repr(token_text)}, tokens_so_far={completion_tokens}"
            )
            break

        event = stream_processor.feed(token_text)
        if event.reasoning_content or event.content:
            yield event

    # Finalize: use adapter.process_complete() on accumulated text
    raw_text = stream_processor.get_accumulated_text()
    if ctx.tools:
        logger.debug(
            f"=== RAW MODEL OUTPUT ({len(raw_text)} chars) ===\n{raw_text}\n=== END RAW OUTPUT ==="
        )

    result = ctx.adapter.process_complete(raw_text, finish_reason)

    if result.tool_calls:
        logger.debug(f"Detected {len(result.tool_calls)} tool calls in streaming response")

    logger.info(
        f"Chat stream complete: {ctx.completion_id}, "
        f"tokens={completion_tokens}, reason={result.finish_reason}"
    )

    if LOGFIRE_AVAILABLE:
        logfire.info(
            "stream_completion_finished",
            completion_id=ctx.completion_id,
            completion_tokens=completion_tokens,
            finish_reason=result.finish_reason,
            has_tool_calls=result.tool_calls is not None,
        )

    # Yield final TextResult as terminal event
    yield result


async def _complete_chat_ir(ctx: _GenContext) -> InferenceResult:
    """Non-streaming generation returning IR result.

    Uses run_on_metal_thread for Metal GPU thread affinity.
    """
    from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

    actual_tokenizer = getattr(ctx.tokenizer, "tokenizer", ctx.tokenizer)
    prompt_tokens = len(actual_tokenizer.encode(ctx.prompt))

    if ctx.pixel_values is not None:
        # Vision path: use mlx_vlm.generate (blocking, not streaming)
        def run_vision_generation() -> tuple[str, str]:
            """Run vision generation in dedicated thread (owns Metal context)."""
            from mlx_vlm import generate as vlm_generate

            response = vlm_generate(
                ctx.model,
                ctx.tokenizer,  # This is actually the Processor for vision models
                ctx.prompt,
                ctx.pixel_values,
                max_tokens=ctx.max_tokens,
                temp=ctx.temperature,
                verbose=False,
            )
            return (str(response.text), "stop")

        response_text, finish_reason = await run_on_metal_thread(
            run_vision_generation, timeout=600.0
        )
        completion_tokens = len(response_text.split())  # Rough estimate for vision
    else:
        # Text path: existing stream_generate logic
        def run_generation() -> tuple[str, str]:
            """Run complete generation in dedicated thread (owns Metal context)."""
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            response_text = ""
            finish_reason = "length"

            sampler = make_sampler(temp=ctx.temperature, top_p=ctx.top_p)

            for response in stream_generate(
                ctx.model,
                ctx.tokenizer,
                ctx.prompt,
                max_tokens=ctx.max_tokens,
                sampler=sampler,
            ):
                token_id = getattr(response, "token", None)
                token_text = getattr(response, "text", str(response))

                if token_id is not None and token_id in ctx.stop_token_ids:
                    finish_reason = "stop"
                    break

                response_text += token_text

            return (response_text, finish_reason)

        response_text, finish_reason = await run_on_metal_thread(run_generation)
        completion_tokens = len(ctx.tokenizer.encode(response_text))

    # Post-process with adapter.process_complete()
    adapter = ctx.adapter
    if adapter is None:
        # No adapter - return raw response without parsing
        result = TextResult(content=response_text, finish_reason=finish_reason)
        return InferenceResult(
            result=result,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    result = adapter.process_complete(response_text, finish_reason)

    if result.tool_calls:
        logger.debug(f"Detected {len(result.tool_calls)} tool calls in response")
    if result.reasoning_content:
        logger.debug(f"Extracted reasoning content ({len(result.reasoning_content)} chars)")

    logger.info(
        f"Chat complete: {ctx.completion_id}, tokens={completion_tokens}, "
        f"reason={result.finish_reason}, has_tools={result.tool_calls is not None}, "
        f"has_reasoning={result.reasoning_content is not None}"
    )

    if LOGFIRE_AVAILABLE:
        logfire.info(
            "completion_finished",
            completion_id=ctx.completion_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=result.finish_reason,
            has_tool_calls=result.tool_calls is not None,
            has_reasoning=result.reasoning_content is not None,
        )

    return InferenceResult(
        result=result,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ── Legacy wrappers (old signatures, for tests + backward compat) ─


async def _stream_chat_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_ids: set[int],
    completion_id: str,
    created: int,
    model_id: str,
    adapter: Any = None,
    tools: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict, None]:
    """Legacy wrapper: yields OpenAI chunk dicts.

    Maintained for test compatibility. New code should use generate_chat_stream().
    Will be removed in Phase 6.
    """
    import json

    from mlx_manager.mlx_server.services.formatters.openai import OpenAIFormatter

    ctx = _GenContext(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
        adapter=adapter,
        model_id=model_id,
        completion_id=completion_id,
        created=created,
        tools=tools,
        pixel_values=None,
    )
    formatter = OpenAIFormatter(model_id=model_id, request_id=completion_id)
    formatter.created = created

    for sse in formatter.stream_start():
        yield json.loads(sse["data"])

    async for item in _stream_chat_ir(ctx):
        if isinstance(item, TextResult):
            for sse in formatter.stream_end(item.finish_reason, tool_calls=item.tool_calls):
                data = sse["data"]
                if data == "[DONE]":
                    break
                yield json.loads(data)
        else:
            for sse in formatter.stream_event(item):
                yield json.loads(sse["data"])


async def _generate_chat_complete(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_ids: set[int],
    completion_id: str,
    created: int,
    model_id: str,
    adapter: Any = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict:
    """Legacy wrapper: returns OpenAI ChatCompletion dict.

    Maintained for test compatibility. New code should use generate_chat_complete_response().
    Will be removed in Phase 6.
    """
    from mlx_manager.mlx_server.services.formatters.openai import OpenAIFormatter

    ctx = _GenContext(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
        adapter=adapter,
        model_id=model_id,
        completion_id=completion_id,
        created=created,
        tools=tools,
        pixel_values=None,
    )
    ir = await _complete_chat_ir(ctx)
    formatter = OpenAIFormatter(model_id=model_id, request_id=completion_id)
    formatter.created = created
    return formatter.format_complete(
        ir.result,
        prompt_tokens=ir.prompt_tokens,
        completion_tokens=ir.completion_tokens,
    )


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
    from mlx_manager.mlx_server.models.pool import get_model_pool

    # Handle list of prompts (use first for now, batch in Phase 9)
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    # Get model from pool
    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    model = loaded.model
    tokenizer = loaded.tokenizer

    # Get stop tokens from adapter
    adapter = loaded.adapter
    if adapter is not None:
        stop_tokens = set(adapter.stop_tokens)
    else:
        # Fallback: compute directly from tokenizer
        from mlx_manager.mlx_server.models.adapters.composable import (
            DefaultAdapter,
        )

        adapter = DefaultAdapter(tokenizer)
        stop_tokens = set(adapter.stop_tokens)

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

    Uses stream_from_metal_thread for Metal GPU thread affinity.
    """
    from mlx_manager.mlx_server.utils.metal import stream_from_metal_thread

    def produce_tokens() -> Iterator[tuple[str, int | None, bool]]:
        """Yield (token_text, token_id, is_stop) tuples on Metal thread."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p)

        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            token_id = getattr(response, "token", None)
            token_text = getattr(response, "text", str(response))

            is_stop = token_id is not None and token_id in stop_tokens
            yield (token_text, token_id, is_stop)

            if is_stop:
                return

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

    async for token_text, token_id, is_stop in stream_from_metal_thread(produce_tokens):
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

    Uses run_on_metal_thread for Metal GPU thread affinity.
    """
    from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

    # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    prompt_tokens = len(actual_tokenizer.encode(prompt))

    def run_generation() -> tuple[str, str]:
        """Run complete generation in dedicated thread (owns Metal context)."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        response_text = ""
        finish_reason = "length"

        # Create sampler with temperature and top_p settings
        sampler = make_sampler(temp=temperature, top_p=top_p)

        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            token_id = getattr(response, "token", None)
            token_text = getattr(response, "text", str(response))

            if token_id is not None and token_id in stop_tokens:
                finish_reason = "stop"
                break

            response_text += token_text

        return (response_text, finish_reason)

    response_text, finish_reason = await run_on_metal_thread(run_generation)

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
