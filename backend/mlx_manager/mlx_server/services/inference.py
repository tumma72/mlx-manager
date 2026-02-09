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

try:  # pragma: no cover
    import logfire  # pragma: no cover

    LOGFIRE_AVAILABLE = True  # pragma: no cover
except ImportError:  # pragma: no cover
    LOGFIRE_AVAILABLE = False  # pragma: no cover


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
        tools: Optional list of tool definitions in OpenAI format

    Yields/Returns:
        Streaming: yields chunk dicts
        Non-streaming: returns complete response dict with optional tool_calls
                      and reasoning_content in the message
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool

    # Get model from pool
    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    model = loaded.model
    tokenizer = loaded.tokenizer

    # Get adapter from loaded model
    adapter = loaded.adapter
    if adapter is None:
        # Fallback for non-TEXT_GEN models or edge cases
        from mlx_manager.mlx_server.models.adapters.composable import (
            DefaultAdapter,
        )

        adapter = DefaultAdapter(tokenizer)

    # Convert messages to model-specific format (handles tool messages, etc.)
    converted_messages = adapter.convert_messages(messages)

    # Build messages with tool definitions if provided
    effective_messages = converted_messages
    use_native_tools = False

    if tools and adapter.supports_tool_calling():
        loaded_caps = getattr(loaded, "capabilities", None)
        if loaded_caps and loaded_caps.supports_native_tools:
            use_native_tools = True
            logger.debug(f"Using native tool support for {model_id}")
        elif enable_prompt_injection:
            tool_prompt = adapter.format_tools_for_prompt(tools)
            if tool_prompt:
                effective_messages = _inject_tools_into_messages(converted_messages, tool_prompt)
                logger.debug(f"Injected tools via prompt injection for {model_id}")
        else:
            logger.info(
                f"Tools requested but model {model_id} has no native support "
                "and prompt injection not enabled"
            )

    # Apply chat template
    prompt: str = adapter.apply_chat_template(
        messages=effective_messages,
        add_generation_prompt=True,
        tools=tools if use_native_tools else None,
    )

    # Debug: Log the final prompt when tools are present
    if tools:
        logger.debug(f"=== PROMPT WITH TOOLS (last 1500 chars) ===\n{prompt[-1500:]}")

    # CRITICAL: Get stop token IDs from adapter
    # Llama 3.x requires BOTH eos_token_id AND <|eot_id|> (end of turn)
    # Without this, models will generate past the assistant's response
    stop_token_ids: set[int] = set(adapter.stop_tokens)

    # Add tool-specific stop tokens when tools are enabled
    if tools and adapter.supports_tool_calling():
        tool_stop_tokens = adapter.get_tool_call_stop_tokens()
        stop_token_ids.update(tool_stop_tokens)

    logger.debug(f"Stop token IDs for {model_id}: {stop_token_ids}")

    # Generate unique ID for this completion
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    logger.info(
        f"Starting generation: {completion_id}, model={model_id}, "
        f"max_tokens={max_tokens}, tools={'yes' if tools else 'no'}"
    )

    # LogFire span for observability
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
                adapter=adapter,
                tools=tools,
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
                adapter=adapter,
                tools=tools,
            )
    finally:
        if span_context:
            span_context.__exit__(None, None, None)


def _inject_tools_into_messages(
    messages: list[dict[str, Any]], tool_prompt: str
) -> list[dict[str, Any]]:
    """Inject tool definitions into the message list.

    Appends tool definitions to the system message if one exists,
    otherwise creates a new system message at the beginning.

    Args:
        messages: Original message list
        tool_prompt: Formatted tool definitions string

    Returns:
        New message list with tool definitions injected
    """
    result = list(messages)  # Make a copy

    # Find system message
    system_idx = None
    for i, msg in enumerate(result):
        if msg.get("role") == "system":
            system_idx = i
            break

    if system_idx is not None:
        # Append to existing system message
        existing_content = result[system_idx].get("content", "")
        result[system_idx] = {
            **result[system_idx],
            "content": f"{existing_content}\n\n{tool_prompt}",
        }
    else:
        # Create new system message at the beginning
        result.insert(0, {"role": "system", "content": tool_prompt})

    return result


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
    adapter: Any = None,
    tools: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict, None]:
    """Generate tokens with streaming for chat completion.

    Uses stream_from_metal_thread to run MLX generation on a dedicated thread
    (required for Metal GPU thread affinity). Tokens are yielded as
    (token_text, token_id, is_stop) tuples.

    Uses StreamingProcessor to filter patterns during generation:
    - Tool call markers (<tool_call>, <function=>) never reach the client
    - Thinking tags (<think>, etc.) filtered from stream
    - Final processing extracts tool_calls and reasoning from accumulated text
    """
    from mlx_manager.mlx_server.services.response_processor import StreamingProcessor
    from mlx_manager.mlx_server.utils.metal import stream_from_metal_thread

    completion_tokens = 0

    # Check if prompt already ends with a thinking start tag (e.g., GLM-4.7)
    # In this case, the model's output continues inside the thinking pattern
    thinking_starts = ["<think>", "<thinking>", "<reasoning>", "<reflection>"]
    starts_in_thinking = any(prompt.rstrip().endswith(tag) for tag in thinking_starts)
    if starts_in_thinking:
        logger.debug("Prompt ends with thinking tag, starting in thinking mode")

    # Use adapter for streaming (provides parsers and markers)
    stream_processor = StreamingProcessor(
        adapter=adapter,
        starts_in_thinking=starts_in_thinking,
    )  # Filters patterns during streaming

    # Debug: Log configuration for this generation
    if tools:
        logger.debug(
            f"Generation config: model={model_id}, max_tokens={max_tokens}, "
            f"stop_tokens={stop_token_ids}, tools_count={len(tools)}, "
            f"starts_in_thinking={starts_in_thinking}"
        )

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

            # CRITICAL: Check for stop token
            is_stop = token_id is not None and token_id in stop_token_ids
            yield (token_text, token_id, is_stop)

            if is_stop:
                return

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

    async for token_text, token_id, is_stop in stream_from_metal_thread(produce_tokens):
        completion_tokens += 1

        if is_stop:
            # Stop token found - don't yield the stop token text
            finish_reason = "stop"
            # Feed to processor but don't yield (updates accumulated text)
            stream_processor.feed(token_text)
            logger.debug(
                f"Stop token encountered: token_id={token_id}, "
                f"token_text={repr(token_text)}, tokens_so_far={completion_tokens}"
            )
            break

        # Process token through StreamingProcessor
        # Returns StreamEvent with reasoning_content or content
        event = stream_processor.feed(token_text)
        if event.reasoning_content or event.content:
            delta: dict[str, Any] = {}
            if event.reasoning_content:
                delta["reasoning_content"] = event.reasoning_content
            if event.content:
                delta["content"] = event.content
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }

    # Debug: Log raw accumulated text before finalization
    raw_text = stream_processor.get_accumulated_text()
    if tools:
        logger.debug(
            f"=== RAW MODEL OUTPUT ({len(raw_text)} chars) ===\n{raw_text}\n=== END RAW OUTPUT ==="
        )

    # Finalize: Use adapter's tool parser for finalization
    tool_calls_list = adapter.tool_parser.extract(raw_text)
    tool_calls = None
    if tool_calls_list:
        tool_calls = [tc.model_dump() for tc in tool_calls_list]
        finish_reason = "tool_calls"
        logger.debug(f"Detected {len(tool_calls)} tool calls in streaming response")

    # Build final chunk with finish_reason and optional tool_calls
    final_delta: dict[str, Any] = {}
    if tool_calls:
        # Include tool_calls in final delta
        final_delta["tool_calls"] = [
            {
                "index": i,
                "id": tc["id"],
                "type": tc["type"],
                "function": tc["function"],
            }
            for i, tc in enumerate(tool_calls)
        ]

    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": final_delta,
                "finish_reason": finish_reason,
            }
        ],
    }

    logger.info(
        f"Chat stream complete: {completion_id}, tokens={completion_tokens}, reason={finish_reason}"
    )

    if LOGFIRE_AVAILABLE:
        logfire.info(
            "stream_completion_finished",
            completion_id=completion_id,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            has_tool_calls=tool_calls is not None,
        )


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
    adapter: Any = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict:
    """Generate complete response (non-streaming) for chat completion.

    Uses run_on_metal_thread for Metal GPU thread affinity.

    Post-processes the response to:
    - Parse tool calls if tools are enabled and adapter supports it
    - Extract reasoning content if adapter supports reasoning mode
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

            # CRITICAL: Check for stop token
            if token_id is not None and token_id in stop_token_ids:
                finish_reason = "stop"
                break

            response_text += token_text

        return (response_text, finish_reason)

    response_text, finish_reason = await run_on_metal_thread(run_generation)
    completion_tokens = len(tokenizer.encode(response_text))

    # Post-process: Use adapter's parsers if adapter is provided
    if adapter is not None:
        tool_calls_list = adapter.tool_parser.extract(response_text)
        reasoning_content = adapter.thinking_parser.extract(response_text)
        final_content = response_text
        if reasoning_content:
            final_content = adapter.thinking_parser.remove(final_content)
        final_content = adapter.clean_response(final_content)
    else:
        # No adapter - no parsing
        tool_calls_list = []
        reasoning_content = None
        final_content = response_text

    # Convert tool calls to dicts
    tool_calls = None
    if tool_calls_list:
        tool_calls = [tc.model_dump() for tc in tool_calls_list]
        finish_reason = "tool_calls"
        logger.debug(f"Detected {len(tool_calls)} tool calls in response")

    if reasoning_content:
        logger.debug(f"Extracted reasoning content ({len(reasoning_content)} chars)")

    logger.info(
        f"Chat complete: {completion_id}, tokens={completion_tokens}, "
        f"reason={finish_reason}, has_tools={tool_calls is not None}, "
        f"has_reasoning={reasoning_content is not None}"
    )

    if LOGFIRE_AVAILABLE:
        logfire.info(
            "completion_finished",
            completion_id=completion_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=finish_reason,
            has_tool_calls=tool_calls is not None,
            has_reasoning=reasoning_content is not None,
        )

    # Build message dict
    message: dict[str, Any] = {
        "role": "assistant",
        "content": final_content,
    }

    # Add tool_calls to message if present
    if tool_calls:
        message["tool_calls"] = tool_calls

    # Add reasoning_content to message if present
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


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
