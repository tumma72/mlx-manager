"""Chat completions endpoint."""

import asyncio
import json
import time
import uuid
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.config import get_settings
from mlx_manager.mlx_server.errors import TimeoutHTTPException
from mlx_manager.mlx_server.models.detection import detect_model_type
from mlx_manager.mlx_server.models.ir import TextResult
from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.mlx_server.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ToolCall,
    Usage,
    extract_content_parts,
)
from mlx_manager.mlx_server.services.audit import audit_service
from mlx_manager.mlx_server.services.batching import (
    BatchRequest,
    ContinuousBatchingScheduler,
    Priority,
    get_scheduler_manager,
)
from mlx_manager.mlx_server.services.cloud.router import get_router
from mlx_manager.mlx_server.services.formatters import OpenAIFormatter
from mlx_manager.mlx_server.services.image_processor import preprocess_images
from mlx_manager.mlx_server.services.inference import (
    generate_chat_complete_response,
    generate_chat_stream,
)
from mlx_manager.mlx_server.services.structured_output import (
    StructuredOutputValidator,
)

router = APIRouter(tags=["chat"])

# Module-level validator instance
_structured_output_validator = StructuredOutputValidator()


def _convert_messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessage objects to dicts preserving ALL fields.

    This is the single conversion point for ChatMessage -> dict format.
    Preserves tool_calls, tool_call_id, and name fields per P3 (no data loss).

    Args:
        messages: List of ChatMessage Pydantic objects

    Returns:
        List of dicts with role, content, and optional tool fields
    """
    result: list[dict[str, Any]] = []
    for m in messages:
        # Extract text from content blocks (for multimodal support)
        if isinstance(m.content, str):
            text: str | None = m.content
        elif m.content is not None:
            text, _ = extract_content_parts(m.content)
        else:
            text = m.content  # None for assistant messages with only tool_calls

        msg: dict[str, Any] = {"role": m.role, "content": text}

        # Preserve tool calling fields (P3: no data loss through layers)
        if m.tool_calls:
            msg["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id

        result.append(msg)
    return result


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> EventSourceResponse | ChatCompletionResponse:
    """Create a chat completion.

    Supports both streaming and non-streaming responses.
    Supports both text-only and multimodal (vision) requests.
    Compatible with OpenAI Chat Completions API.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    logger.info(f"Chat completion request: model={request.model}, stream={request.stream}")

    # Extract text and images from all user messages
    all_image_urls: list[str] = []

    for message in request.messages:
        if message.content is None:
            continue
        _, images = extract_content_parts(message.content)
        if message.role == "user":
            all_image_urls.extend(images)

    has_images = len(all_image_urls) > 0

    # Validate model type if images are present
    if has_images:
        model_type = detect_model_type(request.model)
        if model_type != ModelType.VISION:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is type '{model_type.value}', "
                f"but request contains images. Use a vision model (e.g., Qwen2-VL).",
            )

    async with audit_service.track_request(
        request_id=request_id,
        model=request.model,
        endpoint="/v1/chat/completions",
        backend_type="local",
    ) as audit_ctx:
        try:
            # Unified text and vision path
            # Images (if present) will be handled by the adapter's prepare_input()
            result = await _handle_text_request(request, all_image_urls)

            # Update audit context with usage if available (non-streaming)
            if isinstance(result, ChatCompletionResponse) and result.usage:
                audit_ctx.prompt_tokens = result.usage.prompt_tokens
                audit_ctx.completion_tokens = result.usage.completion_tokens
                audit_ctx.total_tokens = result.usage.total_tokens

            return result
        except HTTPException:
            raise
        except RuntimeError as e:
            logger.exception(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")


async def _handle_text_request(
    request: ChatCompletionRequest,
    image_urls: list[str] | None = None,
) -> EventSourceResponse | ChatCompletionResponse:
    """Handle text and vision requests.

    Routes through cloud router if enabled (text only), batching scheduler if enabled (text only),
    otherwise uses direct inference. Vision models always go through the direct path.

    Note: Cloud routing and batching only work with text models. Vision models require
    direct inference regardless of settings.
    """
    settings = get_settings()

    # Preprocess images if present
    images = None
    if image_urls:
        images = await preprocess_images(image_urls)

    # Vision models always use direct path (cloud/batching don't support vision)
    has_images = images is not None
    if has_images:
        return await _handle_direct_request(request, images)

    # Try cloud routing path if enabled (checks mappings and handles failover)
    if settings.enable_cloud_routing:
        try:
            return await _handle_routed_request(request)
        except Exception as e:
            logger.warning(f"Cloud routing failed, falling back to local: {e}")
            # Fall through to direct/batched path

    # Try batching path if enabled
    if settings.enable_batching:
        try:
            mgr = get_scheduler_manager()
            priority = mgr.get_priority_for_request(
                api_key=None,  # TODO: Extract from request headers
                endpoint="/v1/chat/completions",
            )
            return await _handle_batched_request(request, priority)
        except RuntimeError:
            # Scheduler not initialized - fall through to direct
            logger.warning("Batching enabled but scheduler not initialized, using direct")
        except Exception as e:
            # Other batching error - fall through to direct
            logger.warning(f"Batching unavailable, falling back to direct: {e}")

    # Direct inference path
    return await _handle_direct_request(request, images)


async def _handle_routed_request(
    request: ChatCompletionRequest,
) -> EventSourceResponse | ChatCompletionResponse:
    """Handle text request via backend router.

    Uses BackendRouter to lookup model-to-backend mapping and route
    appropriately. Handles cloud failover when local inference fails.
    """
    # Convert messages preserving all fields (tool_calls, tool_call_id)
    messages = _convert_messages_to_dicts(request.messages)

    # Get router singleton and route request with timeout
    settings = get_settings()
    timeout = settings.timeout_chat_seconds
    backend_router = get_router()

    try:
        result = await asyncio.wait_for(
            backend_router.route_request(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                stream=request.stream,
                top_p=request.top_p,
            ),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning(f"Routed chat completion timed out after {timeout}s")
        raise TimeoutHTTPException(
            timeout_seconds=timeout,
            detail=f"Chat completion timed out after {int(timeout)} seconds. "
            f"The backend may be overloaded or the request too large.",
        )

    if request.stream:
        # result is an async generator
        async def event_generator() -> Any:
            async for chunk in result:  # type: ignore[union-attr]
                yield {"data": json.dumps(chunk)}
            yield {"data": "[DONE]"}

        return EventSourceResponse(event_generator())
    else:
        # result is a dict - convert to Pydantic response model
        result_dict = cast(dict[str, Any], result)
        choice = result_dict["choices"][0]
        return ChatCompletionResponse(
            id=result_dict["id"],
            created=result_dict["created"],
            model=result_dict["model"],
            choices=[
                ChatCompletionChoice(
                    index=choice["index"],
                    message=ChatMessage(
                        role=choice["message"]["role"],
                        content=choice["message"]["content"],
                    ),
                    finish_reason=choice["finish_reason"],
                )
            ],
            usage=Usage(
                prompt_tokens=result_dict["usage"]["prompt_tokens"],
                completion_tokens=result_dict["usage"]["completion_tokens"],
                total_tokens=result_dict["usage"]["total_tokens"],
            ),
        )


async def _handle_direct_request(
    request: ChatCompletionRequest,
    images: list[Any] | None = None,
) -> EventSourceResponse | ChatCompletionResponse:
    """Handle text and vision requests via direct inference (non-batched).

    Supports:
    - Vision: Images passed to inference service for multimodal models
    - Tool calling: Passes tools to inference service
    - Structured output: Validates response against JSON schema
    """
    # Convert messages preserving all fields (tool_calls, tool_call_id)
    messages = _convert_messages_to_dicts(request.messages)

    # Handle stop parameter (can be string or list)
    stop: list[str] | None = (
        request.stop
        if isinstance(request.stop, list)
        else ([request.stop] if request.stop else None)
    )

    # Convert tools to dict format if present (and tool_choice is not "none")
    tools: list[dict[str, Any]] | None = None
    if request.tools and request.tool_choice != "none":
        tools = [tool.model_dump() for tool in request.tools]
        logger.debug(f"Passing {len(tools)} tools to inference")

    if request.stream:
        return await _handle_streaming(request, messages, stop, tools, images)
    else:
        return await _handle_non_streaming(request, messages, stop, tools, images)


async def _handle_streaming(
    request: ChatCompletionRequest,
    messages: list[dict[str, Any]],
    stop: list[str] | None,
    tools: list[dict[str, Any]] | None = None,
    images: list[Any] | None = None,
) -> EventSourceResponse:
    """Handle streaming response with timeout.

    Uses the 3-layer adapter pipeline: inference yields IR StreamEvents,
    OpenAIFormatter converts them to OpenAI Chat Completion chunk SSE dicts.
    """
    settings = get_settings()
    timeout = settings.timeout_chat_seconds

    async def event_generator() -> Any:
        try:
            formatter = OpenAIFormatter(
                model_id=request.model,
                request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            )

            # Emit initial role chunk
            for sse in formatter.stream_start():
                yield sse

            # Apply timeout to preparation (model loading, template application)
            gen = await asyncio.wait_for(
                generate_chat_stream(
                    model_id=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens or 4096,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=stop,
                    tools=tools,
                    images=images,
                ),
                timeout=timeout,
            )

            output_tokens = 0
            async for item in gen:
                if isinstance(item, TextResult):
                    # Final result — emit stream_end with finish reason + tool calls
                    for sse in formatter.stream_end(
                        item.finish_reason,
                        tool_calls=item.tool_calls,
                        output_tokens=output_tokens,
                    ):
                        yield sse
                else:
                    # StreamEvent — emit content delta
                    output_tokens += 1
                    for sse in formatter.stream_event(item):
                        yield sse

        except TimeoutError:
            logger.warning(f"Streaming chat completion timed out after {timeout}s")
            # Send error event before closing (per CONTEXT.md streaming errors)
            error_event = {
                "error": {
                    "type": "https://mlx-manager.dev/errors/timeout",
                    "message": f"Request timed out after {int(timeout)} seconds",
                }
            }
            yield {"event": "error", "data": json.dumps(error_event)}

    return EventSourceResponse(event_generator())


async def _handle_non_streaming(
    request: ChatCompletionRequest,
    messages: list[dict[str, Any]],
    stop: list[str] | None,
    tools: list[dict[str, Any]] | None = None,
    images: list[Any] | None = None,
) -> ChatCompletionResponse:
    """Handle non-streaming response.

    Uses the 3-layer adapter pipeline: inference returns IR TextResult,
    OpenAIFormatter converts it to an OpenAI ChatCompletion response dict.

    Supports:
    - Vision: Images passed to inference service for multimodal models
    - Tool calling: tool_calls included in response message
    - Reasoning: reasoning_content included in response message
    - Structured output: validates response against JSON schema
    """
    settings = get_settings()
    timeout = settings.timeout_chat_seconds

    try:
        inference_result = await asyncio.wait_for(
            generate_chat_complete_response(
                model_id=request.model,
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
                tools=tools,
                images=images,
            ),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning(f"Chat completion timed out after {timeout}s")
        raise TimeoutHTTPException(
            timeout_seconds=timeout,
            detail=f"Chat completion timed out after {int(timeout)} seconds. "
            f"Consider using a smaller model or reducing max_tokens.",
        )

    text_result = inference_result.result

    # Validate structured output if json_schema is specified
    content = text_result.content
    if (
        request.response_format
        and request.response_format.type == "json_schema"
        and request.response_format.json_schema
    ):
        schema = request.response_format.json_schema
        validation_result = _structured_output_validator.validate_and_coerce(content, schema)
        if not validation_result.success:
            logger.warning(f"Structured output validation failed: {validation_result.error}")
            raise HTTPException(
                status_code=400,
                detail=f"Model output failed JSON schema validation: "
                f"{validation_result.error} at path {validation_result.error_path}",
            )
        logger.debug("Structured output validation passed")

    # Format response using OpenAIFormatter
    formatter = OpenAIFormatter(
        model_id=request.model,
        request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
    )
    result_dict = formatter.format_complete(
        text_result,
        prompt_tokens=inference_result.prompt_tokens,
        completion_tokens=inference_result.completion_tokens,
    )

    # Convert formatter dict to Pydantic response model
    choice = result_dict["choices"][0]
    msg = choice["message"]

    chat_message = ChatMessage(
        role=msg["role"],
        content=msg.get("content", ""),
        tool_calls=_convert_tool_calls(msg.get("tool_calls")),
        reasoning_content=msg.get("reasoning_content"),
    )

    return ChatCompletionResponse(
        id=result_dict["id"],
        created=result_dict["created"],
        model=result_dict["model"],
        choices=[
            ChatCompletionChoice(
                index=choice["index"],
                message=chat_message,
                finish_reason=choice["finish_reason"],
            )
        ],
        usage=Usage(
            prompt_tokens=result_dict["usage"]["prompt_tokens"],
            completion_tokens=result_dict["usage"]["completion_tokens"],
            total_tokens=result_dict["usage"]["total_tokens"],
        ),
    )


def _convert_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[ToolCall] | None:
    """Convert tool calls dicts to canonical ToolCall Pydantic objects.

    The inference service returns tool calls as dicts (from ToolCall.model_dump()),
    so we use model_validate to reconstruct the canonical type without manual
    field-by-field bridging.

    Args:
        tool_calls: List of tool call dicts from inference service

    Returns:
        List of ToolCall objects, or None if no tool calls
    """
    if not tool_calls:
        return None

    return [ToolCall.model_validate(tc) for tc in tool_calls]


# --- Batched Request Handlers ---


async def _handle_batched_request(
    request: ChatCompletionRequest,
    priority: Priority,
) -> EventSourceResponse | ChatCompletionResponse:
    """Route request through batching scheduler.

    Tokenizes the prompt, creates a BatchRequest, and submits to the
    scheduler for batched processing.
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool

    pool = get_model_pool()
    loaded = await pool.get_model(request.model)
    adapter = loaded.adapter
    if adapter is None:
        # Fallback for edge cases
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        adapter = create_adapter("default", loaded.tokenizer)

    # Get actual tokenizer (handle Processor wrapper for vision models)
    actual_tokenizer = getattr(loaded.tokenizer, "tokenizer", loaded.tokenizer)

    # Build messages for chat template (preserving all fields)
    messages = _convert_messages_to_dicts(request.messages)

    # Apply chat template to get prompt string
    prompt = adapter.apply_chat_template(messages, add_generation_prompt=True)

    # Tokenize prompt
    prompt_tokens = actual_tokenizer.encode(prompt)

    # Create batch request with unique ID
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    batch_request = BatchRequest(
        request_id=request_id,
        model_id=request.model,
        prompt_tokens=prompt_tokens,
        max_tokens=request.max_tokens or 4096,
        priority=priority,
    )

    # Get scheduler and configure
    mgr = get_scheduler_manager()
    scheduler = await mgr.get_scheduler(request.model)
    await mgr.configure_scheduler(request.model, loaded.model, loaded.tokenizer, adapter)

    # Route based on streaming preference
    if request.stream:
        return await _stream_batched_response(batch_request, scheduler, request.model)
    else:
        return await _complete_batched_response(batch_request, scheduler, request.model)


async def _stream_batched_response(
    batch_request: BatchRequest,
    scheduler: ContinuousBatchingScheduler,
    model: str,
) -> EventSourceResponse:
    """Stream tokens from scheduler as SSE events in OpenAI format."""
    settings = get_settings()
    timeout = settings.timeout_chat_seconds
    created = int(time.time())

    async def generate_stream() -> Any:
        try:
            # Submit to scheduler - returns async generator of token dicts
            token_stream = scheduler.submit(batch_request)

            # Wrap streaming in timeout using async_timeout pattern
            start_time = time.monotonic()
            async for token_data in token_stream:
                # Check elapsed time
                if time.monotonic() - start_time > timeout:
                    raise TimeoutError()

                # Token data from scheduler contains token_id, text, request_id
                token_text = token_data.get("text", "")

                # Format as OpenAI ChatCompletionChunk
                chunk = {
                    "id": batch_request.request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield {"data": json.dumps(chunk)}

            # Send final chunk with finish_reason
            final_chunk = {
                "id": batch_request.request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield {"data": json.dumps(final_chunk)}
            yield {"data": "[DONE]"}

        except TimeoutError:
            logger.warning(f"Streaming batched completion timed out after {timeout}s")
            # Send error event before closing
            error_event = {
                "error": {
                    "type": "https://mlx-manager.dev/errors/timeout",
                    "message": f"Request timed out after {int(timeout)} seconds",
                }
            }
            yield {"event": "error", "data": json.dumps(error_event)}

    return EventSourceResponse(generate_stream())


async def _complete_batched_response(
    batch_request: BatchRequest,
    scheduler: ContinuousBatchingScheduler,
    model: str,
) -> ChatCompletionResponse:
    """Collect all tokens and return complete ChatCompletionResponse."""
    settings = get_settings()
    timeout = settings.timeout_chat_seconds
    created = int(time.time())
    collected_tokens: list[str] = []

    async def collect_tokens() -> None:
        """Collect all tokens from scheduler."""
        token_stream = scheduler.submit(batch_request)
        async for token_data in token_stream:
            token_text = token_data.get("text", "")
            collected_tokens.append(token_text)

    try:
        await asyncio.wait_for(collect_tokens(), timeout=timeout)
    except TimeoutError:
        logger.warning(f"Batched chat completion timed out after {timeout}s")
        raise TimeoutHTTPException(
            timeout_seconds=timeout,
            detail=f"Chat completion timed out after {int(timeout)} seconds. "
            f"Consider reducing max_tokens or batch load.",
        )

    # Build response text
    response_text = "".join(collected_tokens)

    # Build usage statistics
    prompt_tokens = len(batch_request.prompt_tokens)
    completion_tokens = len(collected_tokens)

    return ChatCompletionResponse(
        id=batch_request.request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
