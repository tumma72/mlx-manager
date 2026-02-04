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
from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.mlx_server.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FunctionCall,
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
from mlx_manager.mlx_server.services.image_processor import preprocess_images
from mlx_manager.mlx_server.services.inference import generate_chat_completion
from mlx_manager.mlx_server.services.structured_output import (
    StructuredOutputValidator,
)
from mlx_manager.mlx_server.services.vision import generate_vision_completion

router = APIRouter(tags=["chat"])

# Module-level validator instance
_structured_output_validator = StructuredOutputValidator()


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
        _, images = extract_content_parts(message.content)
        if message.role == "user":
            all_image_urls.extend(images)

    has_images = len(all_image_urls) > 0

    # Check model type - vision models must use vision path even for text-only
    model_type = detect_model_type(request.model)
    is_vision_model = model_type == ModelType.VISION

    async with audit_service.track_request(
        request_id=request_id,
        model=request.model,
        endpoint="/v1/chat/completions",
        backend_type="local",
    ) as audit_ctx:
        try:
            if has_images or is_vision_model:
                # Vision model or multimodal request - use vision path
                # Vision models use Processor (not Tokenizer) and require mlx_vlm
                result = await _handle_vision_request(request, all_image_urls)
            else:
                # Text-only request with text model - use mlx_lm path
                result = await _handle_text_request(request)

            # Update audit context with usage if available (non-streaming)
            if isinstance(result, ChatCompletionResponse) and result.usage:
                audit_ctx.prompt_tokens = result.usage.prompt_tokens
                audit_ctx.completion_tokens = result.usage.completion_tokens
                audit_ctx.total_tokens = result.usage.total_tokens

            return result
        except HTTPException:
            raise
        except RuntimeError as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")


async def _handle_vision_request(
    request: ChatCompletionRequest,
    image_urls: list[str],
) -> EventSourceResponse | ChatCompletionResponse:
    """Handle multimodal request with images."""
    # Check model type before loading
    model_type = detect_model_type(request.model)
    if model_type != ModelType.VISION:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is type '{model_type.value}', "
            f"but request contains images. Use a vision model (e.g., Qwen2-VL).",
        )

    # Preprocess images
    images = await preprocess_images(image_urls)

    # Build text prompt from messages
    # For vision, combine system + user messages into single prompt
    prompt_parts = []
    for message in request.messages:
        text, _ = extract_content_parts(message.content)
        if message.role == "system":
            prompt_parts.append(f"System: {text}")
        elif message.role == "user":
            prompt_parts.append(f"User: {text}")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {text}")
    text_prompt = "\n".join(prompt_parts)

    # Generate vision completion with timeout
    settings = get_settings()
    timeout = settings.timeout_chat_seconds

    try:
        result = await asyncio.wait_for(
            generate_vision_completion(
                model_id=request.model,
                text_prompt=text_prompt,
                images=images,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                stream=request.stream,
            ),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning(f"Vision completion timed out after {timeout}s")
        raise TimeoutHTTPException(
            timeout_seconds=timeout,
            detail=f"Vision completion timed out after {int(timeout)} seconds. "
            f"Consider using a smaller model or fewer images.",
        )

    if request.stream:
        # result is an async generator
        async def event_generator() -> Any:
            async for chunk in result:  # type: ignore[union-attr]
                yield {"data": json.dumps(chunk)}
            yield {"data": "[DONE]"}

        return EventSourceResponse(event_generator())
    else:
        # result is a dict
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


async def _handle_text_request(
    request: ChatCompletionRequest,
) -> EventSourceResponse | ChatCompletionResponse:
    """Handle text-only request.

    Routes through cloud router if enabled, batching scheduler if enabled,
    otherwise uses direct inference.
    """
    settings = get_settings()

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
    return await _handle_direct_request(request)


async def _handle_routed_request(
    request: ChatCompletionRequest,
) -> EventSourceResponse | ChatCompletionResponse:
    """Handle text request via backend router.

    Uses BackendRouter to lookup model-to-backend mapping and route
    appropriately. Handles cloud failover when local inference fails.
    """
    # Convert messages to dict format
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        if isinstance(m.content, str):
            text = m.content
        else:
            text, _ = extract_content_parts(m.content)
        messages.append({"role": m.role, "content": text})

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
) -> EventSourceResponse | ChatCompletionResponse:
    """Handle text request via direct inference (non-batched).

    Supports:
    - Tool calling: Passes tools to inference service
    - Structured output: Validates response against JSON schema
    """
    # Convert messages to dict format, extracting text from content blocks
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        if isinstance(m.content, str):
            text = m.content
        else:
            text, _ = extract_content_parts(m.content)
        messages.append({"role": m.role, "content": text})

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
        return await _handle_streaming(request, messages, stop, tools)
    else:
        return await _handle_non_streaming(request, messages, stop, tools)


async def _handle_streaming(
    request: ChatCompletionRequest,
    messages: list[dict[str, Any]],
    stop: list[str] | None,
    tools: list[dict[str, Any]] | None = None,
) -> EventSourceResponse:
    """Handle streaming response with timeout.

    Supports tool calling in streaming mode - tool calls are detected
    and included in the final chunk.
    """
    settings = get_settings()
    timeout = settings.timeout_chat_seconds

    async def event_generator() -> Any:
        try:
            # Apply timeout to the entire streaming operation
            gen = await asyncio.wait_for(
                generate_chat_completion(
                    model_id=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens or 4096,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=stop,
                    stream=True,
                    tools=tools,
                ),
                timeout=timeout,
            )
            async for chunk in gen:  # type: ignore[union-attr]
                # Format as SSE data
                yield {"data": json.dumps(chunk)}

            # Final [DONE] message
            yield {"data": "[DONE]"}
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
) -> ChatCompletionResponse:
    """Handle non-streaming response.

    Supports:
    - Tool calling: tool_calls included in response message
    - Reasoning: reasoning_content included in response message
    - Structured output: validates response against JSON schema
    """
    settings = get_settings()
    timeout = settings.timeout_chat_seconds

    try:
        result = await asyncio.wait_for(
            generate_chat_completion(
                model_id=request.model,
                messages=messages,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
                stream=False,
                tools=tools,
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

    # Cast to dict since we passed stream=False
    result_dict = cast(dict[str, Any], result)
    choice = result_dict["choices"][0]
    msg = choice["message"]

    # Validate structured output if json_schema is specified
    content = msg.get("content", "")
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

    # Build ChatMessage with optional tool_calls and reasoning_content
    chat_message = ChatMessage(
        role=msg["role"],
        content=content,
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
    """Convert tool calls dict to Pydantic ToolCall objects.

    Args:
        tool_calls: List of tool call dicts from inference service

    Returns:
        List of ToolCall objects, or None if no tool calls
    """
    if not tool_calls:
        return None

    result: list[ToolCall] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        result.append(
            ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=FunctionCall(
                    name=func.get("name", ""),
                    arguments=func.get("arguments", "{}"),
                ),
            )
        )
    return result


# --- Batched Request Handlers ---


async def _handle_batched_request(
    request: ChatCompletionRequest,
    priority: Priority,
) -> EventSourceResponse | ChatCompletionResponse:
    """Route request through batching scheduler.

    Tokenizes the prompt, creates a BatchRequest, and submits to the
    scheduler for batched processing.
    """
    from mlx_manager.mlx_server.models.adapters import get_adapter
    from mlx_manager.mlx_server.models.pool import get_model_pool

    pool = get_model_pool()
    loaded = await pool.get_model(request.model)
    adapter = get_adapter(request.model)

    # Get actual tokenizer (handle Processor wrapper for vision models)
    actual_tokenizer = getattr(loaded.tokenizer, "tokenizer", loaded.tokenizer)

    # Build messages for chat template
    messages: list[dict[str, str]] = []
    for m in request.messages:
        if isinstance(m.content, str):
            text = m.content
        else:
            text, _ = extract_content_parts(m.content)
        messages.append({"role": m.role, "content": text})

    # Apply chat template to get prompt string
    prompt = adapter.apply_chat_template(loaded.tokenizer, messages, add_generation_prompt=True)

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
