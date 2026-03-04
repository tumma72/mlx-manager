"""Chat completions endpoint."""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.config import get_settings
from mlx_manager.mlx_server.errors import ProblemDetail, TimeoutProblem
from mlx_manager.mlx_server.models.detection import detect_model_type
from mlx_manager.mlx_server.models.ir import (
    InferenceResult,
    InternalRequest,
    TextResult,
)
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
from mlx_manager.mlx_server.utils.request_helpers import (
    timeout_error_event,
    validate_model_available,
    with_inference_timeout,
)
from mlx_manager.mlx_server.utils.validation import validate_image_url

router = APIRouter(tags=["chat"])

# Module-level validator instance
_structured_output_validator = StructuredOutputValidator()


@router.post(
    "/chat/completions",
    response_model=None,
    responses={
        200: {
            "description": (
                "Chat completion response. "
                "Returns a JSON object (ChatCompletionResponse) when stream=false, "
                "or an SSE stream of ChatCompletionChunk events "
                "followed by [DONE] when stream=true."
            )
        },
        422: {"model": ProblemDetail, "description": "Validation Error"},
        404: {"model": ProblemDetail, "description": "Model Not Found"},
        408: {"model": TimeoutProblem, "description": "Request Timeout"},
        500: {"model": ProblemDetail, "description": "Internal Server Error"},
    },
)
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

    # Validate model is available
    request.model = validate_model_available(request.model)

    # Extract text and images from all user messages
    all_image_urls: list[str] = []

    for message in request.messages:
        if message.content is None:
            continue
        _, images = extract_content_parts(message.content)
        if message.role == "user":
            all_image_urls.extend(images)

    # Validate image URLs (MIME type and size for data URLs)
    for url in all_image_urls:
        validate_image_url(url)

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

    Creates protocol-neutral IR early, then routes through cloud router
    (passthrough when no rules match), batching scheduler if enabled,
    or direct inference.
    """
    settings = get_settings()

    # Create protocol-neutral IR from request
    ir = OpenAIFormatter.parse_request(request)

    # Preprocess images if present
    if image_urls:
        images = await preprocess_images(image_urls)
        ir = ir.model_copy(update={"images": images})

    # Vision models always use direct path (cloud/batching don't support vision)
    has_images = ir.images is not None
    if has_images:
        return await _handle_direct_inference(ir, request)

    # Always route (router is passthrough when no rules match)
    try:
        return await _route_and_respond(ir, request)
    except Exception as e:
        logger.warning(f"Routing failed, falling back: {e}")

    # Try batching path if enabled
    if settings.enable_batching:
        try:
            mgr = get_scheduler_manager()
            priority = mgr.get_priority_for_request(
                api_key=None,
                endpoint="/v1/chat/completions",
            )
            return await _handle_batched_request(ir, request, priority)
        except RuntimeError:
            logger.warning("Batching enabled but scheduler not initialized, using direct")
        except Exception as e:
            logger.warning(f"Batching unavailable, falling back to direct: {e}")

    # Direct inference path
    return await _handle_direct_inference(ir, request)


async def _route_and_respond(
    ir: InternalRequest,
    request: ChatCompletionRequest,
) -> EventSourceResponse | ChatCompletionResponse:
    """Route via BackendRouter and format the RoutingOutcome.

    Handles four outcome types:
    - Passthrough streaming (raw_stream): wrap as EventSourceResponse
    - Passthrough non-streaming (raw_response): convert to ChatCompletionResponse
    - IR streaming (ir_stream): format with OpenAIFormatter
    - IR non-streaming (ir_result): format with OpenAIFormatter
    """
    settings = get_settings()
    timeout = settings.timeout_chat_seconds
    backend_router = get_router()

    outcome = await with_inference_timeout(
        backend_router.route_request(ir),
        timeout=timeout,
        description="Chat completion (routed)",
    )

    # Passthrough: cloud backend returned protocol-native response
    if outcome.is_passthrough:
        if outcome.raw_stream is not None:
            raw_stream = outcome.raw_stream

            async def event_generator() -> Any:
                async for chunk in raw_stream:
                    yield {"data": json.dumps(chunk)}
                yield {"data": "[DONE]"}

            return EventSourceResponse(event_generator())
        else:
            result_dict = cast(dict[str, Any], outcome.raw_response)
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

    # IR streaming result from router
    if outcome.ir_stream is not None:
        return _format_ir_stream_as_sse(outcome.ir_stream, ir.model)

    # IR non-streaming result from router
    if outcome.ir_result is not None:
        return _format_ir_complete(outcome.ir_result, request)

    raise RuntimeError("RoutingOutcome has no result")


async def _handle_direct_inference(
    ir: InternalRequest,
    request: ChatCompletionRequest,
) -> EventSourceResponse | ChatCompletionResponse:
    """Handle request via direct local inference (non-batched)."""
    if ir.stream:
        return await _handle_streaming(ir, request)
    else:
        return await _handle_non_streaming(ir, request)


async def _handle_streaming(
    ir: InternalRequest,
    request: ChatCompletionRequest,
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
                model_id=ir.model,
                request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            )

            # Emit initial role chunk
            for sse in formatter.stream_start():
                yield sse

            # Apply timeout to preparation (model loading, template application)
            gen = await asyncio.wait_for(
                generate_chat_stream(
                    model_id=ir.model,
                    messages=ir.messages,
                    max_tokens=ir.params.max_tokens or 4096,
                    temperature=ir.params.temperature or 1.0,
                    top_p=ir.params.top_p or 1.0,
                    stop=ir.stop,
                    tools=ir.tools,
                    images=ir.images,
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
            yield timeout_error_event(timeout)

    return EventSourceResponse(event_generator())


async def _handle_non_streaming(
    ir: InternalRequest,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Handle non-streaming response.

    Uses the 3-layer adapter pipeline: inference returns IR TextResult,
    OpenAIFormatter converts it to an OpenAI ChatCompletion response dict.
    """
    settings = get_settings()
    timeout = settings.timeout_chat_seconds

    inference_result = await with_inference_timeout(
        generate_chat_complete_response(
            model_id=ir.model,
            messages=ir.messages,
            max_tokens=ir.params.max_tokens or 4096,
            temperature=ir.params.temperature or 1.0,
            top_p=ir.params.top_p or 1.0,
            stop=ir.stop,
            tools=ir.tools,
            images=ir.images,
        ),
        timeout=timeout,
        description="Chat completion",
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

    return _format_ir_complete(inference_result, request)


def _format_ir_complete(
    inference_result: InferenceResult,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Format an InferenceResult as a ChatCompletionResponse."""
    formatter = OpenAIFormatter(
        model_id=request.model,
        request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
    )
    result_dict = formatter.format_complete(
        inference_result.result,
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


def _format_ir_stream_as_sse(
    ir_stream: AsyncGenerator,
    model: str,
) -> EventSourceResponse:
    """Format an IR stream as an EventSourceResponse with OpenAI SSE format."""

    async def event_generator() -> Any:
        formatter = OpenAIFormatter(
            model_id=model,
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        )
        for sse in formatter.stream_start():
            yield sse

        output_tokens = 0
        async for item in ir_stream:
            if isinstance(item, TextResult):
                for sse in formatter.stream_end(
                    item.finish_reason,
                    tool_calls=item.tool_calls,
                    output_tokens=output_tokens,
                ):
                    yield sse
            else:
                output_tokens += 1
                for sse in formatter.stream_event(item):
                    yield sse

    return EventSourceResponse(event_generator())


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
    ir: InternalRequest,
    request: ChatCompletionRequest,
    priority: Priority,
) -> EventSourceResponse | ChatCompletionResponse:
    """Route request through batching scheduler.

    Tokenizes the prompt, creates a BatchRequest, and submits to the
    scheduler for batched processing.
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool

    pool = get_model_pool()
    loaded = await pool.get_model(ir.model)
    adapter = loaded.adapter
    if adapter is None:
        # Fallback for edge cases
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        adapter = create_adapter("default", loaded.tokenizer, model_type=loaded.model_type)

    # Get actual tokenizer (handle Processor wrapper for vision models)
    actual_tokenizer = getattr(loaded.tokenizer, "tokenizer", loaded.tokenizer)

    # Apply chat template to get prompt string
    prompt = adapter.apply_chat_template(ir.messages, add_generation_prompt=True)

    # Tokenize prompt
    prompt_tokens = actual_tokenizer.encode(prompt)

    # Create batch request with unique ID
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    batch_request = BatchRequest(
        request_id=request_id,
        model_id=ir.model,
        prompt_tokens=prompt_tokens,
        max_tokens=ir.params.max_tokens or 4096,
        priority=priority,
    )

    # Get scheduler and configure
    mgr = get_scheduler_manager()
    scheduler = await mgr.get_scheduler(ir.model)
    await mgr.configure_scheduler(ir.model, loaded.model, loaded.tokenizer, adapter)

    # Route based on streaming preference
    if ir.stream:
        return await _stream_batched_response(batch_request, scheduler, ir.model)
    else:
        return await _complete_batched_response(batch_request, scheduler, ir.model)


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
            yield timeout_error_event(timeout)

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

    await with_inference_timeout(
        collect_tokens(),
        timeout=timeout,
        description="Batched chat completion",
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
