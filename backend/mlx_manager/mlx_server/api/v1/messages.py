"""Anthropic Messages API endpoint."""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.models.ir import TextResult
from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from mlx_manager.mlx_server.services.audit import audit_service
from mlx_manager.mlx_server.services.formatters import AnthropicFormatter
from mlx_manager.mlx_server.services.inference import (
    generate_chat_complete_response,
    generate_chat_stream,
)

router = APIRouter(tags=["messages"])


@router.post("/messages", response_model=None)
async def create_message(
    request: AnthropicMessagesRequest,
) -> EventSourceResponse | AnthropicMessagesResponse:
    """Create a message using Anthropic Messages API format.

    Translates Anthropic-format request to internal format, runs inference,
    and returns Anthropic-format response. Supports both streaming and
    non-streaming modes.

    Reference: https://platform.claude.com/docs/en/api/messages
    """
    request_id = f"msg_{uuid.uuid4().hex[:24]}"
    logger.info(f"Messages request: model={request.model}, stream={request.stream}")

    async with audit_service.track_request(
        request_id=request_id,
        model=request.model,
        endpoint="/v1/messages",
        backend_type="local",
    ) as audit_ctx:
        try:
            # Convert to internal format
            internal = AnthropicFormatter.parse_request(request)

            if request.stream:
                return await _handle_streaming(request, internal)
            else:
                result = await _handle_non_streaming(request, internal)
                # Update audit context with usage
                if result.usage:
                    audit_ctx.prompt_tokens = result.usage.input_tokens
                    audit_ctx.completion_tokens = result.usage.output_tokens
                    audit_ctx.total_tokens = result.usage.input_tokens + result.usage.output_tokens
                return result

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Messages API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


async def _handle_non_streaming(
    request: AnthropicMessagesRequest,
    internal: Any,  # InternalRequest from protocol.py
) -> AnthropicMessagesResponse:
    """Handle non-streaming request.

    Uses the 3-layer adapter pipeline: inference returns IR TextResult,
    AnthropicFormatter converts it to an Anthropic Messages response.
    """
    inference_result = await generate_chat_complete_response(
        model_id=internal.model,
        messages=internal.messages,
        max_tokens=internal.params.max_tokens,
        temperature=internal.params.temperature,
        top_p=internal.params.top_p or 1.0,
        stop=internal.stop,
    )

    formatter = AnthropicFormatter(
        model_id=request.model,
        request_id=f"msg_{uuid.uuid4().hex[:24]}",
    )
    return formatter.format_complete(
        inference_result.result,
        prompt_tokens=inference_result.prompt_tokens,
        completion_tokens=inference_result.completion_tokens,
    )


async def _handle_streaming(
    request: AnthropicMessagesRequest,
    internal: Any,  # InternalRequest from protocol.py
) -> EventSourceResponse:
    """Handle streaming request with Anthropic-format SSE events.

    Uses the 3-layer adapter pipeline: inference yields IR StreamEvents,
    AnthropicFormatter converts them to Anthropic Messages API SSE events.

    Event sequence:
    - event: message_start (initial message metadata)
    - event: content_block_start (begin content block)
    - event: content_block_delta (token chunks)
    - event: content_block_stop (end content block)
    - event: message_delta (final stop_reason and usage)
    - event: message_stop (stream complete)
    """

    async def generate_events() -> Any:
        formatter = AnthropicFormatter(
            model_id=request.model,
            request_id=f"msg_{uuid.uuid4().hex[:24]}",
        )

        # Emit Anthropic message_start + content_block_start
        for sse in formatter.stream_start():
            yield sse

        # Stream IR events and format as Anthropic content_block_delta
        gen = await generate_chat_stream(
            model_id=internal.model,
            messages=internal.messages,
            max_tokens=internal.params.max_tokens,
            temperature=internal.params.temperature,
            top_p=internal.params.top_p or 1.0,
            stop=internal.stop,
        )

        output_tokens = 0
        async for item in gen:
            if isinstance(item, TextResult):
                # Final result — emit closing events
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

    return EventSourceResponse(generate_events())
