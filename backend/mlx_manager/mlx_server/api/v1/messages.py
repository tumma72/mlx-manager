"""Anthropic Messages API endpoint."""

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    TextBlock,
    Usage,
)
from mlx_manager.mlx_server.services.inference import generate_chat_completion
from mlx_manager.mlx_server.services.protocol import get_translator

# Type alias for Anthropic stop reasons
AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["messages"])


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
    logger.info(f"Messages request: model={request.model}, stream={request.stream}")

    translator = get_translator()

    try:
        # Convert to internal format
        internal = translator.anthropic_to_internal(request)

        if request.stream:
            return await _handle_streaming(request, internal)
        else:
            return await _handle_non_streaming(request, internal)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Messages API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_non_streaming(
    request: AnthropicMessagesRequest,
    internal: Any,  # InternalRequest from protocol.py
) -> AnthropicMessagesResponse:
    """Handle non-streaming request."""
    translator = get_translator()

    # Generate completion (non-streaming returns dict, not generator)
    result = cast(
        dict[str, Any],
        await generate_chat_completion(
            model_id=internal.model,
            messages=internal.messages,
            max_tokens=internal.max_tokens,
            temperature=internal.temperature,
            top_p=internal.top_p or 1.0,
            stop=internal.stop,
            stream=False,
        ),
    )

    # Extract from result dict
    choice = result["choices"][0]
    usage = result["usage"]

    # Translate stop reason
    openai_stop = choice.get("finish_reason", "stop")
    anthropic_stop = cast(
        AnthropicStopReason, translator.openai_stop_to_anthropic(openai_stop)
    )

    return AnthropicMessagesResponse(
        id=result["id"].replace("chatcmpl-", "msg_"),
        model=result["model"],
        content=[TextBlock(text=choice["message"]["content"])],
        stop_reason=anthropic_stop,
        usage=Usage(
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        ),
    )


async def _handle_streaming(
    request: AnthropicMessagesRequest,
    internal: Any,  # InternalRequest from protocol.py
) -> EventSourceResponse:
    """Handle streaming request with Anthropic-format SSE events.

    Anthropic streaming uses named event types (not just data: lines):
    - event: message_start (initial message metadata)
    - event: content_block_start (begin content block)
    - event: content_block_delta (token chunks)
    - event: content_block_stop (end content block)
    - event: message_delta (final stop_reason and usage)
    - event: message_stop (stream complete)

    The internal generate_chat_completion returns OpenAI-format chunks which
    we translate to Anthropic events by wrapping each token in content_block_delta.
    """
    translator = get_translator()
    request_id = f"msg_{uuid.uuid4().hex[:24]}"

    async def generate_events() -> Any:
        """Generate Anthropic-format SSE events.

        Translation flow:
        1. Emit Anthropic message_start and content_block_start events
        2. For each OpenAI chunk from generate_chat_completion:
           - Extract delta.content (token text)
           - Wrap in Anthropic content_block_delta event format
        3. Emit Anthropic closing events with translated stop_reason
        """
        output_tokens = 0

        # 1. message_start event - Anthropic requires this as first event
        yield {
            "event": "message_start",
            "data": json.dumps({
                "type": "message_start",
                "message": {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": request.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }),
        }

        # 2. content_block_start event - signals beginning of text block
        yield {
            "event": "content_block_start",
            "data": json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
        }

        # 3. Stream tokens as content_block_delta events
        # generate_chat_completion returns OpenAI-format chunks:
        # {"choices": [{"delta": {"content": "token"}, "finish_reason": null}]}
        finish_reason = "end_turn"
        gen = cast(
            AsyncGenerator[dict[str, Any], None],
            await generate_chat_completion(
                model_id=internal.model,
                messages=internal.messages,
                max_tokens=internal.max_tokens,
                temperature=internal.temperature,
                top_p=internal.top_p or 1.0,
                stop=internal.stop,
                stream=True,
            ),
        )

        async for chunk in gen:
            # Extract token from OpenAI-format chunk
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                token_text = delta.get("content", "")
                chunk_finish = choices[0].get("finish_reason")

                # Translate OpenAI chunk -> Anthropic content_block_delta
                if token_text:
                    output_tokens += 1
                    yield {
                        "event": "content_block_delta",
                        "data": json.dumps({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": token_text},
                        }),
                    }

                # Capture finish_reason for translation at end
                if chunk_finish:
                    finish_reason = translator.openai_stop_to_anthropic(chunk_finish)

        # 4. content_block_stop event - signals end of text block
        yield {
            "event": "content_block_stop",
            "data": json.dumps({
                "type": "content_block_stop",
                "index": 0,
            }),
        }

        # 5. message_delta event - contains final stop_reason and usage
        yield {
            "event": "message_delta",
            "data": json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": finish_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            }),
        }

        # 6. message_stop event - signals stream complete
        yield {
            "event": "message_stop",
            "data": json.dumps({"type": "message_stop"}),
        }

    return EventSourceResponse(generate_events())
