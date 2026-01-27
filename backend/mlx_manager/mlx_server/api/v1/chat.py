"""Chat completions endpoint."""

import json
import logging
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from mlx_manager.mlx_server.services.inference import generate_chat_completion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> EventSourceResponse | ChatCompletionResponse:
    """Create a chat completion.

    Supports both streaming and non-streaming responses.
    Compatible with OpenAI Chat Completions API.
    """
    logger.info(f"Chat completion request: model={request.model}, stream={request.stream}")

    # Convert messages to dict format
    messages: list[dict[str, Any]] = [
        {"role": m.role, "content": m.content} for m in request.messages
    ]

    # Handle stop parameter (can be string or list)
    stop: list[str] | None = (
        request.stop
        if isinstance(request.stop, list)
        else ([request.stop] if request.stop else None)
    )

    try:
        if request.stream:
            return await _handle_streaming(request, messages, stop)
        else:
            return await _handle_non_streaming(request, messages, stop)
    except RuntimeError as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _handle_streaming(
    request: ChatCompletionRequest,
    messages: list[dict[str, Any]],
    stop: list[str] | None,
) -> EventSourceResponse:
    """Handle streaming response."""

    async def event_generator() -> Any:
        gen = await generate_chat_completion(
            model_id=request.model,
            messages=messages,
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
            stream=True,
        )
        async for chunk in gen:  # type: ignore[union-attr]
            # Format as SSE data
            yield {"data": json.dumps(chunk)}

        # Final [DONE] message
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


async def _handle_non_streaming(
    request: ChatCompletionRequest,
    messages: list[dict[str, Any]],
    stop: list[str] | None,
) -> ChatCompletionResponse:
    """Handle non-streaming response."""
    result = await generate_chat_completion(
        model_id=request.model,
        messages=messages,
        max_tokens=request.max_tokens or 4096,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop,
        stream=False,
    )

    # Convert to Pydantic response model
    # Cast to dict since we passed stream=False
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
