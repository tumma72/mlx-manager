"""Legacy completions endpoint."""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.schemas.openai import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from mlx_manager.mlx_server.services.inference import generate_completion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["completions"])


@router.post("/completions", response_model=None)
async def create_completion(
    request: CompletionRequest,
) -> EventSourceResponse | CompletionResponse:
    """Create a text completion (legacy API).

    Supports both streaming and non-streaming responses.
    Compatible with OpenAI Completions API.
    """
    logger.info(f"Completion request: model={request.model}, stream={request.stream}")

    # Handle stop parameter
    stop: list[str] | None = (
        request.stop
        if isinstance(request.stop, list)
        else ([request.stop] if request.stop else None)
    )

    try:
        if request.stream:
            return await _handle_streaming(request, stop)
        else:
            return await _handle_non_streaming(request, stop)
    except RuntimeError as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _handle_streaming(
    request: CompletionRequest,
    stop: list[str] | None,
) -> EventSourceResponse:
    """Handle streaming response."""

    async def event_generator() -> Any:
        gen = await generate_completion(
            model_id=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens or 16,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
            stream=True,
            echo=request.echo,
        )
        async for chunk in gen:  # type: ignore[union-attr]
            yield {"data": json.dumps(chunk)}

        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


async def _handle_non_streaming(
    request: CompletionRequest,
    stop: list[str] | None,
) -> CompletionResponse:
    """Handle non-streaming response."""
    result = await generate_completion(
        model_id=request.model,
        prompt=request.prompt,
        max_tokens=request.max_tokens or 16,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop,
        stream=False,
        echo=request.echo,
    )

    result_dict = result  # type: ignore[assignment]
    choice = result_dict["choices"][0]
    return CompletionResponse(
        id=result_dict["id"],
        created=result_dict["created"],
        model=result_dict["model"],
        choices=[
            CompletionChoice(
                index=choice["index"],
                text=choice["text"],
                finish_reason=choice["finish_reason"],
            )
        ],
        usage=Usage(
            prompt_tokens=result_dict["usage"]["prompt_tokens"],
            completion_tokens=result_dict["usage"]["completion_tokens"],
            total_tokens=result_dict["usage"]["total_tokens"],
        ),
    )
