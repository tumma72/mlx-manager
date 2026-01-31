"""Legacy completions endpoint."""

import asyncio
import json
import logging
import uuid
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.config import get_settings
from mlx_manager.mlx_server.errors import TimeoutHTTPException
from mlx_manager.mlx_server.schemas.openai import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from mlx_manager.mlx_server.services.audit import audit_service
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
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    logger.info(f"Completion request: model={request.model}, stream={request.stream}")

    # Handle stop parameter
    stop: list[str] | None = (
        request.stop
        if isinstance(request.stop, list)
        else ([request.stop] if request.stop else None)
    )

    async with audit_service.track_request(
        request_id=request_id,
        model=request.model,
        endpoint="/v1/completions",
        backend_type="local",
    ) as audit_ctx:
        try:
            if request.stream:
                return await _handle_streaming(request, stop)
            else:
                result = await _handle_non_streaming(request, stop)
                # Update audit context with usage
                if result.usage:
                    audit_ctx.prompt_tokens = result.usage.prompt_tokens
                    audit_ctx.completion_tokens = result.usage.completion_tokens
                    audit_ctx.total_tokens = result.usage.total_tokens
                return result
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
    """Handle streaming response with timeout."""
    settings = get_settings()
    timeout = settings.timeout_completions_seconds

    async def event_generator() -> Any:
        try:
            gen = await asyncio.wait_for(
                generate_completion(
                    model_id=request.model,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens or 16,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=stop,
                    stream=True,
                    echo=request.echo,
                ),
                timeout=timeout,
            )
            async for chunk in gen:  # type: ignore[union-attr]
                yield {"data": json.dumps(chunk)}

            yield {"data": "[DONE]"}
        except asyncio.TimeoutError:
            logger.warning(f"Streaming completion timed out after {timeout}s")
            # Send error event before closing
            error_event = {
                "error": {
                    "type": "https://mlx-manager.dev/errors/timeout",
                    "message": f"Request timed out after {int(timeout)} seconds",
                }
            }
            yield {"event": "error", "data": json.dumps(error_event)}

    return EventSourceResponse(event_generator())


async def _handle_non_streaming(
    request: CompletionRequest,
    stop: list[str] | None,
) -> CompletionResponse:
    """Handle non-streaming response with timeout."""
    settings = get_settings()
    timeout = settings.timeout_completions_seconds

    try:
        result = await asyncio.wait_for(
            generate_completion(
                model_id=request.model,
                prompt=request.prompt,
                max_tokens=request.max_tokens or 16,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
                stream=False,
                echo=request.echo,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Completion timed out after {timeout}s")
        raise TimeoutHTTPException(
            timeout_seconds=timeout,
            detail=f"Completion timed out after {int(timeout)} seconds. "
            f"Consider using a smaller model or reducing max_tokens.",
        )

    # Cast to dict since we passed stream=False
    result_dict = cast(dict[str, Any], result)
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
