"""Shared request handling utilities for MLX Server API routers."""

import asyncio
import json
from collections.abc import Awaitable
from typing import TypeVar

from fastapi import HTTPException
from loguru import logger

from mlx_manager.mlx_server.errors import TimeoutHTTPException

T = TypeVar("T")


def validate_model_available(model: str | None) -> str:
    """Validate and resolve the model field.

    If no model is specified, falls back to the configured default_model.
    Checks that the resolved model is in the available_models list.

    Args:
        model: The model ID from the request, or None.

    Returns:
        The resolved model ID.

    Raises:
        HTTPException: 400 if no model and no default configured.
        HTTPException: 404 if the model is not in available_models.
    """
    from mlx_manager.mlx_server.config import get_settings

    settings = get_settings()

    # Resolve default model
    if not model:
        if settings.default_model:
            model = settings.default_model
        else:
            raise HTTPException(
                status_code=400,
                detail="No model specified and no default model configured",
            )

    # Check against available models (static config + loaded pool models)
    available = set(settings.available_models)
    try:
        from mlx_manager.mlx_server.models.pool import get_model_pool

        pool = get_model_pool()
        available.update(pool.get_loaded_models())
    except RuntimeError:
        pass  # Pool not initialized yet

    if model not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not available. "
            f"Available models: {', '.join(sorted(available))}",
        )

    return model


def timeout_error_event(timeout: float) -> dict:
    """Create SSE error event for timeout.

    Returns an SSE event dict suitable for yielding inside an event_generator.
    Follows the error format documented in CONTEXT.md streaming errors.
    """
    return {
        "event": "error",
        "data": json.dumps(
            {
                "error": {
                    "type": "https://mlx-manager.dev/errors/timeout",
                    "message": f"Request timed out after {int(timeout)} seconds",
                }
            }
        ),
    }


async def with_inference_timeout(
    coro: Awaitable[T],
    timeout: float,
    description: str,
) -> T:
    """Run an awaitable with timeout, raising TimeoutHTTPException on expiry.

    Args:
        coro: The awaitable to run.
        timeout: Seconds before timeout.
        description: What timed out (used in warning log and error detail).

    Returns:
        The result of the awaitable.

    Raises:
        TimeoutHTTPException: If the operation times out.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        logger.warning(f"{description} timed out after {timeout}s")
        raise TimeoutHTTPException(
            timeout_seconds=timeout,
            detail=f"{description} timed out after {int(timeout)} seconds",
        )
