"""Shared request handling utilities for MLX Server API routers."""

import asyncio
import json
from collections.abc import Awaitable
from typing import TypeVar

from loguru import logger

from mlx_manager.mlx_server.errors import TimeoutHTTPException

T = TypeVar("T")


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
