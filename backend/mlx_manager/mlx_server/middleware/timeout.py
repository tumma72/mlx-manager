"""Timeout decorator for async endpoints.

Uses asyncio.wait_for to enforce per-endpoint timeouts.
Per CONTEXT.md: Timeout = error, discard partial response.
Same timeouts apply to both local and cloud backends.
"""

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from loguru import logger

from mlx_manager.mlx_server.errors.problem_details import TimeoutHTTPException

P = ParamSpec("P")
T = TypeVar("T")


def with_timeout(
    seconds: float,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to add timeout to async endpoint.

    Args:
        seconds: Timeout in seconds. If exceeded, raises TimeoutHTTPException.

    Returns:
        Decorator that wraps the endpoint with asyncio.wait_for.

    Example:
        @router.post("/v1/chat/completions")
        @with_timeout(900.0)  # 15 minutes
        async def create_chat_completion(request: ChatCompletionRequest):
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except TimeoutError:
                logger.warning(f"Endpoint {func.__name__} timed out after {seconds}s")
                raise TimeoutHTTPException(
                    timeout_seconds=seconds,
                    detail=f"Request timed out after {int(seconds)} seconds. "
                    f"Consider using a smaller model or reducing max_tokens.",
                )

        return wrapper

    return decorator


def get_timeout_for_endpoint(endpoint: str) -> float:
    """Get configured timeout for an endpoint type.

    Args:
        endpoint: Endpoint path (e.g., "/v1/chat/completions")

    Returns:
        Timeout in seconds from settings.
    """
    from mlx_manager.mlx_server.config import get_settings

    settings = get_settings()

    if "chat" in endpoint:
        return settings.timeout_chat_seconds
    elif "completions" in endpoint:
        return settings.timeout_completions_seconds
    elif "embeddings" in endpoint:
        return settings.timeout_embeddings_seconds
    else:
        # Default to chat timeout for unknown endpoints
        return settings.timeout_chat_seconds
