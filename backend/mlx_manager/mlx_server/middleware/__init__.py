"""Middleware components for MLX Server."""

from mlx_manager.mlx_server.middleware.request_id import RequestIDMiddleware
from mlx_manager.mlx_server.middleware.shutdown import (
    GracefulShutdownMiddleware,
    ShutdownState,
    get_shutdown_state,
    reset_shutdown_state,
)
from mlx_manager.mlx_server.middleware.timeout import get_timeout_for_endpoint, with_timeout

__all__ = [
    "GracefulShutdownMiddleware",
    "RequestIDMiddleware",
    "ShutdownState",
    "get_shutdown_state",
    "get_timeout_for_endpoint",
    "reset_shutdown_state",
    "with_timeout",
]
