"""Error handling for MLX Server."""

from mlx_manager.mlx_server.errors.handlers import register_error_handlers
from mlx_manager.mlx_server.errors.problem_details import (
    ProblemDetail,
    TimeoutHTTPException,
    TimeoutProblem,
)

__all__ = [
    "ProblemDetail",
    "TimeoutProblem",
    "TimeoutHTTPException",
    "register_error_handlers",
]
