"""MLX Server services."""

from mlx_manager.mlx_server.services.inference import (
    generate_chat_completion,
    generate_completion,
)

__all__ = [
    "generate_chat_completion",
    "generate_completion",
]
