"""MLX Server utilities."""

from mlx_manager.mlx_server.utils.kv_cache import estimate_practical_max_tokens
from mlx_manager.mlx_server.utils.memory import (
    clear_cache,
    get_memory_usage,
    reset_peak_memory,
    set_memory_limit,
)
from mlx_manager.mlx_server.utils.metal import (
    get_metal_worker,
    reset_metal_worker,
    run_on_metal_thread,
    stream_from_metal_thread,
)

__all__ = [
    "clear_cache",
    "estimate_practical_max_tokens",
    "get_memory_usage",
    "get_metal_worker",
    "reset_metal_worker",
    "reset_peak_memory",
    "run_on_metal_thread",
    "set_memory_limit",
    "stream_from_metal_thread",
]
