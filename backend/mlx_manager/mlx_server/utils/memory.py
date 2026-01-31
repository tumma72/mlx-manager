"""MLX memory management utilities."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Lazy import to allow testing without MLX
def _get_mx() -> Any:
    """Lazy import mlx.core."""
    import mlx.core as mx

    return mx


def get_memory_usage() -> dict[str, float]:
    """Get current MLX Metal memory usage.

    Returns:
        dict with keys:
        - active_gb: Currently allocated memory
        - peak_gb: Peak memory usage
        - cache_gb: Cached memory
    """
    try:
        mx = _get_mx()
        # Use new API (mx.get_*) instead of deprecated mx.metal.get_*
        return {
            "active_gb": mx.get_active_memory() / (1024**3),
            "peak_gb": mx.get_peak_memory() / (1024**3),
            "cache_gb": mx.get_cache_memory() / (1024**3),
        }
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return {"active_gb": 0.0, "peak_gb": 0.0, "cache_gb": 0.0}


def clear_cache() -> None:
    """Clear MLX Metal cache.

    Should be called after generation to free memory.
    """
    try:
        mx = _get_mx()
        mx.synchronize()  # Wait for pending operations
        # Use new API (mx.clear_cache) instead of deprecated mx.metal.clear_cache
        mx.clear_cache()
        logger.debug("MLX cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear cache: {e}")


def set_memory_limit(max_gb: float) -> None:
    """Set MLX memory limit.

    Args:
        max_gb: Maximum memory in gigabytes
    """
    try:
        mx = _get_mx()
        max_bytes = int(max_gb * 1024**3)
        # Use new API (mx.set_memory_limit) - no relaxed parameter in newer MLX versions
        mx.set_memory_limit(max_bytes)
        logger.info(f"MLX memory limit set to {max_gb:.1f} GB")
    except Exception as e:
        logger.warning(f"Failed to set memory limit: {e}")


def reset_peak_memory() -> None:
    """Reset peak memory counter."""
    try:
        mx = _get_mx()
        # Use new API (mx.reset_peak_memory) instead of deprecated mx.metal.reset_peak_memory
        mx.reset_peak_memory()
    except Exception as e:
        logger.warning(f"Failed to reset peak memory: {e}")
