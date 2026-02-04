"""Background health checker service.

With the embedded MLX Server, this now monitors the model pool health
instead of external subprocess instances.
"""

import asyncio

from loguru import logger


class HealthChecker:
    """Background service that monitors embedded MLX Server health."""

    def __init__(self, interval: int = 30):
        self.interval = interval
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the health check loop."""
        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        """Stop the health check loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self._check_model_pool()
            except Exception as e:
                logger.debug(f"Health check: {e}")

            await asyncio.sleep(self.interval)

    async def _check_model_pool(self) -> None:
        """Check health of the embedded MLX Server model pool."""
        try:
            from mlx_manager.mlx_server.models.pool import get_model_pool
            from mlx_manager.mlx_server.utils.memory import get_memory_usage

            pool = get_model_pool()
            loaded_models = pool.get_loaded_models()
            memory = get_memory_usage()

            logger.debug(
                f"Model pool health: {len(loaded_models)} models loaded, "
                f"{memory['active_gb']:.1f}GB active"
            )
        except RuntimeError:
            # Pool not initialized yet - this is normal during startup
            pass


# Singleton instance
health_checker = HealthChecker()
