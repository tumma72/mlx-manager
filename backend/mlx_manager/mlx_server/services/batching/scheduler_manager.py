"""Scheduler manager for per-model continuous batching schedulers.

This module provides a singleton manager that creates and manages
per-model ContinuousBatchingScheduler instances, determines request
priority, and handles lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from mlx_manager.mlx_server.services.batching.block_manager import PagedBlockManager
from mlx_manager.mlx_server.services.batching.scheduler import (
    ContinuousBatchingScheduler,
)
from mlx_manager.mlx_server.services.batching.types import Priority

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SchedulerManager:
    """Manager for per-model continuous batching schedulers.

    Creates and manages scheduler instances for each model, handles
    priority determination based on API key tier and endpoint, and
    coordinates scheduler lifecycle.

    Attributes:
        block_pool_size: Default number of blocks per model's PagedBlockManager
        max_batch_size: Default maximum batch size for schedulers
    """

    def __init__(
        self,
        block_pool_size: int = 1000,
        max_batch_size: int = 8,
    ) -> None:
        """Initialize the scheduler manager.

        Args:
            block_pool_size: Number of blocks for each model's PagedBlockManager
            max_batch_size: Maximum concurrent requests per scheduler
        """
        self.block_pool_size = block_pool_size
        self.max_batch_size = max_batch_size

        # Per-model resources
        self._schedulers: dict[str, ContinuousBatchingScheduler] = {}
        self._block_managers: dict[str, PagedBlockManager] = {}

        # Thread-safe access
        self._lock = asyncio.Lock()

        logger.info(
            f"SchedulerManager initialized: "
            f"block_pool_size={block_pool_size}, max_batch_size={max_batch_size}"
        )

    async def get_scheduler(self, model_id: str) -> ContinuousBatchingScheduler:
        """Get or create a scheduler for a model.

        Lazy-initializes a scheduler and its block manager for each
        model on first request.

        Args:
            model_id: The model identifier

        Returns:
            ContinuousBatchingScheduler for the model
        """
        async with self._lock:
            if model_id not in self._schedulers:
                # Create block manager for this model
                block_manager = PagedBlockManager(num_blocks=self.block_pool_size)
                self._block_managers[model_id] = block_manager

                # Create scheduler
                scheduler = ContinuousBatchingScheduler(
                    model_id=model_id,
                    block_manager=block_manager,
                    max_batch_size=self.max_batch_size,
                )

                # Start the scheduler
                await scheduler.start()

                self._schedulers[model_id] = scheduler
                logger.info(f"Created scheduler for model {model_id}")

            return self._schedulers[model_id]

    async def configure_scheduler(
        self,
        model_id: str,
        model: Any,
        tokenizer: Any,
        adapter: Any,
    ) -> None:
        """Configure a scheduler with model resources.

        Called when a model is loaded into the pool, to set up the
        BatchInferenceEngine for actual generation.

        Args:
            model_id: The model identifier
            model: The loaded MLX model
            tokenizer: The model's tokenizer
            adapter: The model adapter for chat template handling
        """
        # Ensure scheduler exists
        scheduler = await self.get_scheduler(model_id)

        # Wire the inference engine with the loaded model
        scheduler.set_model(model, tokenizer, adapter)
        logger.info(f"Scheduler configured for model {model_id} with inference engine")

    def get_priority_for_request(
        self,
        api_key: str | None,
        endpoint: str,
    ) -> Priority:
        """Determine request priority based on context.

        Priority determination order:
        1. Endpoint override: /v1/batch/* forces LOW priority
        2. API key tier lookup (placeholder - returns NORMAL)
        3. Default: NORMAL

        Args:
            api_key: API key from request headers (may be None)
            endpoint: Request endpoint path

        Returns:
            Priority enum value
        """
        # Endpoint-based override for batch processing
        if endpoint.startswith("/v1/batch"):
            return Priority.LOW

        # API key tier lookup placeholder
        # Future: database lookup for api_key -> tier -> priority mapping
        # For now, always return NORMAL for any API key
        if api_key is not None:
            # Placeholder: in future, lookup api_key in database
            # to get tier (free/pro/enterprise) -> priority
            pass

        return Priority.NORMAL

    async def shutdown(self) -> None:
        """Shutdown all schedulers gracefully.

        Stops all managed schedulers, allowing running requests to
        complete with a timeout.
        """
        logger.info("Shutting down SchedulerManager...")

        async with self._lock:
            # Stop all schedulers
            for model_id, scheduler in self._schedulers.items():
                logger.debug(f"Stopping scheduler for model {model_id}")
                await scheduler.stop()

            # Clear references
            self._schedulers.clear()
            self._block_managers.clear()

        logger.info("SchedulerManager shutdown complete")


# Singleton instance
_scheduler_manager: SchedulerManager | None = None


def get_scheduler_manager() -> SchedulerManager:
    """Get the singleton SchedulerManager instance.

    Raises:
        RuntimeError: If the manager has not been initialized

    Returns:
        The global SchedulerManager instance
    """
    if _scheduler_manager is None:
        raise RuntimeError("SchedulerManager not initialized")
    return _scheduler_manager


def init_scheduler_manager(**kwargs: Any) -> SchedulerManager:
    """Initialize the global SchedulerManager singleton.

    Args:
        **kwargs: Arguments passed to SchedulerManager constructor

    Returns:
        The initialized SchedulerManager instance
    """
    global _scheduler_manager
    _scheduler_manager = SchedulerManager(**kwargs)
    return _scheduler_manager


def reset_scheduler_manager() -> None:
    """Reset the singleton instance (for testing)."""
    global _scheduler_manager
    _scheduler_manager = None
