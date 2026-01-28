"""Continuous batching services for MLX inference.

This module provides the infrastructure for continuous batching:
- Request lifecycle management (BatchRequest)
- Priority scheduling with aging (PriorityQueueWithAging)
- Status and priority enums
- Paged KV cache block management (KVBlock, BlockTable, PagedBlockManager)
- Prefix caching for KV block sharing (PrefixCache)
- Continuous batching scheduler (ContinuousBatchingScheduler)
"""

from mlx_manager.mlx_server.services.batching.block import (
    BLOCK_SIZE,
    BlockTable,
    KVBlock,
)
from mlx_manager.mlx_server.services.batching.block_manager import PagedBlockManager
from mlx_manager.mlx_server.services.batching.prefix_cache import (
    PrefixCache,
    compute_block_hash,
)
from mlx_manager.mlx_server.services.batching.priority_queue import (
    PriorityQueueWithAging,
)
from mlx_manager.mlx_server.services.batching.request import BatchRequest
from mlx_manager.mlx_server.services.batching.scheduler import (
    ContinuousBatchingScheduler,
)
from mlx_manager.mlx_server.services.batching.types import Priority, RequestStatus

__all__ = [
    "BLOCK_SIZE",
    "BatchRequest",
    "BlockTable",
    "ContinuousBatchingScheduler",
    "KVBlock",
    "PagedBlockManager",
    "PrefixCache",
    "Priority",
    "PriorityQueueWithAging",
    "RequestStatus",
    "compute_block_hash",
]
