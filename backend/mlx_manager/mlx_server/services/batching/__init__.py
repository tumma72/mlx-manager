"""Continuous batching services for MLX inference.

This module provides the infrastructure for continuous batching:
- Request lifecycle management (BatchRequest)
- Priority scheduling with aging (PriorityQueueWithAging)
- Status and priority enums
- Paged KV cache block management (KVBlock, BlockTable, PagedBlockManager)
- Prefix caching for KV block sharing (PrefixCache)
- Continuous batching scheduler (ContinuousBatchingScheduler)
- Batch inference engine (BatchInferenceEngine) for MLX generation
- Scheduler manager (SchedulerManager) for per-model scheduler instances
- Benchmarking utilities (BenchmarkResult, run_benchmark)
"""

from mlx_manager.mlx_server.services.batching.batch_inference import (
    BatchInferenceEngine,
)
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
from mlx_manager.mlx_server.services.batching.scheduler_manager import (
    SchedulerManager,
    get_scheduler_manager,
    init_scheduler_manager,
    reset_scheduler_manager,
)
from mlx_manager.mlx_server.services.batching.types import Priority, RequestStatus
from mlx_manager.mlx_server.services.batching.benchmark import (
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    "BLOCK_SIZE",
    "BatchInferenceEngine",
    "BatchRequest",
    "BenchmarkResult",
    "BlockTable",
    "ContinuousBatchingScheduler",
    "KVBlock",
    "PagedBlockManager",
    "PrefixCache",
    "Priority",
    "PriorityQueueWithAging",
    "RequestStatus",
    "SchedulerManager",
    "compute_block_hash",
    "get_scheduler_manager",
    "init_scheduler_manager",
    "reset_scheduler_manager",
    "run_benchmark",
]
