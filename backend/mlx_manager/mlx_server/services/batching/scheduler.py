"""Continuous batching scheduler for MLX inference.

This module implements the core scheduler that processes multiple requests
per token generation step, achieving 2-4x throughput improvement by
maximizing GPU utilization.

Key features:
- Iteration-level scheduling: requests join/leave at token boundaries
- Adaptive timing: waits longer when idle, processes immediately under load
- Memory-aware batching via PagedBlockManager integration
- BatchInferenceEngine integration for actual MLX generation
- Graceful shutdown with in-flight request completion
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from loguru import logger

from mlx_manager.mlx_server.services.batching.block import BLOCK_SIZE, BlockTable
from mlx_manager.mlx_server.services.batching.block_manager import PagedBlockManager
from mlx_manager.mlx_server.services.batching.priority_queue import (
    PriorityQueueWithAging,
)
from mlx_manager.mlx_server.services.batching.request import BatchRequest
from mlx_manager.mlx_server.services.batching.types import RequestStatus

if TYPE_CHECKING:  # pragma: no cover
    from mlx_manager.mlx_server.models.adapters.base import ModelAdapter
    from mlx_manager.mlx_server.services.batching.batch_inference import (
        BatchInferenceEngine,
    )
    from mlx_manager.mlx_server.services.batching.prefix_cache import PrefixCache


class ContinuousBatchingScheduler:
    """Scheduler for continuous batching with iteration-level scheduling.

    The scheduler maintains two sets of requests:
    - running: Currently generating tokens (up to max_batch_size)
    - waiting: Queued requests waiting for a slot

    Requests join the running set at step boundaries (not mid-generation)
    and are removed immediately upon completion, freeing slots for waiting
    requests (true continuous batching).

    Attributes:
        model_id: Model identifier for logging
        block_manager: PagedBlockManager for KV cache allocation
        max_batch_size: Maximum concurrent requests
        idle_wait_ms: Wait time when no requests are running
        load_wait_ms: Wait time between steps under load
    """

    def __init__(
        self,
        model_id: str,
        block_manager: PagedBlockManager,
        max_batch_size: int = 8,
        idle_wait_ms: float = 50.0,
        load_wait_ms: float = 5.0,
        model: Any = None,
        tokenizer: Any = None,
        adapter: ModelAdapter | None = None,
    ) -> None:
        """Initialize the continuous batching scheduler.

        Args:
            model_id: Model identifier for logging
            block_manager: PagedBlockManager for KV cache allocation
            max_batch_size: Maximum concurrent requests in a batch
            idle_wait_ms: Milliseconds to wait when idle (accumulate requests)
            load_wait_ms: Milliseconds between generation steps under load
            model: Optional MLX model for generation
            tokenizer: Optional tokenizer for encoding/decoding
            adapter: Optional model adapter for stop tokens
        """
        self.model_id = model_id
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.idle_wait_ms = idle_wait_ms
        self.load_wait_ms = load_wait_ms

        # Request tracking
        self.running: list[BatchRequest] = []
        self.waiting = PriorityQueueWithAging()

        # Lifecycle control
        self._shutdown = False
        self._loop_task: asyncio.Task[None] | None = None
        self._step_lock = asyncio.Lock()

        # For MLX Metal thread affinity
        self._generation_thread: threading.Thread | None = None

        # Batch inference engine (created when model is set)
        self._inference_engine: BatchInferenceEngine | None = None
        self._prefix_cache: PrefixCache | None = None

        # Set model if provided
        if model is not None and tokenizer is not None and adapter is not None:
            self.set_model(model, tokenizer, adapter)

        logger.info(
            f"Scheduler initialized: model={model_id}, "
            f"max_batch={max_batch_size}, "
            f"idle_wait={idle_wait_ms}ms, load_wait={load_wait_ms}ms"
        )

    async def submit(self, request: BatchRequest) -> AsyncGenerator[dict[str, Any], None]:
        """Submit a request and yield tokens as they are generated.

        The request is added to the waiting queue and will be picked up
        by the scheduling loop at the next step boundary.

        Args:
            request: The BatchRequest to process

        Yields:
            Token chunks from the request's output queue
        """
        # Add to waiting queue
        await self.waiting.put(request)
        logger.debug(f"Request {request.request_id} submitted to queue")

        # Yield tokens from output queue until completion signal (None)
        while True:
            try:
                # Wait for token from output queue
                token_data = await request.output_queue.get()

                # None signals completion
                if token_data is None:
                    logger.debug(f"Request {request.request_id} completed")
                    break

                yield token_data
            except asyncio.CancelledError:
                # Request cancelled - mark it
                request.mark_cancelled()
                raise

    def _allocate_prompt_blocks(self, request: BatchRequest) -> BlockTable:
        """Allocate blocks for prompt tokens.

        Calculates the number of blocks needed for the prompt and allocates
        them from the block manager.

        Args:
            request: The request needing block allocation

        Returns:
            BlockTable with mappings for prompt blocks

        Raises:
            MemoryError: If insufficient blocks available
        """
        block_table = BlockTable(request_id=request.request_id)

        # Calculate blocks needed for prompt
        num_prompt_tokens = len(request.prompt_tokens)
        num_blocks_needed = (num_prompt_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Allocate blocks
        for logical_idx in range(num_blocks_needed):
            physical_id = self.block_manager.allocate()
            block_table.logical_to_physical[logical_idx] = physical_id

        # Set token count to prompt length
        block_table.num_tokens = num_prompt_tokens

        logger.debug(
            f"Allocated {num_blocks_needed} blocks for request {request.request_id} "
            f"({num_prompt_tokens} prompt tokens)"
        )

        return block_table

    def _release_blocks(self, request: BatchRequest) -> None:
        """Release all blocks associated with a request.

        Args:
            request: The request whose blocks should be released
        """
        if request.block_table is None:
            return

        # Get all physical blocks from the block table
        if isinstance(request.block_table, BlockTable):
            physical_blocks = request.block_table.get_physical_blocks()
        else:
            # Legacy format: list of block IDs
            physical_blocks = request.block_table

        for block_id in physical_blocks:
            self.block_manager.release(block_id)

        logger.debug(f"Released {len(physical_blocks)} blocks for request {request.request_id}")

        # Clear the block table reference
        request.block_table = None

    def get_stats(self) -> dict[str, Any]:
        """Get current scheduler statistics.

        Returns:
            Dictionary with running, waiting, and max_batch counts
        """
        return {
            "running": len(self.running),
            "waiting": self.waiting.qsize(),
            "max_batch": self.max_batch_size,
        }

    def is_running(self) -> bool:
        """Check if the scheduler is running (not shutdown).

        Returns:
            True if scheduler is active, False if shutdown
        """
        return not self._shutdown

    def set_model(
        self,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
    ) -> None:
        """Set or switch the model for generation.

        Creates a new BatchInferenceEngine and PrefixCache for the model.
        This can be called at runtime to switch models.

        Args:
            model: MLX model instance
            tokenizer: HuggingFace tokenizer
            adapter: Model adapter for stop token detection
        """
        from mlx_manager.mlx_server.services.batching.batch_inference import (
            BatchInferenceEngine,
        )
        from mlx_manager.mlx_server.services.batching.prefix_cache import PrefixCache

        # Create prefix cache for this model
        self._prefix_cache = PrefixCache(model_id=self.model_id, block_manager=self.block_manager)

        # Create inference engine with prefix cache
        self._inference_engine = BatchInferenceEngine(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            prefix_cache=self._prefix_cache,
        )

        logger.info(f"Model set for scheduler: {self.model_id}")

    async def _batch_step(self) -> None:
        """Execute one token generation step for all running requests.

        Uses BatchInferenceEngine for actual MLX generation:
        1. Generates one token for each running request
        2. Distributes tokens to each request's output queue
        3. Detects stop tokens and marks requests for completion

        CRITICAL: Uses Queue-based threading pattern for MLX Metal
        thread affinity - generation runs in a dedicated thread via
        BatchInferenceEngine.
        """
        # If no inference engine, fall back to placeholder behavior
        if self._inference_engine is None:
            # Placeholder: just add a dummy token and mark complete if max reached
            for request in self.running:
                if not request.is_complete:
                    # Simulate token generation
                    request.add_token(0)  # Placeholder token

                    # Put token in output queue
                    await request.output_queue.put(
                        {
                            "token_id": 0,
                            "text": "",
                            "request_id": request.request_id,
                        }
                    )
            return

        # Create sampler (TODO: per-request sampling parameters)
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=1.0, top_p=1.0)

        # Generate tokens for all running requests
        try:
            results = await self._inference_engine.generate_batch_step(self.running, sampler)
        except Exception as e:
            logger.exception(f"Batch generation error: {e}")
            # Mark all requests as completed on error to avoid infinite loop
            for request in self.running:
                request.status = RequestStatus.COMPLETED
            return

        # Process results for each request
        for request in self.running:
            if request.request_id in results:
                token_text, token_id, is_stop = results[request.request_id]

                if not is_stop and token_id is not None:
                    # Add token to request's generated list
                    request.generated_tokens.append(token_id)
                    # Push to output queue for streaming
                    await request.output_queue.put(
                        {
                            "token_id": token_id,
                            "text": token_text,
                            "request_id": request.request_id,
                        }
                    )
                else:
                    # Mark for completion (don't yield stop token)
                    request.status = RequestStatus.COMPLETED

    async def _scheduling_loop(self) -> None:
        """Main iteration-level scheduling loop.

        This loop:
        1. Fills batch from waiting queue up to max_batch_size
        2. Executes one generation step for all running requests
        3. Removes completed requests, freeing slots immediately
        4. Sleeps adaptively based on load
        """
        logger.info(f"Scheduling loop started for model {self.model_id}")

        while not self._shutdown:
            # Adaptive wait: longer when idle to accumulate requests
            wait_ms = self.idle_wait_ms if not self.running else self.load_wait_ms

            async with self._step_lock:
                # Fill batch from waiting queue
                while len(self.running) < self.max_batch_size and not self.waiting.empty():
                    try:
                        request = await self.waiting.get()
                    except IndexError:
                        # Queue became empty between check and get
                        break

                    request.mark_prefilling()
                    try:
                        block_table = self._allocate_prompt_blocks(request)
                        request.block_table = block_table
                    except MemoryError as e:
                        # Not enough blocks - put request back in queue
                        logger.warning(f"Not enough blocks for request {request.request_id}: {e}")
                        request.status = RequestStatus.WAITING
                        await self.waiting.put(request)
                        # Wait before retrying to avoid busy loop
                        await asyncio.sleep(self.idle_wait_ms / 1000.0)
                        break

                    self.running.append(request)
                    request.mark_running()
                    logger.debug(
                        f"Request {request.request_id} moved to running "
                        f"(batch size: {len(self.running)})"
                    )

                if self.running:
                    # Generate one token for all running requests
                    await self._batch_step()

                    # Remove completed requests (true continuous batching)
                    completed = [r for r in self.running if r.is_complete]
                    for request in completed:
                        request.mark_completed()
                        await request.output_queue.put(None)  # Signal completion
                        self._release_blocks(request)
                        self.running.remove(request)
                        logger.debug(
                            f"Request {request.request_id} completed "
                            f"({len(request.generated_tokens)} tokens)"
                        )

            # Sleep if idle
            if not self.running and self.waiting.empty():
                await asyncio.sleep(wait_ms / 1000.0)

        logger.info(f"Scheduling loop stopped for model {self.model_id}")

    async def start(self) -> None:
        """Start the scheduling loop.

        Creates an asyncio task for the scheduling loop.
        """
        if self._loop_task is not None:
            logger.warning("Scheduler already started")
            return

        self._shutdown = False
        self._loop_task = asyncio.create_task(self._scheduling_loop())
        logger.info(f"Scheduler started for model {self.model_id}")

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop the scheduler gracefully.

        Waits for running requests to complete (with timeout),
        then cancels remaining waiting requests.

        Args:
            timeout: Maximum seconds to wait for running requests
        """
        logger.info(f"Stopping scheduler for model {self.model_id}")
        self._shutdown = True

        # Wait for running requests to complete (with timeout)
        if self.running:
            start_time = time.time()
            while self.running and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)

            if self.running:
                logger.warning(f"Timeout waiting for {len(self.running)} running requests")
                # Force complete remaining requests
                for request in self.running:
                    request.mark_cancelled()
                    await request.output_queue.put(None)
                    self._release_blocks(request)
                self.running.clear()

        # Cancel waiting requests
        cancelled_count = 0
        while not self.waiting.empty():
            try:
                request = await self.waiting.get()
                request.mark_cancelled()
                await request.output_queue.put(None)
                cancelled_count += 1
            except IndexError:
                break

        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} waiting requests")

        # Cancel the loop task
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        logger.info(f"Scheduler stopped for model {self.model_id}")
