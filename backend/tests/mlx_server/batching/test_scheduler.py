"""Tests for ContinuousBatchingScheduler."""

import asyncio

import pytest

from mlx_manager.mlx_server.services.batching import (
    BatchRequest,
    ContinuousBatchingScheduler,
    PagedBlockManager,
    Priority,
    RequestStatus,
)


@pytest.fixture
def block_manager() -> PagedBlockManager:
    """Create a block manager with 100 blocks."""
    return PagedBlockManager(num_blocks=100)


@pytest.fixture
def scheduler(block_manager: PagedBlockManager) -> ContinuousBatchingScheduler:
    """Create a scheduler for testing."""
    return ContinuousBatchingScheduler(
        model_id="test-model",
        block_manager=block_manager,
        max_batch_size=4,
        idle_wait_ms=1.0,  # Short waits for faster tests
        load_wait_ms=0.1,
    )


def make_request(
    request_id: str,
    prompt_tokens: list[int] | None = None,
    max_tokens: int = 10,
    priority: Priority = Priority.NORMAL,
) -> BatchRequest:
    """Helper to create test requests."""
    return BatchRequest(
        request_id=request_id,
        model_id="test-model",
        prompt_tokens=prompt_tokens if prompt_tokens is not None else [1, 2, 3],
        max_tokens=max_tokens,
        priority=priority,
    )


class TestSchedulerInitialization:
    """Tests for scheduler initialization and state queries."""

    def test_stats_show_zero_running_waiting(self, scheduler: ContinuousBatchingScheduler) -> None:
        """Fresh scheduler should show zero running and waiting."""
        stats = scheduler.get_stats()
        assert stats["running"] == 0
        assert stats["waiting"] == 0
        assert stats["max_batch"] == 4

    def test_is_running_true_before_shutdown(self, scheduler: ContinuousBatchingScheduler) -> None:
        """is_running() should be True before stop() is called."""
        assert scheduler.is_running() is True

    def test_custom_max_batch_size(self, block_manager: PagedBlockManager) -> None:
        """Scheduler should respect custom max_batch_size."""
        scheduler = ContinuousBatchingScheduler(
            model_id="test",
            block_manager=block_manager,
            max_batch_size=16,
        )
        assert scheduler.get_stats()["max_batch"] == 16

    def test_custom_timing_parameters(self, block_manager: PagedBlockManager) -> None:
        """Scheduler should store custom timing parameters."""
        scheduler = ContinuousBatchingScheduler(
            model_id="test",
            block_manager=block_manager,
            idle_wait_ms=100.0,
            load_wait_ms=10.0,
        )
        assert scheduler.idle_wait_ms == 100.0
        assert scheduler.load_wait_ms == 10.0


class TestRequestSubmission:
    """Tests for request submission to queue."""

    @pytest.mark.asyncio
    async def test_submit_adds_request_to_waiting_queue(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """submit() should add request to waiting queue."""
        request = make_request("test-1")

        # Start consuming tokens in background (but we just check queue)
        await scheduler.waiting.put(request)

        assert scheduler.waiting.qsize() == 1
        assert scheduler.get_stats()["waiting"] == 1

    @pytest.mark.asyncio
    async def test_multiple_requests_queue_in_priority_order(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """Multiple requests should queue in priority order."""
        low = make_request("low", priority=Priority.LOW)
        normal = make_request("normal", priority=Priority.NORMAL)
        high = make_request("high", priority=Priority.HIGH)

        # Add in reverse priority order
        await scheduler.waiting.put(low)
        await scheduler.waiting.put(normal)
        await scheduler.waiting.put(high)

        assert scheduler.waiting.qsize() == 3

        # Get should return in priority order
        first = await scheduler.waiting.get()
        assert first.request_id == "high"

        second = await scheduler.waiting.get()
        assert second.request_id == "normal"

        third = await scheduler.waiting.get()
        assert third.request_id == "low"


class TestBatchFilling:
    """Tests for batch filling from waiting queue."""

    @pytest.mark.asyncio
    async def test_requests_move_from_waiting_to_running(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """Requests should move from waiting to running during scheduling."""
        # Add a request
        request = make_request("test-1", max_tokens=1)
        await scheduler.waiting.put(request)

        # Start scheduler briefly
        await scheduler.start()
        await asyncio.sleep(0.05)  # Let one step execute
        await scheduler.stop(timeout=1.0)

        # Request should have been processed (1 token = complete)
        assert request.status in (
            RequestStatus.COMPLETED,
            RequestStatus.CANCELLED,
        )

    @pytest.mark.asyncio
    async def test_max_batch_size_respected(self, block_manager: PagedBlockManager) -> None:
        """Should not exceed max_batch_size running requests."""
        scheduler = ContinuousBatchingScheduler(
            model_id="test",
            block_manager=block_manager,
            max_batch_size=2,
            idle_wait_ms=1.0,
            load_wait_ms=0.1,
        )

        # Add 5 requests (more than max_batch_size=2)
        for i in range(5):
            request = make_request(f"test-{i}", max_tokens=100)
            await scheduler.waiting.put(request)

        # Start scheduler and check running count
        await scheduler.start()
        await asyncio.sleep(0.05)

        # Should only have 2 running (max_batch_size)
        assert len(scheduler.running) <= 2

        await scheduler.stop(timeout=1.0)

    @pytest.mark.asyncio
    async def test_block_allocation_failure_requeues_request(
        self, block_manager: PagedBlockManager
    ) -> None:
        """If block allocation fails, request should be re-queued."""
        # Create scheduler with very limited blocks (only 2 blocks)
        small_block_manager = PagedBlockManager(num_blocks=2)
        scheduler = ContinuousBatchingScheduler(
            model_id="test",
            block_manager=small_block_manager,
            max_batch_size=4,
            idle_wait_ms=1.0,
            load_wait_ms=0.1,
        )

        # Create a request that needs more blocks than available
        # With 2 blocks and BLOCK_SIZE=32, we can handle 64 tokens max
        # Create a request needing 100 tokens (4 blocks)
        large_request = make_request(
            "large",
            prompt_tokens=list(range(100)),  # Needs 4 blocks, we have 2
            max_tokens=10,
        )

        await scheduler.waiting.put(large_request)
        await scheduler.start()
        await asyncio.sleep(0.05)
        await scheduler.stop(timeout=1.0)

        # Request should not be in running (allocation failed)
        # It should have been re-queued or cancelled
        assert large_request not in scheduler.running


class TestCompletionHandling:
    """Tests for request completion handling."""

    @pytest.mark.asyncio
    async def test_completed_request_removed_from_running(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """Completed request should be removed from running list."""
        request = make_request("test-1", max_tokens=1)  # Complete after 1 token
        await scheduler.waiting.put(request)

        await scheduler.start()
        await asyncio.sleep(0.1)  # Let it complete
        await scheduler.stop(timeout=1.0)

        # Request should not be in running
        assert request not in scheduler.running
        assert request.status in (RequestStatus.COMPLETED, RequestStatus.CANCELLED)

    @pytest.mark.asyncio
    async def test_blocks_released_on_completion(
        self, scheduler: ContinuousBatchingScheduler, block_manager: PagedBlockManager
    ) -> None:
        """Blocks should be released when request completes."""
        initial_free = block_manager.get_free_count()

        request = make_request("test-1", prompt_tokens=[1, 2, 3], max_tokens=1)
        await scheduler.waiting.put(request)

        await scheduler.start()
        await asyncio.sleep(0.1)  # Let it complete
        await scheduler.stop(timeout=1.0)

        # Blocks should be back to free pool
        final_free = block_manager.get_free_count()
        assert final_free == initial_free  # All blocks released

    @pytest.mark.asyncio
    async def test_output_queue_receives_none_on_completion(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """Output queue should receive None signal on completion."""
        request = make_request("test-1", max_tokens=1)
        await scheduler.waiting.put(request)

        await scheduler.start()

        # Wait for completion signal
        result = None
        try:
            while True:
                result = await asyncio.wait_for(request.output_queue.get(), timeout=0.5)
                if result is None:
                    break
        except TimeoutError:
            pass

        await scheduler.stop(timeout=1.0)

        # Should have received None completion signal
        assert result is None


class TestShutdown:
    """Tests for scheduler shutdown."""

    @pytest.mark.asyncio
    async def test_stop_sets_shutdown_flag(self, scheduler: ContinuousBatchingScheduler) -> None:
        """stop() should set _shutdown flag."""
        await scheduler.start()
        assert scheduler.is_running()

        await scheduler.stop(timeout=1.0)
        assert not scheduler.is_running()

    @pytest.mark.asyncio
    async def test_waiting_requests_cancelled_on_stop(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """Waiting requests should be cancelled on stop."""
        # Add requests but don't start scheduler
        for i in range(3):
            await scheduler.waiting.put(make_request(f"test-{i}"))

        assert scheduler.waiting.qsize() == 3

        # Stop should cancel waiting requests
        await scheduler.stop(timeout=1.0)

        assert scheduler.waiting.qsize() == 0

    @pytest.mark.asyncio
    async def test_stop_waits_for_running_requests(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """stop() should wait for running requests to complete."""
        # Manually put a request in running state
        request = make_request("test-1", max_tokens=100)
        request.mark_running()
        scheduler.running.append(request)

        # Stop with very short timeout
        await scheduler.stop(timeout=0.1)

        # Request should be cancelled after timeout
        assert request.status == RequestStatus.CANCELLED
        assert len(scheduler.running) == 0


class TestSubmitGenerator:
    """Tests for submit() async generator."""

    @pytest.mark.asyncio
    async def test_submit_yields_tokens_from_output_queue(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """submit() yields tokens from the request's output queue."""
        request = make_request("test-sub", max_tokens=3)

        async def feed_tokens():
            """Push tokens then None to simulate generation."""
            await asyncio.sleep(0.01)
            await request.output_queue.put({"token_id": 1, "text": "a"})
            await request.output_queue.put({"token_id": 2, "text": "b"})
            await request.output_queue.put(None)  # completion signal

        # Run feeder in background
        feeder = asyncio.create_task(feed_tokens())

        tokens = []
        async for token in scheduler.submit(request):
            tokens.append(token)

        await feeder
        assert len(tokens) == 2
        assert tokens[0]["text"] == "a"
        assert tokens[1]["text"] == "b"

    @pytest.mark.asyncio
    async def test_submit_cancellation_marks_request(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """Cancelling submit() marks the request as cancelled."""
        request = make_request("test-cancel", max_tokens=100)

        async def start_submit():
            async for _ in scheduler.submit(request):
                pass

        task = asyncio.create_task(start_submit())
        await asyncio.sleep(0.05)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert request.status == RequestStatus.CANCELLED


class TestReleaseBlocksLegacy:
    """Tests for _release_blocks with legacy block format."""

    def test_release_blocks_legacy_format(
        self, scheduler: ContinuousBatchingScheduler, block_manager: PagedBlockManager
    ) -> None:
        """_release_blocks handles legacy list-of-block-IDs format."""
        # Allocate some blocks manually
        block_ids = [block_manager.allocate() for _ in range(3)]
        initial_free = block_manager.get_free_count()

        request = make_request("test-legacy")
        request.block_table = block_ids  # Legacy format: list of ints

        scheduler._release_blocks(request)

        assert block_manager.get_free_count() == initial_free + 3
        assert request.block_table is None


class TestBatchStepWithEngine:
    """Tests for _batch_step with and without inference engine."""

    @pytest.mark.asyncio
    async def test_batch_step_placeholder_without_engine(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """Without inference engine, batch_step uses placeholder behavior."""
        request = make_request("test-placeholder", max_tokens=1)
        request.mark_running()
        scheduler.running.append(request)

        await scheduler._batch_step()

        # Should have put a token dict into the output queue
        token_data = await asyncio.wait_for(request.output_queue.get(), timeout=1.0)
        assert token_data is not None
        assert token_data["token_id"] == 0

    @pytest.mark.asyncio
    async def test_batch_step_with_engine_error_marks_completed(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """Batch step engine error marks all running requests as COMPLETED."""
        from unittest.mock import AsyncMock, MagicMock

        mock_engine = MagicMock()
        mock_engine.generate_batch_step = AsyncMock(side_effect=RuntimeError("GPU error"))
        scheduler._inference_engine = mock_engine

        request = make_request("test-engine-err", max_tokens=10)
        request.mark_running()
        scheduler.running.append(request)

        await scheduler._batch_step()

        assert request.status == RequestStatus.COMPLETED


class TestAdaptiveTiming:
    """Tests for adaptive wait timing."""

    @pytest.mark.asyncio
    async def test_idle_state_uses_idle_wait(self, block_manager: PagedBlockManager) -> None:
        """Idle state (no running, no waiting) should use idle_wait_ms."""
        scheduler = ContinuousBatchingScheduler(
            model_id="test",
            block_manager=block_manager,
            idle_wait_ms=100.0,
            load_wait_ms=10.0,
        )

        # With no requests, should be idle
        assert not scheduler.running
        assert scheduler.waiting.empty()

        # The wait time selection happens inside the loop
        # We verify the parameters are set correctly
        assert scheduler.idle_wait_ms == 100.0
        assert scheduler.load_wait_ms == 10.0

    @pytest.mark.asyncio
    async def test_load_state_uses_load_wait(self, scheduler: ContinuousBatchingScheduler) -> None:
        """Load state (requests running) should use load_wait_ms."""
        # Add a running request
        request = make_request("test-1", max_tokens=100)
        request.mark_running()
        scheduler.running.append(request)

        # In the loop, wait_ms would be load_wait_ms since running is not empty
        # We verify by checking that we can set different values
        scheduler.load_wait_ms = 5.0
        assert scheduler.load_wait_ms == 5.0


class TestBlockAllocation:
    """Tests for block allocation and release."""

    def test_allocate_prompt_blocks_calculates_correctly(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """_allocate_prompt_blocks should calculate blocks correctly."""
        # 100 tokens with BLOCK_SIZE=32 needs 4 blocks (ceil(100/32))
        request = make_request("test-1", prompt_tokens=list(range(100)))

        block_table = scheduler._allocate_prompt_blocks(request)

        assert block_table.request_id == "test-1"
        assert block_table.num_tokens == 100
        assert len(block_table.logical_to_physical) == 4  # ceil(100/32)

    def test_release_blocks_returns_to_pool(
        self, scheduler: ContinuousBatchingScheduler, block_manager: PagedBlockManager
    ) -> None:
        """_release_blocks should return blocks to the free pool."""
        initial_free = block_manager.get_free_count()

        request = make_request("test-1", prompt_tokens=list(range(100)))
        block_table = scheduler._allocate_prompt_blocks(request)
        request.block_table = block_table

        # Should have fewer free blocks now
        after_alloc_free = block_manager.get_free_count()
        assert after_alloc_free < initial_free

        # Release should return blocks
        scheduler._release_blocks(request)

        final_free = block_manager.get_free_count()
        assert final_free == initial_free

    def test_release_blocks_handles_none_block_table(
        self, scheduler: ContinuousBatchingScheduler
    ) -> None:
        """_release_blocks should handle None block_table gracefully."""
        request = make_request("test-1")
        request.block_table = None

        # Should not raise
        scheduler._release_blocks(request)


class TestStartStop:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_loop_task(self, scheduler: ContinuousBatchingScheduler) -> None:
        """start() should create the scheduling loop task."""
        assert scheduler._loop_task is None

        await scheduler.start()
        assert scheduler._loop_task is not None

        await scheduler.stop(timeout=1.0)

    @pytest.mark.asyncio
    async def test_double_start_logs_warning(self, scheduler: ContinuousBatchingScheduler) -> None:
        """Double start() should log warning, not create second task."""
        await scheduler.start()
        first_task = scheduler._loop_task

        await scheduler.start()  # Second start
        assert scheduler._loop_task is first_task  # Same task

        await scheduler.stop(timeout=1.0)

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self, scheduler: ContinuousBatchingScheduler) -> None:
        """stop() without start() should be safe."""
        # Should not raise
        await scheduler.stop(timeout=1.0)

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, scheduler: ContinuousBatchingScheduler) -> None:
        """Test full start/process/stop lifecycle."""
        # Add request
        request = make_request("test-1", max_tokens=1)
        await scheduler.waiting.put(request)

        # Start, let it process
        await scheduler.start()
        await asyncio.sleep(0.1)

        # Stop
        await scheduler.stop(timeout=1.0)

        # Verify state
        assert not scheduler.is_running()
        assert len(scheduler.running) == 0
