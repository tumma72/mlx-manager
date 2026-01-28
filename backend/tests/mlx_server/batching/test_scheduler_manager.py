"""Tests for SchedulerManager singleton and per-model scheduler management."""

import asyncio

import pytest

from mlx_manager.mlx_server.services.batching import (
    Priority,
    SchedulerManager,
    get_scheduler_manager,
    init_scheduler_manager,
    reset_scheduler_manager,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before and after each test."""
    reset_scheduler_manager()
    yield
    reset_scheduler_manager()


class TestSchedulerManagerSingleton:
    """Test singleton pattern for SchedulerManager."""

    def test_init_creates_manager(self):
        """init_scheduler_manager creates a new manager instance."""
        mgr = init_scheduler_manager()
        assert mgr is not None
        assert isinstance(mgr, SchedulerManager)

    def test_get_returns_same_instance(self):
        """get_scheduler_manager returns the initialized instance."""
        mgr1 = init_scheduler_manager()
        mgr2 = get_scheduler_manager()
        assert mgr1 is mgr2

    def test_get_raises_if_not_initialized(self):
        """get_scheduler_manager raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            get_scheduler_manager()

    def test_reset_clears_singleton(self):
        """reset_scheduler_manager clears the singleton."""
        init_scheduler_manager()
        reset_scheduler_manager()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_scheduler_manager()

    def test_init_with_custom_config(self):
        """init_scheduler_manager accepts custom configuration."""
        mgr = init_scheduler_manager(
            block_pool_size=500,
            max_batch_size=4,
        )
        assert mgr.block_pool_size == 500
        assert mgr.max_batch_size == 4


class TestSchedulerManagerGetScheduler:
    """Test get_scheduler creates per-model schedulers."""

    @pytest.mark.asyncio
    async def test_get_scheduler_creates_new(self):
        """get_scheduler creates a new scheduler for unknown model."""
        mgr = init_scheduler_manager()
        scheduler = await mgr.get_scheduler("test-model")
        assert scheduler is not None
        assert scheduler.model_id == "test-model"
        # Cleanup
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_get_scheduler_returns_same(self):
        """get_scheduler returns the same scheduler for same model."""
        mgr = init_scheduler_manager()
        scheduler1 = await mgr.get_scheduler("test-model")
        scheduler2 = await mgr.get_scheduler("test-model")
        assert scheduler1 is scheduler2
        # Cleanup
        await scheduler1.stop()

    @pytest.mark.asyncio
    async def test_get_scheduler_different_models(self):
        """get_scheduler creates different schedulers for different models."""
        mgr = init_scheduler_manager()
        scheduler1 = await mgr.get_scheduler("model-a")
        scheduler2 = await mgr.get_scheduler("model-b")
        assert scheduler1 is not scheduler2
        assert scheduler1.model_id == "model-a"
        assert scheduler2.model_id == "model-b"
        # Cleanup
        await scheduler1.stop()
        await scheduler2.stop()

    @pytest.mark.asyncio
    async def test_scheduler_uses_manager_config(self):
        """Created schedulers use manager's configuration."""
        mgr = init_scheduler_manager(
            block_pool_size=200,
            max_batch_size=2,
        )
        scheduler = await mgr.get_scheduler("test-model")
        assert scheduler.max_batch_size == 2
        # Cleanup
        await scheduler.stop()


class TestSchedulerManagerPriority:
    """Test priority determination logic."""

    def test_default_priority_is_normal(self):
        """Default priority for standard requests is NORMAL."""
        mgr = init_scheduler_manager()
        priority = mgr.get_priority_for_request(None, "/v1/chat/completions")
        assert priority == Priority.NORMAL

    def test_batch_endpoint_returns_low(self):
        """Batch endpoints force LOW priority."""
        mgr = init_scheduler_manager()
        priority = mgr.get_priority_for_request(None, "/v1/batch/completions")
        assert priority == Priority.LOW

    def test_batch_endpoint_prefix_match(self):
        """Any /v1/batch/* endpoint returns LOW priority."""
        mgr = init_scheduler_manager()
        priority = mgr.get_priority_for_request(None, "/v1/batch/embeddings")
        assert priority == Priority.LOW

    def test_api_key_doesnt_change_priority_yet(self):
        """API key doesn't change priority (placeholder)."""
        mgr = init_scheduler_manager()
        # With API key, still returns NORMAL (tier lookup is placeholder)
        priority = mgr.get_priority_for_request("sk-test123", "/v1/chat/completions")
        assert priority == Priority.NORMAL

    def test_completions_endpoint_normal(self):
        """Standard completions endpoint returns NORMAL."""
        mgr = init_scheduler_manager()
        priority = mgr.get_priority_for_request(None, "/v1/completions")
        assert priority == Priority.NORMAL


class TestSchedulerManagerShutdown:
    """Test graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_schedulers(self):
        """shutdown() stops all managed schedulers."""
        mgr = init_scheduler_manager()

        # Create multiple schedulers
        scheduler1 = await mgr.get_scheduler("model-a")
        scheduler2 = await mgr.get_scheduler("model-b")

        # Verify they're running
        assert scheduler1.is_running()
        assert scheduler2.is_running()

        # Shutdown
        await mgr.shutdown()

        # Verify they're stopped
        assert not scheduler1.is_running()
        assert not scheduler2.is_running()

    @pytest.mark.asyncio
    async def test_shutdown_clears_internal_state(self):
        """shutdown() clears internal scheduler and block manager dicts."""
        mgr = init_scheduler_manager()

        await mgr.get_scheduler("model-a")
        await mgr.get_scheduler("model-b")

        assert len(mgr._schedulers) == 2
        assert len(mgr._block_managers) == 2

        await mgr.shutdown()

        assert len(mgr._schedulers) == 0
        assert len(mgr._block_managers) == 0

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """shutdown() can be called multiple times safely."""
        mgr = init_scheduler_manager()
        await mgr.get_scheduler("test-model")

        await mgr.shutdown()
        await mgr.shutdown()  # Should not raise


class TestSchedulerManagerConfigure:
    """Test configure_scheduler method."""

    @pytest.mark.asyncio
    async def test_configure_creates_scheduler_if_needed(self):
        """configure_scheduler creates scheduler if not exists."""
        mgr = init_scheduler_manager()

        # Scheduler doesn't exist yet
        assert "test-model" not in mgr._schedulers

        # Configure (with stub resources)
        await mgr.configure_scheduler("test-model", None, None, None)

        # Scheduler now exists
        assert "test-model" in mgr._schedulers

        # Cleanup
        await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_configure_idempotent(self):
        """configure_scheduler can be called multiple times."""
        mgr = init_scheduler_manager()

        await mgr.configure_scheduler("test-model", None, None, None)
        scheduler1 = await mgr.get_scheduler("test-model")

        await mgr.configure_scheduler("test-model", None, None, None)
        scheduler2 = await mgr.get_scheduler("test-model")

        # Same scheduler instance
        assert scheduler1 is scheduler2

        # Cleanup
        await mgr.shutdown()


class TestSchedulerManagerConcurrency:
    """Test thread-safety of scheduler manager."""

    @pytest.mark.asyncio
    async def test_concurrent_get_scheduler(self):
        """Concurrent get_scheduler calls for same model return same scheduler."""
        mgr = init_scheduler_manager()

        # Create multiple concurrent requests for same model
        schedulers = await asyncio.gather(*[
            mgr.get_scheduler("test-model") for _ in range(10)
        ])

        # All should be the same instance
        first = schedulers[0]
        for s in schedulers[1:]:
            assert s is first

        # Cleanup
        await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_get_different_models(self):
        """Concurrent get_scheduler calls for different models work correctly."""
        mgr = init_scheduler_manager()

        # Create concurrent requests for different models
        models = [f"model-{i}" for i in range(5)]
        schedulers = await asyncio.gather(*[
            mgr.get_scheduler(model) for model in models
        ])

        # Each should have correct model_id
        for i, scheduler in enumerate(schedulers):
            assert scheduler.model_id == f"model-{i}"

        # All should be different instances
        assert len(set(id(s) for s in schedulers)) == 5

        # Cleanup
        await mgr.shutdown()
