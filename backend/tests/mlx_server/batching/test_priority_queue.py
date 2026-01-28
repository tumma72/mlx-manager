"""Tests for PriorityQueueWithAging."""

from unittest.mock import patch

import pytest

from mlx_manager.mlx_server.services.batching import (
    BatchRequest,
    Priority,
    PriorityQueueWithAging,
)


@pytest.fixture
def queue() -> PriorityQueueWithAging:
    """Create a fresh priority queue for each test."""
    return PriorityQueueWithAging(aging_rate=0.1)


def make_request(request_id: str, priority: Priority = Priority.NORMAL) -> BatchRequest:
    """Helper to create test requests."""
    return BatchRequest(
        request_id=request_id,
        model_id="test-model",
        prompt_tokens=[1, 2, 3],
        max_tokens=100,
        priority=priority,
    )


class TestBasicOperations:
    """Tests for basic queue operations."""

    @pytest.mark.asyncio
    async def test_put_and_get_single_request(
        self, queue: PriorityQueueWithAging
    ) -> None:
        """Should be able to put and get a single request."""
        request = make_request("test-1")

        await queue.put(request)
        assert queue.qsize() == 1

        got = await queue.get()
        assert got.request_id == "test-1"
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_empty_and_qsize(self, queue: PriorityQueueWithAging) -> None:
        """empty() and qsize() should reflect queue state."""
        assert queue.empty()
        assert queue.qsize() == 0

        await queue.put(make_request("r1"))
        assert not queue.empty()
        assert queue.qsize() == 1

        await queue.put(make_request("r2"))
        assert queue.qsize() == 2

        await queue.get()
        assert queue.qsize() == 1

        await queue.get()
        assert queue.empty()
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_get_empty_raises_index_error(
        self, queue: PriorityQueueWithAging
    ) -> None:
        """get() on empty queue should raise IndexError."""
        with pytest.raises(IndexError, match="Queue is empty"):
            await queue.get()

    @pytest.mark.asyncio
    async def test_peek_returns_top_without_removing(
        self, queue: PriorityQueueWithAging
    ) -> None:
        """peek() should return top item without removing it."""
        await queue.put(make_request("test-1"))

        peeked = await queue.peek()
        assert peeked is not None
        assert peeked.request_id == "test-1"
        assert queue.qsize() == 1  # Still in queue

        got = await queue.get()
        assert got.request_id == "test-1"
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_peek_empty_returns_none(
        self, queue: PriorityQueueWithAging
    ) -> None:
        """peek() on empty queue should return None."""
        result = await queue.peek()
        assert result is None


class TestPriorityOrdering:
    """Tests for priority-based ordering."""

    @pytest.mark.asyncio
    async def test_high_priority_returned_first(
        self, queue: PriorityQueueWithAging
    ) -> None:
        """HIGH priority requests should be returned before LOW."""
        # Add in reverse order to ensure priority, not insertion order, matters
        await queue.put(make_request("low", Priority.LOW))
        await queue.put(make_request("normal", Priority.NORMAL))
        await queue.put(make_request("high", Priority.HIGH))

        first = await queue.get()
        assert first.request_id == "high"

        second = await queue.get()
        assert second.request_id == "normal"

        third = await queue.get()
        assert third.request_id == "low"

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(
        self, queue: PriorityQueueWithAging
    ) -> None:
        """Same-priority requests should be FIFO."""
        await queue.put(make_request("first", Priority.NORMAL))
        await queue.put(make_request("second", Priority.NORMAL))
        await queue.put(make_request("third", Priority.NORMAL))

        assert (await queue.get()).request_id == "first"
        assert (await queue.get()).request_id == "second"
        assert (await queue.get()).request_id == "third"


class TestAgingMechanism:
    """Tests for priority aging."""

    @pytest.mark.asyncio
    async def test_aging_promotes_low_to_beat_normal(self) -> None:
        """After sufficient wait time, LOW priority should beat NORMAL.

        With aging_rate=0.1:
        - LOW starts at priority 2
        - After 15s: effective_priority = 2 - (15 * 0.1) = 0.5
        - NORMAL at priority 1 with 0s wait: effective_priority = 1
        - 0.5 < 1, so LOW wins
        """
        queue = PriorityQueueWithAging(aging_rate=0.1)

        base_time = 1000.0

        # Add LOW priority request at base time
        with patch("time.time", return_value=base_time):
            await queue.put(make_request("low-old", Priority.LOW))

        # Add NORMAL priority request 15 seconds later
        with patch("time.time", return_value=base_time + 15.0):
            await queue.put(make_request("normal-new", Priority.NORMAL))

        # Get should return the aged LOW request first (at the later time)
        with patch("time.time", return_value=base_time + 15.0):
            first = await queue.get()
            assert first.request_id == "low-old", (
                f"Expected low-old to win due to aging, got {first.request_id}"
            )

            second = await queue.get()
            assert second.request_id == "normal-new"

    @pytest.mark.asyncio
    async def test_no_aging_effect_immediate_get(self) -> None:
        """Without wait time, priority ordering should be standard."""
        queue = PriorityQueueWithAging(aging_rate=0.1)

        fixed_time = 1000.0

        with patch("time.time", return_value=fixed_time):
            await queue.put(make_request("low", Priority.LOW))
            await queue.put(make_request("high", Priority.HIGH))

            # No time has passed, so no aging
            first = await queue.get()
            assert first.request_id == "high"

    @pytest.mark.asyncio
    async def test_custom_aging_rate(self) -> None:
        """Custom aging rate should affect promotion speed.

        With aging_rate=1.0:
        - LOW (2) becomes HIGH (0) after just 2 seconds
        """
        queue = PriorityQueueWithAging(aging_rate=1.0)  # Fast aging

        base_time = 1000.0

        with patch("time.time", return_value=base_time):
            await queue.put(make_request("low", Priority.LOW))

        # After 2.5 seconds, LOW effective = 2 - 2.5 = -0.5
        # This beats HIGH at 0
        with patch("time.time", return_value=base_time + 2.5):
            await queue.put(make_request("high", Priority.HIGH))

        with patch("time.time", return_value=base_time + 2.5):
            first = await queue.get()
            assert first.request_id == "low"

    @pytest.mark.asyncio
    async def test_aging_preserves_fifo_at_equal_effective_priority(self) -> None:
        """When aging makes priorities equal, FIFO should still apply."""
        queue = PriorityQueueWithAging(aging_rate=0.1)

        base_time = 1000.0

        # Add LOW at base time
        with patch("time.time", return_value=base_time):
            await queue.put(make_request("low", Priority.LOW))

        # Add NORMAL 10 seconds later - at this point:
        # LOW effective = 2 - (10 * 0.1) = 1.0
        # NORMAL effective = 1 - (0 * 0.1) = 1.0
        # Equal, so FIFO: LOW was added first
        with patch("time.time", return_value=base_time + 10.0):
            await queue.put(make_request("normal", Priority.NORMAL))

        with patch("time.time", return_value=base_time + 10.0):
            first = await queue.get()
            assert first.request_id == "low"  # Added first, same effective priority


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_many_requests_ordering(
        self, queue: PriorityQueueWithAging
    ) -> None:
        """Large number of requests should maintain correct ordering."""
        # Add 10 requests with mixed priorities
        for i in range(10):
            priority = [Priority.HIGH, Priority.NORMAL, Priority.LOW][i % 3]
            await queue.put(make_request(f"r{i}", priority))

        # All HIGH should come out first, then NORMAL, then LOW
        priorities_seen = []
        while not queue.empty():
            r = await queue.get()
            priorities_seen.append(r.priority)

        # Verify non-increasing priority values (remembering lower = higher priority)
        for i in range(len(priorities_seen) - 1):
            assert priorities_seen[i].value <= priorities_seen[i + 1].value
