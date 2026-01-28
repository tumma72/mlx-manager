"""Priority queue with aging mechanism for continuous batching.

This module implements a priority queue that prevents starvation
by gradually increasing the effective priority of waiting requests
over time.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field

from mlx_manager.mlx_server.services.batching.request import BatchRequest


@dataclass(order=True)
class QueueEntry:
    """Entry in the priority queue with comparison support.

    Uses effective_priority for primary ordering, entry_count as
    FIFO tie-breaker for same-priority requests.
    """

    effective_priority: float  # Lower = higher priority
    entry_count: int  # FIFO tie-breaker
    request: BatchRequest = field(compare=False)
    entry_time: float = field(compare=False)


class PriorityQueueWithAging:
    """Priority queue with aging to prevent starvation.

    Aging mechanism: as requests wait longer, their effective priority
    decreases (higher priority) to ensure they eventually get processed.

    Formula: effective_priority = base_priority - (wait_time * aging_rate)

    With default rate=0.1:
    - LOW (2) becomes NORMAL (1) after 10 seconds
    - LOW (2) becomes HIGH (0) after 20 seconds

    Thread-safe via asyncio.Lock for async contexts.
    """

    def __init__(self, aging_rate: float = 0.1) -> None:
        """Initialize the priority queue.

        Args:
            aging_rate: Rate at which priority improves per second.
                       Higher rate = faster promotion of waiting requests.
        """
        self._heap: list[QueueEntry] = []
        self._entry_counter: int = 0
        self._aging_rate = aging_rate
        self._lock = asyncio.Lock()

    def _update_priorities(self) -> None:
        """Recalculate effective priorities based on wait time.

        Updates all entries in the heap based on current time,
        then restores heap property.
        """
        current_time = time.time()
        for entry in self._heap:
            wait_time = current_time - entry.entry_time
            entry.effective_priority = (
                entry.request.base_priority - (wait_time * self._aging_rate)
            )
        heapq.heapify(self._heap)

    async def put(self, request: BatchRequest) -> None:
        """Add a request to the queue.

        Args:
            request: The BatchRequest to queue
        """
        async with self._lock:
            entry = QueueEntry(
                effective_priority=float(request.base_priority),
                entry_count=self._entry_counter,
                request=request,
                entry_time=time.time(),
            )
            self._entry_counter += 1
            heapq.heappush(self._heap, entry)

    async def get(self) -> BatchRequest:
        """Get the highest priority request, blocking if empty.

        Returns:
            The BatchRequest with highest effective priority

        Raises:
            IndexError: If queue is empty (use empty() to check first)
        """
        async with self._lock:
            if not self._heap:
                raise IndexError("Queue is empty")

            # Update priorities before selecting
            self._update_priorities()

            entry = heapq.heappop(self._heap)
            return entry.request

    async def peek(self) -> BatchRequest | None:
        """View the top request without removing it.

        Returns:
            The highest priority BatchRequest, or None if empty
        """
        async with self._lock:
            if not self._heap:
                return None

            # Update priorities to get accurate view
            self._update_priorities()

            return self._heap[0].request

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            True if no requests are queued
        """
        return len(self._heap) == 0

    def qsize(self) -> int:
        """Get the number of queued requests.

        Returns:
            Count of requests in the queue
        """
        return len(self._heap)
