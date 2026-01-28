"""Batching types for continuous batching scheduler.

This module defines the foundational enums and types used throughout
the continuous batching system for request lifecycle management.
"""

from enum import IntEnum


class RequestStatus(IntEnum):
    """Status of a batch request through its lifecycle.

    Requests transition through states:
    WAITING -> PREFILLING -> RUNNING -> COMPLETED/CANCELLED
    """

    WAITING = 0  # In queue, waiting to be scheduled
    PREFILLING = 1  # Processing prompt tokens (prefill phase)
    RUNNING = 2  # Generating tokens (decode phase)
    COMPLETED = 3  # Finished successfully
    CANCELLED = 4  # Cancelled by user or system


class Priority(IntEnum):
    """Request priority levels.

    Lower numeric value = higher priority.
    Used by PriorityQueueWithAging for scheduling order.
    """

    HIGH = 0  # Interactive requests, low latency required
    NORMAL = 1  # Default priority for most requests
    LOW = 2  # Background/batch processing requests
