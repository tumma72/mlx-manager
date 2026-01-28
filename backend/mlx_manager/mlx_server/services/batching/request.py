"""BatchRequest dataclass for continuous batching.

Represents a single generation request with state management,
streaming support, and lifecycle tracking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlx_manager.mlx_server.services.batching.types import Priority, RequestStatus

if TYPE_CHECKING:
    from mlx_manager.mlx_server.services.batching.block import BlockTable


@dataclass
class BatchRequest:
    """A request in the continuous batching queue.

    Tracks request state through its lifecycle from submission
    to completion, including generated tokens and timing information.
    """

    # Required fields
    request_id: str
    model_id: str
    prompt_tokens: list[int]
    max_tokens: int
    priority: Priority = Priority.NORMAL

    # State tracking
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int] = field(default_factory=list)

    # Streaming output - None signals completion
    output_queue: asyncio.Queue[dict[str, Any] | None] = field(
        default_factory=asyncio.Queue,
        repr=False,
    )

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None

    # Block management (for paged KV cache)
    # Can be BlockTable (new system) or list[int] (legacy)
    block_table: BlockTable | list[int] | None = None

    @property
    def base_priority(self) -> int:
        """Get the base priority value for queue ordering.

        Returns:
            Integer priority value (lower = higher priority)
        """
        return self.priority.value

    @property
    def is_complete(self) -> bool:
        """Check if request has finished generating.

        Returns:
            True if max_tokens reached or status is terminal
        """
        # Terminal statuses
        if self.status in (RequestStatus.COMPLETED, RequestStatus.CANCELLED):
            return True

        # Max tokens reached
        return len(self.generated_tokens) >= self.max_tokens

    @property
    def effective_tokens(self) -> int:
        """Get total token count (prompt + generated).

        Returns:
            Combined length of prompt and generated tokens
        """
        return len(self.prompt_tokens) + len(self.generated_tokens)

    def add_token(self, token_id: int) -> None:
        """Add a generated token.

        Args:
            token_id: The generated token ID
        """
        self.generated_tokens.append(token_id)

    def mark_prefilling(self) -> None:
        """Transition to PREFILLING state."""
        self.status = RequestStatus.PREFILLING
        if self.started_at is None:
            self.started_at = time.time()

    def mark_running(self) -> None:
        """Transition to RUNNING (decode) state."""
        self.status = RequestStatus.RUNNING

    def mark_completed(self) -> None:
        """Transition to COMPLETED state."""
        self.status = RequestStatus.COMPLETED

    def mark_cancelled(self) -> None:
        """Transition to CANCELLED state."""
        self.status = RequestStatus.CANCELLED
