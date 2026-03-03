"""Model loading progress tracking with SSE subscriber support.

Manages per-model loading progress events and delivers them to SSE subscribers.
Late-joining subscribers receive the latest event for catch-up.
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum

from loguru import logger
from pydantic import BaseModel


class LoadingEventType(StrEnum):
    """Types of model loading progress events."""

    DOWNLOAD_PROGRESS = "download_progress"
    WEIGHTS_LOADING = "weights_loading"
    ADAPTER_INIT = "adapter_init"
    READY = "ready"
    ERROR = "error"


class LoadingEvent(BaseModel):
    """A single loading progress event."""

    event: LoadingEventType
    model_id: str
    progress: float | None = None  # 0-100 percent
    message: str | None = None
    timestamp: float


class LoadingProgressManager:
    """Manages per-model loading progress and SSE subscribers."""

    def __init__(self) -> None:
        # model_id -> list of subscriber queues
        self._subscribers: dict[str, list[asyncio.Queue[LoadingEvent | None]]] = {}
        # model_id -> latest event (for late joiners)
        self._latest: dict[str, LoadingEvent] = {}

    def subscribe(self, model_id: str) -> asyncio.Queue[LoadingEvent | None]:
        """Subscribe to loading events for a model. Returns a queue to await."""
        queue: asyncio.Queue[LoadingEvent | None] = asyncio.Queue()
        if model_id not in self._subscribers:
            self._subscribers[model_id] = []
        self._subscribers[model_id].append(queue)

        # Send latest event to late joiners
        if model_id in self._latest:
            latest = self._latest[model_id]
            queue.put_nowait(latest)
            # If the latest event is terminal, also send the None sentinel
            if latest.event in (LoadingEventType.READY, LoadingEventType.ERROR):
                queue.put_nowait(None)

        return queue

    def unsubscribe(self, model_id: str, queue: asyncio.Queue[LoadingEvent | None]) -> None:
        """Remove a subscriber."""
        if model_id in self._subscribers:
            try:
                self._subscribers[model_id].remove(queue)
            except ValueError:
                pass
            if not self._subscribers[model_id]:
                del self._subscribers[model_id]

    def emit(
        self,
        model_id: str,
        event_type: LoadingEventType,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        """Emit a loading event to all subscribers."""
        event = LoadingEvent(
            event=event_type,
            model_id=model_id,
            progress=progress,
            message=message,
            timestamp=time.time(),
        )
        self._latest[model_id] = event

        for queue in self._subscribers.get(model_id, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Loading progress queue full for {model_id}")

        # Send None sentinel to signal stream end for terminal events
        if event_type in (LoadingEventType.READY, LoadingEventType.ERROR):
            for queue in self._subscribers.get(model_id, []):
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

    def cleanup(self, model_id: str) -> None:
        """Clean up state for a model."""
        self._latest.pop(model_id, None)
        self._subscribers.pop(model_id, None)


# Module-level singleton
_loading_progress: LoadingProgressManager | None = None


def get_loading_progress() -> LoadingProgressManager:
    """Get or create the loading progress singleton."""
    global _loading_progress
    if _loading_progress is None:
        _loading_progress = LoadingProgressManager()
    return _loading_progress


def reset_loading_progress() -> None:
    """Reset for testing."""
    global _loading_progress
    _loading_progress = None
