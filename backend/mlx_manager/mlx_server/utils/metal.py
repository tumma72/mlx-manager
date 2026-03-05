"""Metal GPU thread affinity utilities.

Provides helpers for running MLX operations on a **persistent** dedicated thread,
which is required because Metal GPU operations have thread affinity.

A single ``MetalWorker`` daemon thread stays alive across requests, maintaining
the Metal GPU context. Concurrent requests are serialized through a job queue
so that exactly one operation uses the GPU at a time.

Two patterns are supported:
- run_on_metal_thread: for single-result operations (e.g. embeddings, non-streaming completions)
- stream_from_metal_thread: for streaming operations (e.g. token-by-token generation)

stream_from_metal_thread uses asyncio.Event-based signaling: the producer thread
calls loop.call_soon_threadsafe(event.set) after each put, waking the consumer
with near-zero latency instead of relying on busy-polling.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator, Callable, Iterator
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, TypeVar

from mlx_manager.mlx_server.utils.memory import clear_cache

logger = logging.getLogger(__name__)

T = TypeVar("T")

_SENTINEL = object()  # Marks end of stream


# ---------------------------------------------------------------------------
# Job descriptors
# ---------------------------------------------------------------------------


@dataclass
class _RunJob:
    """A single-result job submitted to the Metal worker."""

    fn: Callable[[], Any]
    result_queue: Queue[Any] = field(default_factory=Queue)


@dataclass
class _StreamJob:
    """A streaming job submitted to the Metal worker."""

    fn: Callable[[], Iterator[Any]]
    item_queue: Queue[Any] = field(default_factory=Queue)
    loop: asyncio.AbstractEventLoop | None = None
    ready: asyncio.Event | None = None


_SHUTDOWN = object()  # Poison pill for the job queue


# ---------------------------------------------------------------------------
# MetalWorker
# ---------------------------------------------------------------------------


class MetalWorker:
    """Persistent daemon thread that owns the Metal GPU context.

    All GPU work is submitted as jobs and executed sequentially on the single
    worker thread, preserving Metal thread affinity across requests.
    """

    def __init__(self) -> None:
        self._job_queue: Queue[_RunJob | _StreamJob | object] = Queue()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="metal-worker",
        )
        self._thread.start()
        logger.debug("MetalWorker started (thread=%s)", self._thread.ident)

    # -- internal loop -------------------------------------------------------

    def _run_loop(self) -> None:
        """Process jobs sequentially until shutdown."""
        while True:
            job = self._job_queue.get()

            if job is _SHUTDOWN:
                logger.debug("MetalWorker received shutdown signal")
                break

            if isinstance(job, _RunJob):
                self._handle_run(job)
            elif isinstance(job, _StreamJob):
                self._handle_stream(job)

    @staticmethod
    def _handle_run(job: _RunJob) -> None:
        try:
            job.result_queue.put(job.fn())
        except Exception as exc:
            job.result_queue.put(exc)

    @staticmethod
    def _handle_stream(job: _StreamJob) -> None:
        loop = job.loop
        ready = job.ready
        try:
            for item in job.fn():
                job.item_queue.put(item)
                if loop is not None and ready is not None:
                    loop.call_soon_threadsafe(ready.set)
        except Exception as exc:
            job.item_queue.put(exc)
            if loop is not None and ready is not None:
                loop.call_soon_threadsafe(ready.set)
        finally:
            job.item_queue.put(_SENTINEL)
            if loop is not None and ready is not None:
                loop.call_soon_threadsafe(ready.set)

    # -- public submission API -----------------------------------------------

    def submit_run(self, fn: Callable[[], Any]) -> Queue[Any]:
        """Submit a single-result job and return its result queue."""
        job = _RunJob(fn=fn)
        self._job_queue.put(job)
        return job.result_queue

    def submit_stream(
        self,
        fn: Callable[[], Iterator[Any]],
        loop: asyncio.AbstractEventLoop,
        ready: asyncio.Event,
    ) -> Queue[Any]:
        """Submit a streaming job and return its item queue."""
        job = _StreamJob(fn=fn, item_queue=Queue(), loop=loop, ready=ready)
        self._job_queue.put(job)
        return job.item_queue

    @property
    def thread(self) -> threading.Thread:
        """The underlying worker thread (for introspection / testing)."""
        return self._thread

    def shutdown(self) -> None:
        """Signal the worker to exit and wait for it to finish."""
        self._job_queue.put(_SHUTDOWN)
        self._thread.join(timeout=5.0)
        logger.debug("MetalWorker shut down")


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_worker: MetalWorker | None = None
_lock = threading.Lock()


def get_metal_worker() -> MetalWorker:
    """Get or create the MetalWorker singleton."""
    global _worker
    if _worker is None:
        with _lock:
            if _worker is None:
                _worker = MetalWorker()
    return _worker


def reset_metal_worker() -> None:
    """Shut down and discard the MetalWorker singleton (for testing / shutdown)."""
    global _worker
    with _lock:
        if _worker is not None:
            _worker.shutdown()
            _worker = None


# ---------------------------------------------------------------------------
# Public async helpers (unchanged signatures)
# ---------------------------------------------------------------------------


async def run_on_metal_thread(
    fn: Callable[[], T],
    timeout: float = 300.0,
    error_context: str | None = None,
) -> T:
    """Run *fn* on the persistent Metal GPU thread.

    Args:
        fn: Zero-arg callable executed on the Metal thread. Must return a value.
        timeout: Seconds to wait for the result (default 5 min).
        error_context: If set, exceptions are wrapped in
            ``RuntimeError(f"{error_context}: {exc}")`` with chaining.
            If ``None``, the original exception is re-raised as-is.

    Returns:
        The value returned by *fn*.
    """
    worker = get_metal_worker()
    result_queue = worker.submit_run(fn)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: result_queue.get(timeout=timeout))

        if isinstance(result, Exception):
            if error_context is not None:
                raise RuntimeError(f"{error_context}: {result}") from result
            raise result

        return result
    finally:
        clear_cache()


async def stream_from_metal_thread(
    fn: Callable[[], Iterator[T]],
    poll_interval: float = 0.05,
) -> AsyncGenerator[T, None]:
    """Yield items produced by an iterator running on the persistent Metal thread.

    Args:
        fn: Zero-arg callable that returns an ``Iterator[T]``.
            Each yielded item is forwarded to the async consumer.
        poll_interval: Fallback timeout (seconds) for event wait. Only used
            if the thread-safe event signal is missed. Default 50 ms.

    Yields:
        Items produced by the iterator, in order.
    """
    loop = asyncio.get_running_loop()
    ready = asyncio.Event()
    worker = get_metal_worker()
    item_queue = worker.submit_stream(fn, loop, ready)

    try:
        while True:
            # Wait for producer signal (with fallback timeout)
            try:
                await asyncio.wait_for(ready.wait(), timeout=poll_interval)
            except TimeoutError:
                pass
            ready.clear()

            # Drain all available items
            while not item_queue.empty():
                result = item_queue.get_nowait()

                if result is _SENTINEL:
                    return

                if isinstance(result, Exception):
                    raise result

                yield result
    finally:
        clear_cache()
