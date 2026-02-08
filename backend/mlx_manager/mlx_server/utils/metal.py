"""Metal GPU thread affinity utilities.

Provides helpers for running MLX operations on a dedicated thread,
which is required because Metal GPU operations have thread affinity.

Two patterns are supported:
- run_on_metal_thread: for single-result operations (e.g. embeddings, non-streaming completions)
- stream_from_metal_thread: for streaming operations (e.g. token-by-token generation)
"""

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable, Iterator
from queue import Empty, Queue
from typing import TypeVar

from mlx_manager.mlx_server.utils.memory import clear_cache

T = TypeVar("T")

_SENTINEL = object()  # Marks end of stream


async def run_on_metal_thread(
    fn: Callable[[], T],
    timeout: float = 300.0,
    error_context: str | None = None,
) -> T:
    """Run *fn* on a dedicated thread that owns the Metal GPU context.

    Args:
        fn: Zero-arg callable executed on the Metal thread. Must return a value.
        timeout: Seconds to wait for the result (default 5 min).
        error_context: If set, exceptions are wrapped in
            ``RuntimeError(f"{error_context}: {exc}")`` with chaining.
            If ``None``, the original exception is re-raised as-is.

    Returns:
        The value returned by *fn*.
    """
    result_queue: Queue[T | Exception] = Queue()

    def _worker() -> None:
        try:
            result_queue.put(fn())
        except Exception as exc:
            result_queue.put(exc)

    gen_thread = threading.Thread(target=_worker, daemon=True)
    gen_thread.start()

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: result_queue.get(timeout=timeout))

        gen_thread.join(timeout=1.0)

        if isinstance(result, Exception):
            if error_context is not None:
                raise RuntimeError(f"{error_context}: {result}") from result
            raise result

        return result  # type: ignore[return-value]
    finally:
        clear_cache()


async def stream_from_metal_thread(
    fn: Callable[[], Iterator[T]],
    poll_interval: float = 0.1,
) -> AsyncGenerator[T, None]:
    """Yield items produced by an iterator running on a dedicated Metal thread.

    Args:
        fn: Zero-arg callable that returns an ``Iterator[T]``.
            Each yielded item is forwarded to the async consumer.
        poll_interval: Seconds between queue polls (default 100 ms).

    Yields:
        Items produced by the iterator, in order.
    """
    item_queue: Queue[T | Exception | object] = Queue()

    def _worker() -> None:
        try:
            for item in fn():
                item_queue.put(item)
        except Exception as exc:
            item_queue.put(exc)
        finally:
            item_queue.put(_SENTINEL)

    gen_thread = threading.Thread(target=_worker, daemon=True)
    gen_thread.start()

    try:
        loop = asyncio.get_running_loop()

        while True:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: item_queue.get(timeout=poll_interval),
                )
            except Empty:
                continue

            if result is _SENTINEL:
                break

            if isinstance(result, Exception):
                raise result

            yield result  # type: ignore[misc]

        gen_thread.join(timeout=1.0)
    finally:
        clear_cache()
