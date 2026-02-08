"""Tests for Metal GPU thread affinity utilities."""

import threading
from unittest.mock import patch

import pytest

from mlx_manager.mlx_server.utils.metal import (
    _SENTINEL,
    run_on_metal_thread,
    stream_from_metal_thread,
)

# ---------------------------------------------------------------------------
# run_on_metal_thread
# ---------------------------------------------------------------------------


class TestRunOnMetalThread:
    """Tests for the single-result utility."""

    @pytest.mark.asyncio
    async def test_returns_scalar(self):
        result = await run_on_metal_thread(lambda: 42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        result = await run_on_metal_thread(lambda: ([1.0, 2.0], 5))
        assert result == ([1.0, 2.0], 5)

    @pytest.mark.asyncio
    async def test_returns_none(self):
        result = await run_on_metal_thread(lambda: None)
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_bare_raise(self):
        """Without error_context, original exception is re-raised."""

        def boom():
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            await run_on_metal_thread(boom)

    @pytest.mark.asyncio
    async def test_exception_wrapped(self):
        """With error_context, exception is wrapped in RuntimeError with chaining."""

        def boom():
            raise ValueError("inner")

        with pytest.raises(RuntimeError, match="Generation failed: inner") as exc_info:
            await run_on_metal_thread(boom, error_context="Generation failed")
        assert isinstance(exc_info.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Queue.get raises Empty (via timeout) which propagates."""
        import queue

        def block_forever():
            import time

            time.sleep(10)
            return "never"

        with pytest.raises(queue.Empty):
            await run_on_metal_thread(block_forever, timeout=0.1)

    @pytest.mark.asyncio
    async def test_runs_on_separate_thread(self):
        main_thread = threading.current_thread().ident

        result = await run_on_metal_thread(lambda: threading.current_thread().ident)
        assert result != main_thread

    @pytest.mark.asyncio
    async def test_thread_is_daemon(self):
        result = await run_on_metal_thread(lambda: threading.current_thread().daemon)
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_cache_called_on_success(self):
        with patch("mlx_manager.mlx_server.utils.metal.clear_cache") as mock_clear:
            await run_on_metal_thread(lambda: "ok")
            mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache_called_on_error(self):
        with patch("mlx_manager.mlx_server.utils.metal.clear_cache") as mock_clear:
            with pytest.raises(RuntimeError):
                await run_on_metal_thread(
                    lambda: (_ for _ in ()).throw(RuntimeError("fail")) and None  # noqa: B018
                )
            # clear_cache is called in finally regardless
            mock_clear.assert_called_once()


# ---------------------------------------------------------------------------
# stream_from_metal_thread
# ---------------------------------------------------------------------------


class TestStreamFromMetalThread:
    """Tests for the streaming utility."""

    @pytest.mark.asyncio
    async def test_yields_items_in_order(self):
        def produce():
            yield "a"
            yield "b"
            yield "c"

        items = [item async for item in stream_from_metal_thread(produce)]
        assert items == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_yields_tuples(self):
        def produce():
            yield ("tok1", 1, False)
            yield ("tok2", 2, True)

        items = [item async for item in stream_from_metal_thread(produce)]
        assert items == [("tok1", 1, False), ("tok2", 2, True)]

    @pytest.mark.asyncio
    async def test_empty_iterator(self):
        def produce():
            return iter([])

        items = [item async for item in stream_from_metal_thread(produce)]
        assert items == []

    @pytest.mark.asyncio
    async def test_exception_propagated(self):
        def produce():
            yield "ok"
            raise ValueError("stream error")

        items = []
        with pytest.raises(ValueError, match="stream error"):
            async for item in stream_from_metal_thread(produce):
                items.append(item)
        assert items == ["ok"]

    @pytest.mark.asyncio
    async def test_clear_cache_called_on_success(self):
        def produce():
            yield 1

        with patch("mlx_manager.mlx_server.utils.metal.clear_cache") as mock_clear:
            _ = [item async for item in stream_from_metal_thread(produce)]
            mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache_called_on_error(self):
        def produce():
            raise RuntimeError("fail")
            yield  # noqa: RET503 — unreachable, needed for generator type

        with patch("mlx_manager.mlx_server.utils.metal.clear_cache") as mock_clear:
            with pytest.raises(RuntimeError):
                async for _ in stream_from_metal_thread(produce):
                    pass
            mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_runs_on_separate_thread(self):
        main_thread = threading.current_thread().ident

        def produce():
            yield threading.current_thread().ident

        items = [item async for item in stream_from_metal_thread(produce)]
        assert len(items) == 1
        assert items[0] != main_thread

    @pytest.mark.asyncio
    async def test_sentinel_is_private(self):
        """Sentinel is a private module-level object, not None."""
        assert _SENTINEL is not None
        assert type(_SENTINEL) is object
