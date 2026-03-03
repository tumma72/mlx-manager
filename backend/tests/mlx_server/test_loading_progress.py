"""Tests for model loading progress SSE service and endpoint."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from mlx_manager.mlx_server.services.loading_progress import (
    LoadingEvent,
    LoadingEventType,
    LoadingProgressManager,
    get_loading_progress,
    reset_loading_progress,
)

# ============================================================================
# LoadingProgressManager Unit Tests
# ============================================================================


class TestLoadingProgressManager:
    """Tests for the LoadingProgressManager class."""

    def setup_method(self) -> None:
        self.mgr = LoadingProgressManager()

    def test_subscribe_creates_queue(self) -> None:
        queue = self.mgr.subscribe("model-a")
        assert isinstance(queue, asyncio.Queue)
        assert "model-a" in self.mgr._subscribers
        assert queue in self.mgr._subscribers["model-a"]

    def test_subscribe_multiple_subscribers(self) -> None:
        q1 = self.mgr.subscribe("model-a")
        q2 = self.mgr.subscribe("model-a")
        assert len(self.mgr._subscribers["model-a"]) == 2
        assert q1 in self.mgr._subscribers["model-a"]
        assert q2 in self.mgr._subscribers["model-a"]

    def test_unsubscribe_removes_queue(self) -> None:
        queue = self.mgr.subscribe("model-a")
        self.mgr.unsubscribe("model-a", queue)
        # Should clean up empty list
        assert "model-a" not in self.mgr._subscribers

    def test_unsubscribe_nonexistent_model(self) -> None:
        """Unsubscribing from a model that doesn't exist should not raise."""
        queue: asyncio.Queue[LoadingEvent | None] = asyncio.Queue()
        self.mgr.unsubscribe("nonexistent", queue)  # Should not raise

    def test_unsubscribe_nonexistent_queue(self) -> None:
        """Unsubscribing a queue that was never subscribed should not raise."""
        self.mgr.subscribe("model-a")
        other_queue: asyncio.Queue[LoadingEvent | None] = asyncio.Queue()
        self.mgr.unsubscribe("model-a", other_queue)  # Should not raise
        assert len(self.mgr._subscribers["model-a"]) == 1

    def test_emit_delivers_to_subscribers(self) -> None:
        q1 = self.mgr.subscribe("model-a")
        q2 = self.mgr.subscribe("model-a")

        self.mgr.emit("model-a", LoadingEventType.WEIGHTS_LOADING, progress=50)

        event1 = q1.get_nowait()
        event2 = q2.get_nowait()
        assert event1 is not None
        assert event1.event == LoadingEventType.WEIGHTS_LOADING
        assert event1.progress == 50
        assert event1.model_id == "model-a"
        assert event2 is not None
        assert event2.event == LoadingEventType.WEIGHTS_LOADING

    def test_emit_stores_latest(self) -> None:
        self.mgr.emit("model-a", LoadingEventType.WEIGHTS_LOADING, progress=25)
        assert "model-a" in self.mgr._latest
        assert self.mgr._latest["model-a"].progress == 25

    def test_emit_with_message(self) -> None:
        queue = self.mgr.subscribe("model-a")
        self.mgr.emit("model-a", LoadingEventType.ERROR, message="Out of memory")

        event = queue.get_nowait()
        assert event is not None
        assert event.event == LoadingEventType.ERROR
        assert event.message == "Out of memory"

    def test_late_joiner_receives_latest_event(self) -> None:
        # Emit before any subscriber
        self.mgr.emit("model-a", LoadingEventType.WEIGHTS_LOADING, progress=75)

        # Late joiner subscribes
        queue = self.mgr.subscribe("model-a")

        # Should have the latest event immediately
        event = queue.get_nowait()
        assert event is not None
        assert event.event == LoadingEventType.WEIGHTS_LOADING
        assert event.progress == 75

    def test_ready_event_sends_none_sentinel(self) -> None:
        queue = self.mgr.subscribe("model-a")

        self.mgr.emit("model-a", LoadingEventType.READY, progress=100)

        # Should get the event followed by None sentinel
        event = queue.get_nowait()
        assert event is not None
        assert event.event == LoadingEventType.READY

        sentinel = queue.get_nowait()
        assert sentinel is None

    def test_error_event_sends_none_sentinel(self) -> None:
        queue = self.mgr.subscribe("model-a")

        self.mgr.emit("model-a", LoadingEventType.ERROR, message="fail")

        event = queue.get_nowait()
        assert event is not None
        assert event.event == LoadingEventType.ERROR

        sentinel = queue.get_nowait()
        assert sentinel is None

    def test_non_terminal_event_no_sentinel(self) -> None:
        queue = self.mgr.subscribe("model-a")

        self.mgr.emit("model-a", LoadingEventType.WEIGHTS_LOADING, progress=50)

        event = queue.get_nowait()
        assert event is not None
        assert event.event == LoadingEventType.WEIGHTS_LOADING

        # Queue should be empty — no sentinel
        assert queue.empty()

    def test_emit_no_subscribers(self) -> None:
        """Emitting with no subscribers should not raise."""
        self.mgr.emit("model-a", LoadingEventType.WEIGHTS_LOADING, progress=50)
        # Should still track latest
        assert "model-a" in self.mgr._latest

    def test_cleanup_removes_state(self) -> None:
        self.mgr.subscribe("model-a")
        self.mgr.emit("model-a", LoadingEventType.WEIGHTS_LOADING, progress=50)

        self.mgr.cleanup("model-a")

        assert "model-a" not in self.mgr._latest
        assert "model-a" not in self.mgr._subscribers

    def test_cleanup_nonexistent_model(self) -> None:
        """Cleanup on a model that doesn't exist should not raise."""
        self.mgr.cleanup("nonexistent")

    def test_event_timestamp(self) -> None:
        before = time.time()
        self.mgr.emit("model-a", LoadingEventType.WEIGHTS_LOADING)
        after = time.time()

        event = self.mgr._latest["model-a"]
        assert before <= event.timestamp <= after

    def test_multiple_models_independent(self) -> None:
        q_a = self.mgr.subscribe("model-a")
        q_b = self.mgr.subscribe("model-b")

        self.mgr.emit("model-a", LoadingEventType.WEIGHTS_LOADING, progress=10)
        self.mgr.emit("model-b", LoadingEventType.ADAPTER_INIT, message="init")

        event_a = q_a.get_nowait()
        assert event_a is not None
        assert event_a.model_id == "model-a"
        assert event_a.progress == 10

        event_b = q_b.get_nowait()
        assert event_b is not None
        assert event_b.model_id == "model-b"
        assert event_b.event == LoadingEventType.ADAPTER_INIT

        # No cross-contamination
        assert q_a.empty()
        assert q_b.empty()


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for module-level singleton lifecycle."""

    def setup_method(self) -> None:
        reset_loading_progress()

    def teardown_method(self) -> None:
        reset_loading_progress()

    def test_get_returns_singleton(self) -> None:
        mgr1 = get_loading_progress()
        mgr2 = get_loading_progress()
        assert mgr1 is mgr2

    def test_reset_clears_singleton(self) -> None:
        mgr1 = get_loading_progress()
        reset_loading_progress()
        mgr2 = get_loading_progress()
        assert mgr1 is not mgr2


# ============================================================================
# LoadingEvent Model Tests
# ============================================================================


class TestLoadingEvent:
    """Tests for the LoadingEvent Pydantic model."""

    def test_serialization(self) -> None:
        event = LoadingEvent(
            event=LoadingEventType.WEIGHTS_LOADING,
            model_id="mlx-community/test-model",
            progress=42.5,
            message="Loading weights",
            timestamp=1234567890.0,
        )
        data = json.loads(event.model_dump_json())
        assert data["event"] == "weights_loading"
        assert data["model_id"] == "mlx-community/test-model"
        assert data["progress"] == 42.5
        assert data["message"] == "Loading weights"
        assert data["timestamp"] == 1234567890.0

    def test_optional_fields(self) -> None:
        event = LoadingEvent(
            event=LoadingEventType.ADAPTER_INIT,
            model_id="test",
            timestamp=0.0,
        )
        assert event.progress is None
        assert event.message is None

    def test_all_event_types(self) -> None:
        for event_type in LoadingEventType:
            event = LoadingEvent(
                event=event_type,
                model_id="test",
                timestamp=0.0,
            )
            assert event.event == event_type


# ============================================================================
# SSE Endpoint Tests
# ============================================================================


class TestLoadingProgressEndpoint:
    """Tests for the SSE endpoint in admin router."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> None:
        reset_loading_progress()
        yield  # type: ignore[misc]
        reset_loading_progress()

    @pytest.fixture
    def app(self):
        """Create a minimal FastAPI app with the admin router for testing."""
        from fastapi import FastAPI

        from mlx_manager.mlx_server.api.v1.admin import router

        test_app = FastAPI()
        # Mount without admin token dependency for testing
        test_app.include_router(router, prefix="/v1")
        return test_app

    @pytest.mark.asyncio
    async def test_endpoint_returns_sse_content_type(self, app) -> None:
        """Endpoint should return text/event-stream content type."""
        mgr = get_loading_progress()

        # Pre-emit a terminal event so the stream will end
        mgr.emit("test-model", LoadingEventType.READY, progress=100, message="Done")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/admin/models/test-model/loading-progress")
            assert response.headers["content-type"].startswith("text/event-stream")

    @pytest.mark.asyncio
    async def test_endpoint_streams_events(self, app) -> None:
        """Endpoint should stream events in SSE format."""
        mgr = get_loading_progress()

        # Pre-emit events. The late-joiner will see the latest (READY),
        # then the None sentinel will close the stream.
        mgr.emit("test-model", LoadingEventType.READY, progress=100, message="Done")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/admin/models/test-model/loading-progress")
            body = response.text

            # Should contain SSE formatted event
            assert "event: ready" in body
            assert "data: " in body
            # Verify data is valid JSON
            for line in body.strip().split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    assert data["model_id"] == "test-model"
                    assert data["event"] == "ready"

    @pytest.mark.asyncio
    async def test_endpoint_with_background_emitter(self, app) -> None:
        """Test streaming with events emitted asynchronously."""
        mgr = get_loading_progress()

        async def emit_events():
            await asyncio.sleep(0.05)
            mgr.emit("bg-model", LoadingEventType.WEIGHTS_LOADING, progress=0)
            await asyncio.sleep(0.05)
            mgr.emit("bg-model", LoadingEventType.WEIGHTS_LOADING, progress=100)
            await asyncio.sleep(0.05)
            mgr.emit("bg-model", LoadingEventType.READY, progress=100, message="Done")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Start emitter in background
            task = asyncio.create_task(emit_events())

            response = await client.get("/v1/admin/models/bg-model/loading-progress")
            body = response.text

            await task

            # Should have received multiple events
            assert "event: weights_loading" in body
            assert "event: ready" in body

    @pytest.mark.asyncio
    async def test_endpoint_error_event_closes_stream(self, app) -> None:
        """Error events should close the stream gracefully."""
        mgr = get_loading_progress()
        mgr.emit("err-model", LoadingEventType.ERROR, message="Load failed")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/admin/models/err-model/loading-progress")
            body = response.text

            assert "event: error" in body
            for line in body.strip().split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    assert data["message"] == "Load failed"

    @pytest.mark.asyncio
    async def test_endpoint_model_id_with_slash(self, app) -> None:
        """Model IDs with slashes (e.g., mlx-community/Llama-3B) should work."""
        mgr = get_loading_progress()
        model_id = "mlx-community/Llama-3B-4bit"
        mgr.emit(model_id, LoadingEventType.READY, progress=100)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/v1/admin/models/{model_id}/loading-progress")
            body = response.text
            assert "event: ready" in body


# ============================================================================
# Pool Integration Tests (mock-based)
# ============================================================================


class TestPoolProgressIntegration:
    """Tests that model pool loading emits progress events."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> None:
        reset_loading_progress()
        yield  # type: ignore[misc]
        reset_loading_progress()

    @pytest.mark.asyncio
    async def test_load_model_emits_progress_events(self) -> None:
        """_load_model should emit WEIGHTS_LOADING, ADAPTER_INIT, and READY events."""
        from mlx_manager.mlx_server.models.pool import ModelPoolManager

        pool = ModelPoolManager(max_memory_gb=100.0, max_models=10)
        mgr = get_loading_progress()
        queue = mgr.subscribe("test/model")

        # Mock all the heavy dependencies. get_memory_usage and get_session
        # are imported locally inside _load_model, so patch at their source.
        with (
            patch("mlx_manager.mlx_server.models.pool.detect_model_type") as mock_detect,
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 2.0},
            ),
            patch("mlx_lm.load") as mock_load,
            patch(
                "mlx_manager.mlx_server.models.adapters.composable.create_adapter"
            ) as mock_create_adapter,
            patch(
                "mlx_manager.mlx_server.models.adapters.registry.detect_model_family",
                return_value="default",
            ),
            patch(
                "mlx_manager.database.get_session",
                side_effect=Exception("no db"),
            ),
        ):
            from mlx_manager.mlx_server.models.types import ModelType

            mock_detect.return_value = ModelType.TEXT_GEN
            mock_load.return_value = (AsyncMock(), AsyncMock())
            mock_adapter = AsyncMock()
            mock_adapter.tool_parser.parser_id = "none"
            mock_adapter.thinking_parser.parser_id = "none"
            mock_adapter.post_load_configure = AsyncMock()
            mock_create_adapter.return_value = mock_adapter

            await pool._load_model("test/model")

        # Collect all events
        events: list[LoadingEvent] = []
        while not queue.empty():
            evt = queue.get_nowait()
            if evt is not None:
                events.append(evt)

        event_types = [e.event for e in events]
        assert LoadingEventType.WEIGHTS_LOADING in event_types
        assert LoadingEventType.ADAPTER_INIT in event_types
        assert LoadingEventType.READY in event_types

        # First WEIGHTS_LOADING should have progress=0
        weights_events = [e for e in events if e.event == LoadingEventType.WEIGHTS_LOADING]
        assert weights_events[0].progress == 0
        assert weights_events[1].progress == 100

    @pytest.mark.asyncio
    async def test_load_model_emits_error_on_failure(self) -> None:
        """_load_model should emit ERROR event when loading fails."""
        from mlx_manager.mlx_server.models.pool import ModelPoolManager

        pool = ModelPoolManager(max_memory_gb=100.0, max_models=10)
        mgr = get_loading_progress()
        queue = mgr.subscribe("bad/model")

        with (
            patch("mlx_manager.mlx_server.models.pool.detect_model_type") as mock_detect,
            patch("mlx_lm.load", side_effect=RuntimeError("model not found")),
            patch(
                "mlx_manager.database.get_session",
                side_effect=Exception("no db"),
            ),
        ):
            from mlx_manager.mlx_server.models.types import ModelType

            mock_detect.return_value = ModelType.TEXT_GEN

            with pytest.raises(RuntimeError, match="Failed to load model"):
                await pool._load_model("bad/model")

        # Collect events
        events: list[LoadingEvent] = []
        while not queue.empty():
            evt = queue.get_nowait()
            if evt is not None:
                events.append(evt)

        event_types = [e.event for e in events]
        assert LoadingEventType.WEIGHTS_LOADING in event_types  # progress=0 emitted before load
        assert LoadingEventType.ERROR in event_types

        error_event = next(e for e in events if e.event == LoadingEventType.ERROR)
        assert "model not found" in (error_event.message or "")
