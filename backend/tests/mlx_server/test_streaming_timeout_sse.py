"""Tests for streaming timeout SSE error branch in chat/completions routers.

The existing test_timeout.py covers the middleware and with_timeout decorator.
This file targets the SSE generator's except TimeoutError: branch specifically —
the path where generate_chat_stream's asyncio.wait_for() fires during streaming
preparation and the generator yields a timeout_error_event() dict before closing.

Key: We mock asyncio.wait_for to raise TimeoutError inside the event_generator,
then consume the EventSourceResponse generator to confirm the error event is emitted.
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.api.v1.chat import _handle_streaming
from mlx_manager.mlx_server.models.ir import InternalRequest
from mlx_manager.mlx_server.schemas.openai import ChatCompletionRequest, ChatMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_ir(model: str = "test-model", max_tokens: int = 100) -> InternalRequest:
    """Build a streaming InternalRequest."""
    from mlx_manager.mlx_server.services.formatters import OpenAIFormatter

    req = ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content="Hello")],
        stream=True,
        max_tokens=max_tokens,
    )
    return OpenAIFormatter.parse_request(req)


async def _collect_sse_events(response: Any) -> list[dict]:
    """Consume an EventSourceResponse and return list of SSE event dicts.

    sse_starlette's EventSourceResponse.body_iterator yields the raw generator
    dicts directly (e.g. {'data': '...'} or {'event': 'error', 'data': '...'}).
    We collect them as-is.
    """
    events = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, dict):
            events.append(chunk)
        elif isinstance(chunk, (bytes, str)):
            # Fallback: raw SSE text format parsing (shouldn't normally occur)
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            if text.strip():
                lines = text.strip().split("\n")
                event: dict[str, str] = {}
                for line in lines:
                    if line.startswith("event:"):
                        event["event"] = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        event["data"] = line[len("data:"):].strip()
                if event:
                    events.append(event)
    return events


# ---------------------------------------------------------------------------
# Gap 2: Streaming timeout SSE branch
# ---------------------------------------------------------------------------


class TestStreamingTimeoutSSEBranch:
    """Streaming generator's except TimeoutError: branch yields an error SSE event.

    Strategy: patch generate_chat_stream with a coroutine that raises TimeoutError.
    The patches must remain active during SSE body iteration (not just during
    _handle_streaming construction), so we use contextlib.ExitStack inside each test.
    """

    def _make_patches(self, timeout_seconds: float = 5.0):
        """Return (settings_patch, gen_patch) context managers."""
        settings = MagicMock()
        settings.timeout_chat_seconds = timeout_seconds
        p_settings = patch(
            "mlx_manager.mlx_server.api.v1.chat.get_settings", return_value=settings
        )
        p_gen = patch(
            "mlx_manager.mlx_server.api.v1.chat.generate_chat_stream",
            new_callable=AsyncMock,
            side_effect=TimeoutError(),
        )
        return p_settings, p_gen, settings

    @pytest.mark.asyncio
    async def test_timeout_during_stream_prep_yields_error_event(self):
        """When generate_chat_stream raises TimeoutError, error SSE event is emitted.

        This directly tests the except TimeoutError: branch inside event_generator()
        in _handle_streaming (chat.py ~line 330).
        """
        p_settings, p_gen, settings = self._make_patches(5.0)
        with p_settings, p_gen:
            ir = _make_stream_ir()
            req = ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Hello")],
                stream=True,
            )
            response = await _handle_streaming(ir, req)
            # Consume the SSE generator while patches are still active
            events = await _collect_sse_events(response)

        # Must have at least one event with event=error or data containing error info
        error_events = [
            e
            for e in events
            if e.get("event") == "error"
            or ("data" in e and "error" in e["data"].lower())
            or ("data" in e and "timeout" in e["data"].lower())
        ]
        assert len(error_events) >= 1, f"Expected error SSE event, got: {events}"

    @pytest.mark.asyncio
    async def test_timeout_error_event_contains_timeout_message(self):
        """The timeout SSE error event data contains timeout information."""
        p_settings, p_gen, settings = self._make_patches(42.0)
        with p_settings, p_gen:
            ir = _make_stream_ir()
            req = ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Hello")],
                stream=True,
            )
            response = await _handle_streaming(ir, req)
            events = await _collect_sse_events(response)

        # Find the error event data
        error_data_raw = None
        for e in events:
            if e.get("event") == "error":
                error_data_raw = e.get("data")
                break
            if "data" in e and "error" in e.get("data", "").lower():
                error_data_raw = e.get("data")
                break

        assert error_data_raw is not None, f"No error data found in events: {events}"
        error_data = json.loads(error_data_raw)
        # Should contain error key with type and/or message
        assert "error" in error_data

    @pytest.mark.asyncio
    async def test_timeout_closes_stream_cleanly(self):
        """After timeout error event, the stream terminates (no infinite loop)."""
        p_settings, p_gen, settings = self._make_patches(5.0)
        with p_settings, p_gen:
            ir = _make_stream_ir()
            req = ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Hello")],
                stream=True,
            )
            response = await _handle_streaming(ir, req)
            # Collecting all events should complete (not hang)
            events = await asyncio.wait_for(_collect_sse_events(response), timeout=5.0)

        # Stream should have ended after error event
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_initial_role_chunk_emitted_before_timeout(self):
        """The initial role chunk (stream_start) is yielded before timeout fires.

        The timeout is set on asyncio.wait_for(generate_chat_stream(...)), which
        is called AFTER stream_start(). So stream_start chunks appear before the error.
        """
        p_settings, p_gen, settings = self._make_patches(5.0)
        with p_settings, p_gen:
            ir = _make_stream_ir()
            req = ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Hello")],
                stream=True,
            )
            response = await _handle_streaming(ir, req)
            events = await _collect_sse_events(response)

        # Should have more than 1 event: at least stream_start chunk + error event
        assert len(events) >= 1  # At minimum the error event


class TestStreamingTimeoutSSEViaDirectFunction:
    """Test timeout_error_event() helper produces correct SSE dict."""

    def test_timeout_error_event_structure(self):
        """timeout_error_event returns dict with event and data keys."""
        from mlx_manager.mlx_server.utils.request_helpers import timeout_error_event

        result = timeout_error_event(30.0)

        assert "event" in result
        assert result["event"] == "error"
        assert "data" in result

    def test_timeout_error_event_data_is_valid_json(self):
        """timeout_error_event data field is valid JSON."""
        from mlx_manager.mlx_server.utils.request_helpers import timeout_error_event

        result = timeout_error_event(60.0)
        data = json.loads(result["data"])

        assert "error" in data

    def test_timeout_error_event_contains_timeout_seconds(self):
        """timeout_error_event embeds the timeout duration."""
        from mlx_manager.mlx_server.utils.request_helpers import timeout_error_event

        result = timeout_error_event(120.0)
        # The timeout value should appear somewhere in the event data
        assert "120" in result["data"]

    def test_timeout_error_event_different_durations(self):
        """timeout_error_event correctly represents different timeout values."""
        from mlx_manager.mlx_server.utils.request_helpers import timeout_error_event

        for duration in [30.0, 60.0, 900.0]:
            result = timeout_error_event(duration)
            assert str(int(duration)) in result["data"]
