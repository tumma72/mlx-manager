"""Tests for response processor minimal module.

NOTE: Most functionality moved to composable adapters (test_composable_adapters.py).
This module only tests StreamEvent and StreamProcessor.
"""

import pytest

from mlx_manager.mlx_server.services.response_processor import (
    StreamEvent,
    StreamProcessor,
)


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_stream_event_empty(self) -> None:
        """Empty stream event."""
        event = StreamEvent()
        assert event.content is None
        assert event.reasoning_content is None
        assert event.is_complete is False

    def test_stream_event_with_content(self) -> None:
        """Stream event with regular content."""
        event = StreamEvent(content="Hello")
        assert event.content == "Hello"
        assert event.reasoning_content is None

    def test_stream_event_with_reasoning(self) -> None:
        """Stream event with reasoning content."""
        event = StreamEvent(reasoning_content="Thinking...")
        assert event.reasoning_content == "Thinking..."
        assert event.content is None

    def test_stream_event_complete(self) -> None:
        """Stream event marking completion."""
        event = StreamEvent(reasoning_content="Done", is_complete=True)
        assert event.is_complete is True


class TestStreamProcessor:
    """Tests for StreamProcessor (requires adapter).

    NOTE: Comprehensive tests in test_inference.py E2E tests.
    These are basic unit tests for the streaming API.
    """

    def test_streaming_processor_requires_adapter(self) -> None:
        """StreamProcessor requires an adapter."""
        with pytest.raises(TypeError):
            StreamProcessor()  # Missing required adapter argument
