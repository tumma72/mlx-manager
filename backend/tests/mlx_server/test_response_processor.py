"""Tests for response processor minimal module.

NOTE: Most functionality moved to composable adapters (test_composable_adapters.py).
This module only tests StreamEvent, ParseResult, and StreamingProcessor.
"""

import pytest

from mlx_manager.mlx_server.schemas.openai import FunctionCall, ToolCall
from mlx_manager.mlx_server.services.response_processor import (
    ParseResult,
    StreamEvent,
    StreamingProcessor,
)


class TestPydanticModels:
    """Tests for Pydantic model behavior."""

    def test_parse_result_with_content_only(self) -> None:
        """ParseResult can be created with just content."""
        result = ParseResult(content="Hello, world!")
        assert result.content == "Hello, world!"
        assert result.tool_calls == []
        assert result.reasoning is None

    def test_parse_result_with_tool_calls(self) -> None:
        """ParseResult with tool calls."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="get_weather", arguments='{"city": "SF"}'),
        )
        result = ParseResult(content="Let me check", tool_calls=[tool_call])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"

    def test_parse_result_with_reasoning(self) -> None:
        """ParseResult with reasoning content."""
        result = ParseResult(
            content="The answer is 42",
            reasoning="First I thought about it deeply",
        )
        assert result.reasoning == "First I thought about it deeply"


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


class TestStreamingProcessor:
    """Tests for StreamingProcessor (requires adapter).

    NOTE: Comprehensive tests in test_inference.py E2E tests.
    These are basic unit tests for the streaming API.
    """

    def test_streaming_processor_requires_adapter(self) -> None:
        """StreamingProcessor requires an adapter."""
        with pytest.raises(TypeError):
            StreamingProcessor()  # Missing required adapter argument
