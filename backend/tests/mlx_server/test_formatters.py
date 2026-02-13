"""Tests for ProtocolFormatter ABC, OpenAIFormatter, and package structure."""

from __future__ import annotations

import json
from typing import Any

import pytest

from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult
from mlx_manager.mlx_server.services.formatters import (
    AnthropicFormatter,
    OpenAIFormatter,
    ProtocolFormatter,
)
from mlx_manager.mlx_server.services.formatters.anthropic import openai_stop_to_anthropic
from mlx_manager.mlx_server.services.formatters.base import (
    ProtocolFormatter as BaseProtocolFormatter,
)


class ConcreteFormatter(ProtocolFormatter):
    """Minimal concrete implementation for testing ABC contract."""

    def stream_start(self) -> list[dict[str, Any]]:
        return [{"event": "start", "data": self.model_id}]

    def stream_event(self, event: StreamEvent) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        if event.content:
            chunks.append({"data": event.content})
        if event.reasoning_content:
            chunks.append({"data": f"reasoning:{event.reasoning_content}"})
        return chunks

    def stream_end(
        self,
        finish_reason: str,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        output_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        return [{"data": f"end:{finish_reason}"}]

    def format_complete(
        self,
        result: TextResult,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> Any:
        return {
            "content": result.content,
            "finish_reason": result.finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


class TestProtocolFormatterABC:
    """Test the ABC contract and instantiation."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ProtocolFormatter("model", "req-1")  # type: ignore[abstract]

    def test_concrete_instantiation(self) -> None:
        fmt = ConcreteFormatter("test-model", "req-123")
        assert fmt.model_id == "test-model"
        assert fmt.request_id == "req-123"
        assert isinstance(fmt.created, int)
        assert fmt.created > 0

    def test_re_export_matches_base(self) -> None:
        assert ProtocolFormatter is BaseProtocolFormatter


class TestStreamStart:
    def test_returns_list_of_dicts(self) -> None:
        fmt = ConcreteFormatter("model", "req-1")
        result = fmt.stream_start()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["event"] == "start"
        assert result[0]["data"] == "model"


class TestStreamEvent:
    def test_content_event(self) -> None:
        fmt = ConcreteFormatter("model", "req-1")
        event = StreamEvent(type="content", content="hello")
        result = fmt.stream_event(event)
        assert len(result) == 1
        assert result[0]["data"] == "hello"

    def test_reasoning_event(self) -> None:
        fmt = ConcreteFormatter("model", "req-1")
        event = StreamEvent(type="reasoning_content", reasoning_content="thinking...")
        result = fmt.stream_event(event)
        assert len(result) == 1
        assert result[0]["data"] == "reasoning:thinking..."

    def test_empty_event(self) -> None:
        fmt = ConcreteFormatter("model", "req-1")
        event = StreamEvent()
        result = fmt.stream_event(event)
        assert result == []


class TestStreamEnd:
    def test_returns_end_events(self) -> None:
        fmt = ConcreteFormatter("model", "req-1")
        result = fmt.stream_end("stop")
        assert len(result) == 1
        assert result[0]["data"] == "end:stop"

    def test_tool_calls_finish_reason(self) -> None:
        fmt = ConcreteFormatter("model", "req-1")
        result = fmt.stream_end(
            "tool_calls",
            tool_calls=[{"id": "tc_1", "type": "function", "function": {"name": "f"}}],
        )
        assert result[0]["data"] == "end:tool_calls"


class TestFormatComplete:
    def test_basic_response(self) -> None:
        fmt = ConcreteFormatter("model", "req-1")
        result = TextResult(content="Hello world", finish_reason="stop")
        response = fmt.format_complete(result, prompt_tokens=10, completion_tokens=5)
        assert response["content"] == "Hello world"
        assert response["finish_reason"] == "stop"
        assert response["prompt_tokens"] == 10
        assert response["completion_tokens"] == 5

    def test_tool_calls_response(self) -> None:
        fmt = ConcreteFormatter("model", "req-1")
        result = TextResult(
            content="",
            tool_calls=[{"id": "tc_1", "type": "function", "function": {"name": "get_weather"}}],
            finish_reason="tool_calls",
        )
        response = fmt.format_complete(result, prompt_tokens=20, completion_tokens=15)
        assert response["finish_reason"] == "tool_calls"

    def test_reasoning_response(self) -> None:
        fmt = ConcreteFormatter("model", "req-1")
        result = TextResult(
            content="The answer is 42.",
            reasoning_content="Let me think about this...",
            finish_reason="stop",
        )
        response = fmt.format_complete(result, prompt_tokens=5, completion_tokens=10)
        assert response["content"] == "The answer is 42."


# ── OpenAIFormatter tests ─────────────────────────────────────────


def _parse_sse(sse_dict: dict[str, Any]) -> dict[str, Any] | str:
    """Parse an SSE-ready dict, returning the parsed JSON or raw string."""
    data = sse_dict["data"]
    if data == "[DONE]":
        return data
    return json.loads(data)


class TestOpenAIFormatterStreamStart:
    def test_emits_role_chunk(self) -> None:
        fmt = OpenAIFormatter("test-model", "chatcmpl-abc")
        events = fmt.stream_start()
        assert len(events) == 1
        chunk = _parse_sse(events[0])
        assert isinstance(chunk, dict)
        assert chunk["id"] == "chatcmpl-abc"
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "test-model"
        choice = chunk["choices"][0]
        assert choice["delta"] == {"role": "assistant", "content": ""}
        assert choice["finish_reason"] is None


class TestOpenAIFormatterStreamEvent:
    def test_content_token(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        event = StreamEvent(type="content", content="Hello")
        events = fmt.stream_event(event)
        assert len(events) == 1
        chunk = _parse_sse(events[0])
        assert isinstance(chunk, dict)
        assert chunk["choices"][0]["delta"] == {"content": "Hello"}
        assert chunk["choices"][0]["finish_reason"] is None

    def test_reasoning_token(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        event = StreamEvent(type="reasoning_content", reasoning_content="thinking...")
        events = fmt.stream_event(event)
        assert len(events) == 1
        chunk = _parse_sse(events[0])
        assert isinstance(chunk, dict)
        assert chunk["choices"][0]["delta"] == {"reasoning_content": "thinking..."}

    def test_mixed_content_and_reasoning(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        event = StreamEvent(type="content", content="answer", reasoning_content="thought")
        events = fmt.stream_event(event)
        assert len(events) == 1
        chunk = _parse_sse(events[0])
        assert isinstance(chunk, dict)
        delta = chunk["choices"][0]["delta"]
        assert delta["content"] == "answer"
        assert delta["reasoning_content"] == "thought"

    def test_empty_event_yields_nothing(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        event = StreamEvent()
        events = fmt.stream_event(event)
        assert events == []


class TestOpenAIFormatterStreamEnd:
    def test_stop_finish(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        events = fmt.stream_end("stop")
        assert len(events) == 2
        chunk = _parse_sse(events[0])
        assert isinstance(chunk, dict)
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert chunk["choices"][0]["delta"] == {}
        assert _parse_sse(events[1]) == "[DONE]"

    def test_length_finish(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        events = fmt.stream_end("length")
        chunk = _parse_sse(events[0])
        assert isinstance(chunk, dict)
        assert chunk["choices"][0]["finish_reason"] == "length"

    def test_tool_calls_in_final_chunk(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
            },
            {
                "id": "call_def",
                "type": "function",
                "function": {"name": "get_time", "arguments": "{}"},
            },
        ]
        events = fmt.stream_end("tool_calls", tool_calls=tool_calls)
        assert len(events) == 2
        chunk = _parse_sse(events[0])
        assert isinstance(chunk, dict)
        delta = chunk["choices"][0]["delta"]
        assert chunk["choices"][0]["finish_reason"] == "tool_calls"
        assert len(delta["tool_calls"]) == 2
        assert delta["tool_calls"][0]["index"] == 0
        assert delta["tool_calls"][0]["id"] == "call_abc"
        assert delta["tool_calls"][1]["index"] == 1
        assert delta["tool_calls"][1]["id"] == "call_def"

    def test_done_sentinel_always_last(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        events = fmt.stream_end("stop")
        last = events[-1]
        assert last["data"] == "[DONE]"


class TestOpenAIFormatterComplete:
    def test_basic_response_structure(self) -> None:
        fmt = OpenAIFormatter("test-model", "chatcmpl-xyz")
        result = TextResult(content="Hello world", finish_reason="stop")
        response = fmt.format_complete(result, prompt_tokens=10, completion_tokens=5)
        assert response["id"] == "chatcmpl-xyz"
        assert response["object"] == "chat.completion"
        assert response["model"] == "test-model"
        assert isinstance(response["created"], int)
        choice = response["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Hello world"
        assert choice["finish_reason"] == "stop"
        assert "tool_calls" not in choice["message"]
        assert "reasoning_content" not in choice["message"]

    def test_usage_tokens(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        result = TextResult(content="ok")
        response = fmt.format_complete(result, prompt_tokens=100, completion_tokens=50)
        usage = response["usage"]
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_tool_calls_included(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"test"}'},
            }
        ]
        result = TextResult(content="", tool_calls=tool_calls, finish_reason="tool_calls")
        response = fmt.format_complete(result, prompt_tokens=10, completion_tokens=20)
        msg = response["choices"][0]["message"]
        assert msg["tool_calls"] == tool_calls
        assert response["choices"][0]["finish_reason"] == "tool_calls"

    def test_reasoning_content_included(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        result = TextResult(
            content="42",
            reasoning_content="Let me think...",
            finish_reason="stop",
        )
        response = fmt.format_complete(result, prompt_tokens=5, completion_tokens=10)
        msg = response["choices"][0]["message"]
        assert msg["reasoning_content"] == "Let me think..."
        assert msg["content"] == "42"

    def test_empty_content_no_tool_calls(self) -> None:
        fmt = OpenAIFormatter("model", "id-1")
        result = TextResult(content="", finish_reason="length")
        response = fmt.format_complete(result)
        msg = response["choices"][0]["message"]
        assert msg["content"] == ""
        assert "tool_calls" not in msg
        assert response["choices"][0]["finish_reason"] == "length"

    def test_matches_inference_output_format(self) -> None:
        """Verify formatter output matches what inference.py currently produces."""
        fmt = OpenAIFormatter("test-model", "chatcmpl-123")
        result = TextResult(
            content="Hello",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            ],
            reasoning_content="thought",
            finish_reason="tool_calls",
        )
        response = fmt.format_complete(result, prompt_tokens=10, completion_tokens=5)

        # Match inference.py _generate_chat_complete output structure
        assert set(response.keys()) == {"id", "object", "created", "model", "choices", "usage"}
        choice = response["choices"][0]
        assert set(choice.keys()) == {"index", "message", "finish_reason"}
        msg = choice["message"]
        assert "role" in msg
        assert "content" in msg
        assert "tool_calls" in msg
        assert "reasoning_content" in msg


# ── AnthropicFormatter tests ──────────────────────────────────────


class TestOpenAIStopToAnthropic:
    """Test the stop reason translation (mirrors test_protocol.py coverage)."""

    def test_stop_to_end_turn(self) -> None:
        assert openai_stop_to_anthropic("stop") == "end_turn"

    def test_length_to_max_tokens(self) -> None:
        assert openai_stop_to_anthropic("length") == "max_tokens"

    def test_tool_calls_to_tool_use(self) -> None:
        assert openai_stop_to_anthropic("tool_calls") == "tool_use"

    def test_content_filter_to_end_turn(self) -> None:
        assert openai_stop_to_anthropic("content_filter") == "end_turn"

    def test_none_to_end_turn(self) -> None:
        assert openai_stop_to_anthropic(None) == "end_turn"

    def test_unknown_to_end_turn(self) -> None:
        assert openai_stop_to_anthropic("something_unknown") == "end_turn"


class TestAnthropicFormatterStreamStart:
    def test_emits_message_start_and_content_block_start(self) -> None:
        fmt = AnthropicFormatter("claude-model", "msg_abc123")
        events = fmt.stream_start()
        assert len(events) == 2

        # message_start
        assert events[0]["event"] == "message_start"
        data = json.loads(events[0]["data"])
        assert data["type"] == "message_start"
        msg = data["message"]
        assert msg["id"] == "msg_abc123"
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert msg["content"] == []
        assert msg["model"] == "claude-model"
        assert msg["stop_reason"] is None
        assert msg["stop_sequence"] is None
        assert msg["usage"] == {"input_tokens": 0, "output_tokens": 0}

        # content_block_start
        assert events[1]["event"] == "content_block_start"
        data = json.loads(events[1]["data"])
        assert data["type"] == "content_block_start"
        assert data["index"] == 0
        assert data["content_block"] == {"type": "text", "text": ""}


class TestAnthropicFormatterStreamEvent:
    def test_content_token(self) -> None:
        fmt = AnthropicFormatter("model", "msg_1")
        event = StreamEvent(type="content", content="Hello")
        events = fmt.stream_event(event)
        assert len(events) == 1
        assert events[0]["event"] == "content_block_delta"
        data = json.loads(events[0]["data"])
        assert data["type"] == "content_block_delta"
        assert data["index"] == 0
        assert data["delta"] == {"type": "text_delta", "text": "Hello"}

    def test_reasoning_only_yields_nothing(self) -> None:
        """Anthropic protocol doesn't support reasoning_content in streaming."""
        fmt = AnthropicFormatter("model", "msg_1")
        event = StreamEvent(type="reasoning_content", reasoning_content="thinking...")
        events = fmt.stream_event(event)
        assert events == []

    def test_empty_event_yields_nothing(self) -> None:
        fmt = AnthropicFormatter("model", "msg_1")
        event = StreamEvent()
        events = fmt.stream_event(event)
        assert events == []

    def test_empty_content_string_yields_nothing(self) -> None:
        fmt = AnthropicFormatter("model", "msg_1")
        event = StreamEvent(type="content", content="")
        events = fmt.stream_event(event)
        assert events == []


class TestAnthropicFormatterStreamEnd:
    def test_emits_three_closing_events(self) -> None:
        fmt = AnthropicFormatter("model", "msg_1")
        events = fmt.stream_end("stop", output_tokens=42)
        assert len(events) == 3

        # content_block_stop
        assert events[0]["event"] == "content_block_stop"
        data = json.loads(events[0]["data"])
        assert data["type"] == "content_block_stop"
        assert data["index"] == 0

        # message_delta
        assert events[1]["event"] == "message_delta"
        data = json.loads(events[1]["data"])
        assert data["type"] == "message_delta"
        assert data["delta"]["stop_reason"] == "end_turn"
        assert data["delta"]["stop_sequence"] is None
        assert data["usage"]["output_tokens"] == 42

        # message_stop
        assert events[2]["event"] == "message_stop"
        data = json.loads(events[2]["data"])
        assert data["type"] == "message_stop"

    def test_stop_reason_translation(self) -> None:
        fmt = AnthropicFormatter("model", "msg_1")
        events = fmt.stream_end("tool_calls")
        data = json.loads(events[1]["data"])
        assert data["delta"]["stop_reason"] == "tool_use"

    def test_length_stop_reason(self) -> None:
        fmt = AnthropicFormatter("model", "msg_1")
        events = fmt.stream_end("length")
        data = json.loads(events[1]["data"])
        assert data["delta"]["stop_reason"] == "max_tokens"


class TestAnthropicFormatterComplete:
    def test_basic_response(self) -> None:
        fmt = AnthropicFormatter("test-model", "msg_xyz")
        result = TextResult(content="Hello world", finish_reason="stop")
        response = fmt.format_complete(result, prompt_tokens=10, completion_tokens=5)

        # Returns Pydantic model
        from mlx_manager.mlx_server.schemas.anthropic import AnthropicMessagesResponse

        assert isinstance(response, AnthropicMessagesResponse)
        assert response.id == "msg_xyz"
        assert response.model == "test-model"
        assert response.role == "assistant"
        assert response.type == "message"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "Hello world"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_tool_calls_stop_reason(self) -> None:
        fmt = AnthropicFormatter("model", "msg_1")
        result = TextResult(content="", finish_reason="tool_calls")
        response = fmt.format_complete(result, prompt_tokens=5, completion_tokens=10)
        assert response.stop_reason == "tool_use"

    def test_length_stop_reason(self) -> None:
        fmt = AnthropicFormatter("model", "msg_1")
        result = TextResult(content="partial", finish_reason="length")
        response = fmt.format_complete(result, prompt_tokens=5, completion_tokens=10)
        assert response.stop_reason == "max_tokens"

    def test_matches_messages_router_output(self) -> None:
        """Verify formatter produces same shape as messages.py _handle_non_streaming."""
        fmt = AnthropicFormatter("test-model", "msg_test123")
        result = TextResult(content="Hello", finish_reason="stop")
        response = fmt.format_complete(result, prompt_tokens=10, completion_tokens=5)

        # Serialize to dict to verify JSON structure
        d = response.model_dump()
        assert d["id"] == "msg_test123"
        assert d["type"] == "message"
        assert d["role"] == "assistant"
        assert d["model"] == "test-model"
        assert len(d["content"]) == 1
        assert d["content"][0]["type"] == "text"
        assert d["content"][0]["text"] == "Hello"
        assert d["stop_reason"] == "end_turn"
        assert d["usage"]["input_tokens"] == 10
        assert d["usage"]["output_tokens"] == 5

    def test_full_streaming_event_sequence(self) -> None:
        """Verify the complete streaming lifecycle matches messages.py behavior."""
        fmt = AnthropicFormatter("model", "msg_lifecycle")

        # Collect all events
        all_events: list[dict[str, Any]] = []
        all_events.extend(fmt.stream_start())
        all_events.extend(fmt.stream_event(StreamEvent(type="content", content="Hello")))
        all_events.extend(fmt.stream_event(StreamEvent(type="content", content=" world")))
        all_events.extend(fmt.stream_end("stop", output_tokens=2))

        # Verify event sequence
        event_types = [e["event"] for e in all_events]
        assert event_types == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
