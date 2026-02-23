"""Additional tests for AnthropicFormatter covering uncovered lines.

Targets:
- parse_request: dict-based content blocks (lines 187-227)
- stream_end: tool_use content blocks (lines 322-351)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from mlx_manager.mlx_server.services.formatters.anthropic import AnthropicFormatter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(
    model: str = "test-model",
    messages: list | None = None,
    system: str | None = None,
    tools: list | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool = False,
    stop_sequences: list | None = None,
):
    """Create a mock AnthropicMessagesRequest with specified parameters."""
    req = MagicMock()
    req.model = model
    req.messages = messages or []
    req.system = system
    req.tools = tools
    req.max_tokens = max_tokens
    req.temperature = temperature
    req.top_p = top_p
    req.stream = stream
    req.stop_sequences = stop_sequences
    return req


def make_msg(role: str, content):
    """Create a mock message."""
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg


def make_dict_block(block_type: str, **kwargs) -> dict:
    """Create a content block as a plain dict."""
    return {"type": block_type, **kwargs}


# ---------------------------------------------------------------------------
# Tests for parse_request dict-based content blocks (lines 187-227)
# ---------------------------------------------------------------------------


class TestParseRequestDictBlocks:
    """Test parse_request handling of content blocks as plain dicts."""

    def test_dict_text_block(self):
        """Dict text block extracts text into message content."""
        msg = make_msg(
            "user",
            [make_dict_block("text", text="What is the weather?")],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        user_msgs = [m for m in result.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "What is the weather?"

    def test_dict_image_block(self):
        """Dict image block creates data URL and adds to images list."""
        msg = make_msg(
            "user",
            [
                make_dict_block("text", text="Describe this"),
                make_dict_block(
                    "image",
                    source={"media_type": "image/jpeg", "data": "abc123"},
                ),
            ],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        assert result.images is not None
        assert len(result.images) == 1
        assert result.images[0] == "data:image/jpeg;base64,abc123"

    def test_dict_image_block_default_media_type(self):
        """Dict image block defaults to image/png when no media_type specified."""
        msg = make_msg(
            "user",
            [
                make_dict_block(
                    "image",
                    source={"data": "xyz789"},  # No media_type
                ),
            ],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        assert result.images is not None
        assert "image/png" in result.images[0]

    def test_dict_tool_use_block(self):
        """Dict tool_use block becomes an assistant message with tool_calls."""
        msg = make_msg(
            "assistant",
            [
                make_dict_block(
                    "tool_use",
                    id="toolu_01",
                    name="get_weather",
                    input={"location": "Tokyo"},
                ),
            ],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        tool_msgs = [m for m in result.messages if "tool_calls" in m]
        assert len(tool_msgs) == 1
        tc = tool_msgs[0]["tool_calls"][0]
        assert tc["id"] == "toolu_01"
        assert tc["function"]["name"] == "get_weather"
        parsed_args = json.loads(tc["function"]["arguments"])
        assert parsed_args["location"] == "Tokyo"

    def test_dict_tool_use_block_no_input(self):
        """Dict tool_use block with no input defaults to empty dict."""
        msg = make_msg(
            "assistant",
            [
                make_dict_block(
                    "tool_use",
                    id="toolu_02",
                    name="get_time",
                    # No 'input' key
                ),
            ],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        tool_msgs = [m for m in result.messages if "tool_calls" in m]
        assert len(tool_msgs) == 1
        parsed_args = json.loads(tool_msgs[0]["tool_calls"][0]["function"]["arguments"])
        assert parsed_args == {}

    def test_dict_tool_result_block_string_content(self):
        """Dict tool_result block with string content becomes tool message."""
        msg = make_msg(
            "user",
            [
                make_dict_block(
                    "tool_result",
                    tool_use_id="toolu_01",
                    content="Tokyo: 25°C, sunny",
                ),
            ],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "toolu_01"
        assert tool_msgs[0]["content"] == "Tokyo: 25°C, sunny"

    def test_dict_tool_result_block_list_content(self):
        """Dict tool_result block with list content joins text items."""
        msg = make_msg(
            "user",
            [
                make_dict_block(
                    "tool_result",
                    tool_use_id="toolu_02",
                    content=[
                        {"type": "text", "text": "Weather: sunny"},
                        {"type": "text", "text": "Temperature: 25°C"},
                        {"type": "image", "url": "data:..."},  # Non-text, should be skipped
                    ],
                ),
            ],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "Weather: sunny" in tool_msgs[0]["content"]
        assert "Temperature: 25°C" in tool_msgs[0]["content"]

    def test_dict_mixed_text_blocks(self):
        """Multiple dict text blocks are joined into single message content."""
        msg = make_msg(
            "user",
            [
                make_dict_block("text", text="First part"),
                make_dict_block("text", text="second part"),
            ],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        user_msgs = [m for m in result.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "First part" in user_msgs[0]["content"]
        assert "second part" in user_msgs[0]["content"]

    def test_dict_tool_use_block_skips_text_collection(self):
        """Dict tool_use block uses continue, not adding to text_parts."""
        msg = make_msg(
            "assistant",
            [
                make_dict_block("text", text="I'll check the weather."),
                make_dict_block(
                    "tool_use",
                    id="toolu_03",
                    name="get_weather",
                    input={"location": "Paris"},
                ),
            ],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        # Text block should be collected in a separate message
        text_msgs = [
            m
            for m in result.messages
            if m.get("role") == "assistant" and "content" in m and not m.get("tool_calls")
        ]
        tool_msgs = [
            m for m in result.messages if m.get("role") == "assistant" and "tool_calls" in m
        ]

        # Tool use is in its own message
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_dict_tool_result_block_empty_content_default(self):
        """Dict tool_result block with no content defaults to empty string."""
        msg = make_msg(
            "user",
            [
                make_dict_block(
                    "tool_result",
                    tool_use_id="toolu_04",
                    # No 'content' key
                ),
            ],
        )
        req = make_request(messages=[msg])
        result = AnthropicFormatter.parse_request(req)

        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == ""


# ---------------------------------------------------------------------------
# Tests for stream_end with tool_calls (lines 322-351)
# ---------------------------------------------------------------------------


class TestAnthropicFormatterStreamEndWithTools:
    """Test stream_end emits proper tool_use content blocks."""

    def test_stream_end_with_single_tool_call(self):
        """stream_end with one tool call emits content_block_start, input_json_delta, and stop."""
        fmt = AnthropicFormatter("model", "msg_1")
        tool_calls = [
            {
                "id": "toolu_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Tokyo"}',
                },
            }
        ]
        events = fmt.stream_end("tool_calls", tool_calls=tool_calls)

        # Events: content_block_stop (text), content_block_start (tool, empty input),
        #         content_block_delta (input_json_delta), content_block_stop (tool),
        #         message_delta, message_stop
        assert len(events) == 6

        event_names = [e["event"] for e in events]
        assert event_names[0] == "content_block_stop"  # Close text block
        assert event_names[1] == "content_block_start"  # Start tool_use block
        assert event_names[2] == "content_block_delta"  # input_json_delta
        assert event_names[3] == "content_block_stop"  # Stop tool_use block
        assert event_names[4] == "message_delta"
        assert event_names[5] == "message_stop"

        # Verify the tool_use content_block_start has empty input
        tool_block_start = json.loads(events[1]["data"])
        assert tool_block_start["type"] == "content_block_start"
        assert tool_block_start["index"] == 1  # Text is index 0
        cb = tool_block_start["content_block"]
        assert cb["type"] == "tool_use"
        assert cb["id"] == "toolu_abc"
        assert cb["name"] == "get_weather"
        assert cb["input"] == {}  # Empty per Anthropic spec; input is sent via input_json_delta

        # Verify the input_json_delta carries the actual tool arguments
        input_delta_event = json.loads(events[2]["data"])
        assert input_delta_event["type"] == "content_block_delta"
        assert input_delta_event["index"] == 1
        delta = input_delta_event["delta"]
        assert delta["type"] == "input_json_delta"
        assert json.loads(delta["partial_json"]) == {"location": "Tokyo"}

        # Verify the tool_use content_block_stop
        tool_block_stop = json.loads(events[3]["data"])
        assert tool_block_stop["type"] == "content_block_stop"
        assert tool_block_stop["index"] == 1

    def test_stream_end_with_multiple_tool_calls(self):
        """stream_end with multiple tool calls emits start/delta/stop for each."""
        fmt = AnthropicFormatter("model", "msg_2")
        tool_calls = [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
            },
            {
                "id": "toolu_2",
                "type": "function",
                "function": {"name": "get_time", "arguments": '{"timezone": "JST"}'},
            },
        ]
        events = fmt.stream_end("tool_calls", tool_calls=tool_calls)

        # content_block_stop(text)
        # + 2*(content_block_start + content_block_delta + content_block_stop)
        # + message_delta + message_stop
        assert len(events) == 9

        # First tool: content_block_start at events[1], delta at events[2], stop at events[3]
        first_start = json.loads(events[1]["data"])
        assert first_start["index"] == 1
        assert first_start["content_block"]["name"] == "get_weather"
        assert first_start["content_block"]["input"] == {}

        first_delta = json.loads(events[2]["data"])
        assert first_delta["index"] == 1
        assert first_delta["delta"]["type"] == "input_json_delta"
        assert json.loads(first_delta["delta"]["partial_json"]) == {"location": "Tokyo"}

        # Second tool: content_block_start at events[4], delta at events[5], stop at events[6]
        second_start = json.loads(events[4]["data"])
        assert second_start["index"] == 2
        assert second_start["content_block"]["name"] == "get_time"
        assert second_start["content_block"]["input"] == {}

        second_delta = json.loads(events[5]["data"])
        assert second_delta["index"] == 2
        assert second_delta["delta"]["type"] == "input_json_delta"
        assert json.loads(second_delta["delta"]["partial_json"]) == {"timezone": "JST"}

    def test_stream_end_with_tool_call_invalid_json_arguments(self):
        """stream_end handles invalid JSON in tool call arguments gracefully (lines 329-330)."""
        fmt = AnthropicFormatter("model", "msg_3")
        tool_calls = [
            {
                "id": "toolu_bad",
                "type": "function",
                "function": {
                    "name": "broken_tool",
                    "arguments": "not valid json {{{",
                },
            }
        ]
        events = fmt.stream_end("tool_calls", tool_calls=tool_calls)

        # Should not raise; produces start + delta + stop for the tool block
        assert len(events) == 6

        tool_start = json.loads(events[1]["data"])
        cb = tool_start["content_block"]
        assert cb["type"] == "tool_use"
        assert cb["input"] == {}  # Always empty in content_block_start

        # The input_json_delta should carry the serialised empty-dict fallback
        input_delta = json.loads(events[2]["data"])
        assert input_delta["delta"]["type"] == "input_json_delta"
        assert json.loads(input_delta["delta"]["partial_json"]) == {}  # Invalid JSON → empty dict

    def test_stream_end_with_tool_call_missing_function_key(self):
        """stream_end handles tool call with missing function key gracefully."""
        fmt = AnthropicFormatter("model", "msg_4")
        tool_calls = [
            {
                "id": "toolu_missing",
                "type": "function",
                # No "function" key
            }
        ]
        events = fmt.stream_end("tool_calls", tool_calls=tool_calls)

        assert len(events) == 6
        tool_start = json.loads(events[1]["data"])
        cb = tool_start["content_block"]
        assert cb["name"] == ""  # No function name
        assert cb["input"] == {}  # content_block_start always has empty input

        # input_json_delta carries the empty-dict fallback
        input_delta = json.loads(events[2]["data"])
        assert input_delta["delta"]["type"] == "input_json_delta"
        assert json.loads(input_delta["delta"]["partial_json"]) == {}

    def test_stream_end_with_tool_call_no_id_generates_fallback(self):
        """stream_end uses fallback id 'toolu_{i}' when id is missing."""
        fmt = AnthropicFormatter("model", "msg_5")
        tool_calls = [
            {
                # No "id" key
                "type": "function",
                "function": {"name": "fallback_tool", "arguments": "{}"},
            }
        ]
        events = fmt.stream_end("tool_calls", tool_calls=tool_calls)

        assert len(events) == 6
        tool_start = json.loads(events[1]["data"])
        cb = tool_start["content_block"]
        assert cb["id"] == "toolu_0"  # Fallback ID

    def test_stream_end_no_tool_calls_emits_three_events(self):
        """stream_end without tool_calls emits exactly 3 events (existing behavior preserved)."""
        fmt = AnthropicFormatter("model", "msg_6")
        events = fmt.stream_end("stop", tool_calls=None)

        assert len(events) == 3
        assert events[0]["event"] == "content_block_stop"
        assert events[1]["event"] == "message_delta"
        assert events[2]["event"] == "message_stop"

    def test_stream_end_tool_calls_stop_reason_is_tool_use(self):
        """stream_end with tool_calls has 'tool_use' stop reason in message_delta."""
        fmt = AnthropicFormatter("model", "msg_7")
        tool_calls = [
            {
                "id": "toolu_x",
                "function": {"name": "test", "arguments": "{}"},
            }
        ]
        events = fmt.stream_end("tool_calls", tool_calls=tool_calls)

        msg_delta = json.loads(events[-2]["data"])
        assert msg_delta["delta"]["stop_reason"] == "tool_use"

    def test_stream_end_tool_calls_output_tokens_in_message_delta(self):
        """stream_end with tool_calls passes output_tokens to message_delta."""
        fmt = AnthropicFormatter("model", "msg_8")
        tool_calls = [
            {
                "id": "toolu_y",
                "function": {"name": "test", "arguments": "{}"},
            }
        ]
        events = fmt.stream_end("tool_calls", tool_calls=tool_calls, output_tokens=42)

        msg_delta = json.loads(events[-2]["data"])
        assert msg_delta["usage"]["output_tokens"] == 42

    def test_full_streaming_with_tool_call(self):
        """Full streaming lifecycle with tool call produces correct event sequence."""
        from mlx_manager.mlx_server.models.ir import StreamEvent

        fmt = AnthropicFormatter("model", "msg_lifecycle_tools")

        all_events: list[dict] = []
        all_events.extend(fmt.stream_start())
        all_events.extend(fmt.stream_event(StreamEvent(type="content", content="Let me check")))
        all_events.extend(
            fmt.stream_end(
                "tool_calls",
                tool_calls=[
                    {
                        "id": "toolu_z",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                    }
                ],
                output_tokens=15,
            )
        )

        event_types = [e["event"] for e in all_events]
        assert event_types == [
            "message_start",
            "content_block_start",  # text block
            "content_block_delta",  # "Let me check"
            "content_block_stop",  # close text block
            "content_block_start",  # tool_use block (empty input)
            "content_block_delta",  # input_json_delta with actual arguments
            "content_block_stop",  # close tool_use block
            "message_delta",
            "message_stop",
        ]

        # Verify tool_use block content_block_start has empty input
        tool_start_data = json.loads(all_events[4]["data"])
        assert tool_start_data["content_block"]["name"] == "get_weather"
        assert tool_start_data["content_block"]["input"] == {}

        # Verify input_json_delta carries the actual arguments
        input_delta_data = json.loads(all_events[5]["data"])
        assert input_delta_data["delta"]["type"] == "input_json_delta"
        assert json.loads(input_delta_data["delta"]["partial_json"]) == {"location": "NYC"}
