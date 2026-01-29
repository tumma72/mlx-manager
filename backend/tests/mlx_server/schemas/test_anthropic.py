"""Tests for Anthropic Messages API schemas."""

import pytest
from pydantic import ValidationError

from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ImageBlockParam,
    ImageSource,
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    MessageStartMessage,
    MessageStopEvent,
    TextBlock,
    TextBlockParam,
    TextDelta,
    Usage,
    extract_anthropic_content,
)


class TestAnthropicMessagesRequest:
    """Tests for request validation."""

    def test_max_tokens_required(self) -> None:
        """max_tokens is required (unlike OpenAI)."""
        with pytest.raises(ValidationError) as exc_info:
            AnthropicMessagesRequest(
                model="claude-3-opus",
                messages=[MessageParam(role="user", content="Hello")],
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_tokens",) for e in errors)

    def test_valid_request_string_content(self) -> None:
        """Valid request with string content."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=1000,
            messages=[
                MessageParam(role="user", content="Hello"),
                MessageParam(role="assistant", content="Hi!"),
            ],
        )

        assert request.model == "claude-3-opus"
        assert request.max_tokens == 1000
        assert len(request.messages) == 2
        assert request.stream is False

    def test_valid_request_content_blocks(self) -> None:
        """Valid request with content blocks (text + image)."""
        image_source = ImageSource(
            type="base64",
            media_type="image/png",
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        )
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=1000,
            messages=[
                MessageParam(
                    role="user",
                    content=[
                        TextBlockParam(type="text", text="What is in this image?"),
                        ImageBlockParam(type="image", source=image_source),
                    ],
                ),
            ],
        )

        assert len(request.messages) == 1
        assert isinstance(request.messages[0].content, list)
        assert len(request.messages[0].content) == 2

    def test_system_message_string(self) -> None:
        """System message as string (separate from messages)."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            system="You are a helpful assistant.",
        )

        assert request.system == "You are a helpful assistant."

    def test_system_message_text_blocks(self) -> None:
        """System message as list of TextBlockParam."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            system=[
                TextBlockParam(type="text", text="You are a helpful assistant."),
                TextBlockParam(type="text", text="Be concise."),
            ],
        )

        assert isinstance(request.system, list)
        assert len(request.system) == 2

    def test_temperature_bounds_valid(self) -> None:
        """Temperature within valid range (0.0 to 1.0)."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            temperature=0.5,
        )

        assert request.temperature == 0.5

    def test_temperature_bounds_min(self) -> None:
        """Temperature at minimum (0.0)."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            temperature=0.0,
        )

        assert request.temperature == 0.0

    def test_temperature_bounds_max(self) -> None:
        """Temperature at maximum (1.0)."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            temperature=1.0,
        )

        assert request.temperature == 1.0

    def test_temperature_bounds_invalid_high(self) -> None:
        """Temperature above 1.0 is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            AnthropicMessagesRequest(
                model="claude-3-opus",
                max_tokens=1000,
                messages=[MessageParam(role="user", content="Hello")],
                temperature=1.5,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("temperature",) for e in errors)

    def test_temperature_bounds_invalid_negative(self) -> None:
        """Temperature below 0.0 is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            AnthropicMessagesRequest(
                model="claude-3-opus",
                max_tokens=1000,
                messages=[MessageParam(role="user", content="Hello")],
                temperature=-0.1,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("temperature",) for e in errors)

    def test_optional_fields(self) -> None:
        """Optional fields default correctly."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
        )

        assert request.system is None
        assert request.top_p is None
        assert request.top_k is None
        assert request.stop_sequences is None
        assert request.metadata is None

    def test_stop_sequences(self) -> None:
        """Stop sequences are accepted."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            stop_sequences=["STOP", "END"],
        )

        assert request.stop_sequences == ["STOP", "END"]


class TestAnthropicMessagesResponse:
    """Tests for response validation."""

    def test_valid_response(self) -> None:
        """Valid response with text content."""
        response = AnthropicMessagesResponse(
            id="msg_01XFDUDYJgAACzvnptvVoYEL",
            type="message",
            role="assistant",
            model="claude-3-opus-20240229",
            content=[TextBlock(type="text", text="Hello! How can I help you?")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=8),
        )

        assert response.id.startswith("msg_")
        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.content[0].text == "Hello! How can I help you?"

    def test_stop_reason_values(self) -> None:
        """All valid stop_reason values."""
        for stop_reason in ["end_turn", "max_tokens", "stop_sequence", "tool_use"]:
            response = AnthropicMessagesResponse(
                id="msg_test",
                model="claude-3-opus",
                content=[TextBlock(text="test")],
                stop_reason=stop_reason,  # type: ignore[arg-type]
                usage=Usage(input_tokens=1, output_tokens=1),
            )
            assert response.stop_reason == stop_reason

    def test_stop_reason_none(self) -> None:
        """stop_reason can be None (streaming not complete)."""
        response = AnthropicMessagesResponse(
            id="msg_test",
            model="claude-3-opus",
            content=[TextBlock(text="partial...")],
            stop_reason=None,
            usage=Usage(input_tokens=1, output_tokens=1),
        )

        assert response.stop_reason is None

    def test_stop_sequence_field(self) -> None:
        """stop_sequence captures matched sequence."""
        response = AnthropicMessagesResponse(
            id="msg_test",
            model="claude-3-opus",
            content=[TextBlock(text="Hello STOP")],
            stop_reason="stop_sequence",
            stop_sequence="STOP",
            usage=Usage(input_tokens=1, output_tokens=3),
        )

        assert response.stop_reason == "stop_sequence"
        assert response.stop_sequence == "STOP"

    def test_usage_tokens(self) -> None:
        """Usage tracks token counts."""
        usage = Usage(input_tokens=150, output_tokens=42)

        assert usage.input_tokens == 150
        assert usage.output_tokens == 42


class TestExtractAnthropicContent:
    """Tests for content extraction helper."""

    def test_string_content(self) -> None:
        """String content returns string."""
        result = extract_anthropic_content("Hello, world!")

        assert result == "Hello, world!"

    def test_list_text_blocks_pydantic(self) -> None:
        """List of TextBlockParam returns concatenated text."""
        blocks = [
            TextBlockParam(type="text", text="Hello"),
            TextBlockParam(type="text", text="world"),
        ]
        result = extract_anthropic_content(blocks)  # type: ignore[arg-type]

        assert result == "Hello world"

    def test_list_text_blocks_dict(self) -> None:
        """List of dict text blocks returns concatenated text."""
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
        ]
        result = extract_anthropic_content(blocks)  # type: ignore[arg-type]

        assert result == "Hello world"

    def test_mixed_content_extracts_text(self) -> None:
        """Mixed content extracts only text portions."""
        image_source = ImageSource(
            type="base64",
            media_type="image/png",
            data="base64data",
        )
        blocks = [
            TextBlockParam(type="text", text="Look at this"),
            ImageBlockParam(type="image", source=image_source),
            TextBlockParam(type="text", text="image"),
        ]
        result = extract_anthropic_content(blocks)  # type: ignore[arg-type]

        assert result == "Look at this image"

    def test_empty_list(self) -> None:
        """Empty list returns empty string."""
        result = extract_anthropic_content([])  # type: ignore[arg-type]

        assert result == ""


class TestStreamingEvents:
    """Tests for streaming event schemas."""

    def test_message_start_event(self) -> None:
        """MessageStartEvent structure."""
        message = MessageStartMessage(
            id="msg_test",
            model="claude-3-opus",
            usage=Usage(input_tokens=10, output_tokens=0),
        )
        event = MessageStartEvent(type="message_start", message=message)

        assert event.type == "message_start"
        assert event.message.id == "msg_test"
        assert event.message.role == "assistant"
        assert event.message.content == []

    def test_content_block_start_event(self) -> None:
        """ContentBlockStartEvent structure."""
        from mlx_manager.mlx_server.schemas.anthropic import ContentBlockStartBlock

        event = ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=ContentBlockStartBlock(type="text", text=""),
        )

        assert event.type == "content_block_start"
        assert event.index == 0
        assert event.content_block.type == "text"

    def test_content_block_delta_event(self) -> None:
        """ContentBlockDeltaEvent with TextDelta."""
        event = ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="Hello"),
        )

        assert event.type == "content_block_delta"
        assert event.index == 0
        assert event.delta.text == "Hello"

    def test_content_block_stop_event(self) -> None:
        """ContentBlockStopEvent structure."""
        event = ContentBlockStopEvent(type="content_block_stop", index=0)

        assert event.type == "content_block_stop"
        assert event.index == 0

    def test_message_delta_event(self) -> None:
        """MessageDeltaEvent with stop reason."""
        from mlx_manager.mlx_server.schemas.anthropic import (
            MessageDeltaDelta,
            MessageDeltaUsage,
        )

        event = MessageDeltaEvent(
            type="message_delta",
            delta=MessageDeltaDelta(stop_reason="end_turn"),
            usage=MessageDeltaUsage(output_tokens=15),
        )

        assert event.type == "message_delta"
        assert event.delta.stop_reason == "end_turn"
        assert event.usage.output_tokens == 15

    def test_message_stop_event(self) -> None:
        """MessageStopEvent structure."""
        event = MessageStopEvent(type="message_stop")

        assert event.type == "message_stop"


class TestMessageParam:
    """Tests for MessageParam validation."""

    def test_valid_user_role(self) -> None:
        """User role is valid."""
        msg = MessageParam(role="user", content="Hello")
        assert msg.role == "user"

    def test_valid_assistant_role(self) -> None:
        """Assistant role is valid."""
        msg = MessageParam(role="assistant", content="Hello")
        assert msg.role == "assistant"

    def test_system_role_invalid(self) -> None:
        """System role is NOT allowed in messages (use system field instead)."""
        with pytest.raises(ValidationError) as exc_info:
            MessageParam(role="system", content="You are helpful")  # type: ignore[arg-type]

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("role",) for e in errors)
