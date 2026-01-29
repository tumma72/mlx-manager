"""Tests for protocol translation between OpenAI and Anthropic formats."""

import pytest

from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    ImageBlockParam,
    ImageSource,
    MessageParam,
    TextBlockParam,
)
from mlx_manager.mlx_server.services.protocol import (
    InternalRequest,
    ProtocolTranslator,
    get_translator,
    reset_translator,
)


@pytest.fixture
def translator() -> ProtocolTranslator:
    """Create a fresh translator for each test."""
    return ProtocolTranslator()


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset the translator singleton before each test."""
    reset_translator()


class TestAnthropicToInternal:
    """Tests for anthropic_to_internal conversion."""

    def test_system_message_placed_first(self, translator: ProtocolTranslator) -> None:
        """System message from Anthropic system field is placed first."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            system="You are a helpful assistant",
        )

        internal = translator.anthropic_to_internal(request)

        assert len(internal.messages) == 2
        assert internal.messages[0]["role"] == "system"
        assert internal.messages[0]["content"] == "You are a helpful assistant"
        assert internal.messages[1]["role"] == "user"

    def test_system_message_as_list_of_blocks(self, translator: ProtocolTranslator) -> None:
        """System message can be a list of TextBlockParam."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            system=[
                TextBlockParam(text="First part."),
                TextBlockParam(text="Second part."),
            ],
        )

        internal = translator.anthropic_to_internal(request)

        assert internal.messages[0]["role"] == "system"
        assert internal.messages[0]["content"] == "First part. Second part."

    def test_no_system_message(self, translator: ProtocolTranslator) -> None:
        """Works without system message."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
        )

        internal = translator.anthropic_to_internal(request)

        assert len(internal.messages) == 1
        assert internal.messages[0]["role"] == "user"

    def test_string_content_preserved(self, translator: ProtocolTranslator) -> None:
        """String content is preserved unchanged."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello, world!")],
        )

        internal = translator.anthropic_to_internal(request)

        assert internal.messages[0]["content"] == "Hello, world!"

    def test_content_blocks_extracted(self, translator: ProtocolTranslator) -> None:
        """Content blocks are extracted and concatenated."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[
                MessageParam(
                    role="user",
                    content=[
                        TextBlockParam(text="First text."),
                        TextBlockParam(text="Second text."),
                    ],
                )
            ],
        )

        internal = translator.anthropic_to_internal(request)

        assert internal.messages[0]["content"] == "First text. Second text."

    def test_mixed_content_extracts_only_text(self, translator: ProtocolTranslator) -> None:
        """Mixed content (text + images) extracts only text."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[
                MessageParam(
                    role="user",
                    content=[
                        TextBlockParam(text="Look at this:"),
                        ImageBlockParam(
                            source=ImageSource(media_type="image/jpeg", data="base64data...")
                        ),
                        TextBlockParam(text="What is it?"),
                    ],
                )
            ],
        )

        internal = translator.anthropic_to_internal(request)

        assert internal.messages[0]["content"] == "Look at this: What is it?"

    def test_multiple_messages_preserved_in_order(self, translator: ProtocolTranslator) -> None:
        """Multiple messages maintain their order."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[
                MessageParam(role="user", content="First message"),
                MessageParam(role="assistant", content="Response"),
                MessageParam(role="user", content="Follow up"),
            ],
        )

        internal = translator.anthropic_to_internal(request)

        assert len(internal.messages) == 3
        assert internal.messages[0]["role"] == "user"
        assert internal.messages[0]["content"] == "First message"
        assert internal.messages[1]["role"] == "assistant"
        assert internal.messages[1]["content"] == "Response"
        assert internal.messages[2]["role"] == "user"
        assert internal.messages[2]["content"] == "Follow up"

    def test_stop_sequences_mapped(self, translator: ProtocolTranslator) -> None:
        """Stop sequences are mapped to stop parameter."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            stop_sequences=["END", "STOP"],
        )

        internal = translator.anthropic_to_internal(request)

        assert internal.stop == ["END", "STOP"]

    def test_parameters_passed_through(self, translator: ProtocolTranslator) -> None:
        """Temperature, top_p, max_tokens, stream are passed through."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=2000,
            messages=[MessageParam(role="user", content="Hello")],
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )

        internal = translator.anthropic_to_internal(request)

        assert internal.model == "claude-3-opus"
        assert internal.max_tokens == 2000
        assert internal.temperature == 0.7
        assert internal.top_p == 0.9
        assert internal.stream is True


class TestExtractTextContent:
    """Tests for _extract_text_content method."""

    def test_string_returns_unchanged(self, translator: ProtocolTranslator) -> None:
        """String input returns unchanged."""
        result = translator._extract_text_content("Hello, world!")
        assert result == "Hello, world!"

    def test_text_block_param_list(self, translator: ProtocolTranslator) -> None:
        """List of TextBlockParam returns joined text."""
        content = [
            TextBlockParam(text="First."),
            TextBlockParam(text="Second."),
        ]
        result = translator._extract_text_content(content)
        assert result == "First. Second."

    def test_dict_content_blocks(self, translator: ProtocolTranslator) -> None:
        """Dict representations of content blocks are handled."""
        content = [
            {"type": "text", "text": "Dict text."},
            {"type": "text", "text": "More text."},
        ]
        result = translator._extract_text_content(content)
        assert result == "Dict text. More text."

    def test_mixed_pydantic_and_dict(self, translator: ProtocolTranslator) -> None:
        """Mixed Pydantic models and dicts are handled."""
        content = [
            TextBlockParam(text="Pydantic."),
            {"type": "text", "text": "Dict."},
        ]
        result = translator._extract_text_content(content)
        assert result == "Pydantic. Dict."

    def test_non_text_blocks_ignored(self, translator: ProtocolTranslator) -> None:
        """Non-text blocks are ignored in extraction."""
        content = [
            TextBlockParam(text="Text."),
            ImageBlockParam(source=ImageSource(media_type="image/png", data="base64...")),
        ]
        result = translator._extract_text_content(content)
        assert result == "Text."

    def test_empty_list_returns_empty_string(self, translator: ProtocolTranslator) -> None:
        """Empty list returns empty string."""
        result = translator._extract_text_content([])
        assert result == ""


class TestStopReasonTranslation:
    """Tests for stop reason bidirectional translation."""

    def test_openai_stop_to_anthropic_stop(self, translator: ProtocolTranslator) -> None:
        """OpenAI 'stop' maps to Anthropic 'end_turn'."""
        assert translator.openai_stop_to_anthropic("stop") == "end_turn"

    def test_openai_length_to_anthropic_max_tokens(self, translator: ProtocolTranslator) -> None:
        """OpenAI 'length' maps to Anthropic 'max_tokens'."""
        assert translator.openai_stop_to_anthropic("length") == "max_tokens"

    def test_openai_content_filter_to_anthropic(self, translator: ProtocolTranslator) -> None:
        """OpenAI 'content_filter' maps to Anthropic 'end_turn'."""
        assert translator.openai_stop_to_anthropic("content_filter") == "end_turn"

    def test_openai_tool_calls_to_anthropic(self, translator: ProtocolTranslator) -> None:
        """OpenAI 'tool_calls' maps to Anthropic 'tool_use'."""
        assert translator.openai_stop_to_anthropic("tool_calls") == "tool_use"

    def test_openai_none_to_anthropic(self, translator: ProtocolTranslator) -> None:
        """None maps to default 'end_turn'."""
        assert translator.openai_stop_to_anthropic(None) == "end_turn"

    def test_anthropic_end_turn_to_openai_stop(self, translator: ProtocolTranslator) -> None:
        """Anthropic 'end_turn' maps to OpenAI 'stop'."""
        assert translator.anthropic_stop_to_openai("end_turn") == "stop"

    def test_anthropic_max_tokens_to_openai_length(self, translator: ProtocolTranslator) -> None:
        """Anthropic 'max_tokens' maps to OpenAI 'length'."""
        assert translator.anthropic_stop_to_openai("max_tokens") == "length"

    def test_anthropic_stop_sequence_to_openai(self, translator: ProtocolTranslator) -> None:
        """Anthropic 'stop_sequence' maps to OpenAI 'stop'."""
        assert translator.anthropic_stop_to_openai("stop_sequence") == "stop"

    def test_anthropic_tool_use_to_openai(self, translator: ProtocolTranslator) -> None:
        """Anthropic 'tool_use' maps to OpenAI 'tool_calls'."""
        assert translator.anthropic_stop_to_openai("tool_use") == "tool_calls"

    def test_anthropic_none_to_openai(self, translator: ProtocolTranslator) -> None:
        """None maps to default 'stop'."""
        assert translator.anthropic_stop_to_openai(None) == "stop"

    def test_unknown_stop_reason_defaults(self, translator: ProtocolTranslator) -> None:
        """Unknown stop reasons use default values."""
        assert translator.openai_stop_to_anthropic("unknown") == "end_turn"
        assert translator.anthropic_stop_to_openai("unknown") == "stop"


class TestInternalToAnthropicResponse:
    """Tests for internal_to_anthropic_response conversion."""

    def test_response_text_wrapped_in_text_block(self, translator: ProtocolTranslator) -> None:
        """Response text is wrapped in TextBlock."""
        response = translator.internal_to_anthropic_response(
            response_text="Hello, world!",
            request_id="msg_123",
            model="claude-3",
            stop_reason="stop",
            input_tokens=10,
            output_tokens=20,
        )

        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "Hello, world!"

    def test_stop_reason_translated(self, translator: ProtocolTranslator) -> None:
        """Stop reason is translated to Anthropic format."""
        response = translator.internal_to_anthropic_response(
            response_text="Output",
            request_id="msg_123",
            model="claude-3",
            stop_reason="length",
            input_tokens=10,
            output_tokens=20,
        )

        assert response.stop_reason == "max_tokens"

    def test_usage_populated(self, translator: ProtocolTranslator) -> None:
        """Usage statistics are populated correctly."""
        response = translator.internal_to_anthropic_response(
            response_text="Output",
            request_id="msg_123",
            model="claude-3",
            stop_reason="stop",
            input_tokens=100,
            output_tokens=50,
        )

        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50

    def test_metadata_populated(self, translator: ProtocolTranslator) -> None:
        """ID and model are set correctly."""
        response = translator.internal_to_anthropic_response(
            response_text="Output",
            request_id="msg_abc123",
            model="claude-3-opus",
            stop_reason="stop",
            input_tokens=10,
            output_tokens=20,
        )

        assert response.id == "msg_abc123"
        assert response.model == "claude-3-opus"

    def test_response_type_and_role_defaults(self, translator: ProtocolTranslator) -> None:
        """Response type is 'message' and role is 'assistant'."""
        response = translator.internal_to_anthropic_response(
            response_text="Output",
            request_id="msg_123",
            model="claude-3",
            stop_reason="stop",
            input_tokens=10,
            output_tokens=20,
        )

        assert response.type == "message"
        assert response.role == "assistant"


class TestInternalRequest:
    """Tests for InternalRequest dataclass."""

    def test_dataclass_fields(self) -> None:
        """InternalRequest has expected fields."""
        request = InternalRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            stop=["END"],
        )

        assert request.model == "test-model"
        assert request.messages == [{"role": "user", "content": "Hello"}]
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stream is True
        assert request.stop == ["END"]

    def test_optional_fields_can_be_none(self) -> None:
        """Optional fields can be None."""
        request = InternalRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=1.0,
            top_p=None,
            stream=False,
            stop=None,
        )

        assert request.top_p is None
        assert request.stop is None


class TestSingleton:
    """Tests for get_translator singleton."""

    def test_returns_same_instance(self) -> None:
        """get_translator returns the same instance."""
        reset_translator()
        translator1 = get_translator()
        translator2 = get_translator()

        assert translator1 is translator2

    def test_reset_clears_singleton(self) -> None:
        """reset_translator clears the singleton."""
        translator1 = get_translator()
        reset_translator()
        translator2 = get_translator()

        assert translator1 is not translator2

    def test_singleton_is_protocol_translator(self) -> None:
        """get_translator returns a ProtocolTranslator instance."""
        translator = get_translator()
        assert isinstance(translator, ProtocolTranslator)
