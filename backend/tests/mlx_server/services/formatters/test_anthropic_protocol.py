"""Tests for Anthropic protocol translation (input parsing and stop reason conversion)."""

from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    ImageBlockParam,
    ImageSource,
    MessageParam,
    TextBlockParam,
)
from mlx_manager.mlx_server.services.formatters import (
    AnthropicFormatter,
    InternalRequest,
    anthropic_stop_to_openai,
    openai_stop_to_anthropic,
)
from mlx_manager.mlx_server.services.formatters.anthropic import _extract_text_content
from mlx_manager.models.value_objects import InferenceParams


class TestParseRequest:
    """Tests for AnthropicFormatter.parse_request() conversion."""

    def test_system_message_placed_first(self) -> None:
        """System message from Anthropic system field is placed first."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            system="You are a helpful assistant",
        )

        internal = AnthropicFormatter.parse_request(request)

        assert len(internal.messages) == 2
        assert internal.messages[0]["role"] == "system"
        assert internal.messages[0]["content"] == "You are a helpful assistant"
        assert internal.messages[1]["role"] == "user"

    def test_system_message_as_list_of_blocks(self) -> None:
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

        internal = AnthropicFormatter.parse_request(request)

        assert internal.messages[0]["role"] == "system"
        assert internal.messages[0]["content"] == "First part. Second part."

    def test_no_system_message(self) -> None:
        """Works without system message."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
        )

        internal = AnthropicFormatter.parse_request(request)

        assert len(internal.messages) == 1
        assert internal.messages[0]["role"] == "user"

    def test_string_content_preserved(self) -> None:
        """String content is preserved unchanged."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello, world!")],
        )

        internal = AnthropicFormatter.parse_request(request)

        assert internal.messages[0]["content"] == "Hello, world!"

    def test_content_blocks_extracted(self) -> None:
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

        internal = AnthropicFormatter.parse_request(request)

        assert internal.messages[0]["content"] == "First text. Second text."

    def test_mixed_content_extracts_only_text(self) -> None:
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

        internal = AnthropicFormatter.parse_request(request)

        assert internal.messages[0]["content"] == "Look at this: What is it?"

    def test_multiple_messages_preserved_in_order(self) -> None:
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

        internal = AnthropicFormatter.parse_request(request)

        assert len(internal.messages) == 3
        assert internal.messages[0]["role"] == "user"
        assert internal.messages[0]["content"] == "First message"
        assert internal.messages[1]["role"] == "assistant"
        assert internal.messages[1]["content"] == "Response"
        assert internal.messages[2]["role"] == "user"
        assert internal.messages[2]["content"] == "Follow up"

    def test_stop_sequences_mapped(self) -> None:
        """Stop sequences are mapped to stop parameter."""
        request = AnthropicMessagesRequest(
            model="claude-3",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            stop_sequences=["END", "STOP"],
        )

        internal = AnthropicFormatter.parse_request(request)

        assert internal.stop == ["END", "STOP"]

    def test_parameters_passed_through(self) -> None:
        """Temperature, top_p, max_tokens, stream are passed through."""
        request = AnthropicMessagesRequest(
            model="claude-3-opus",
            max_tokens=2000,
            messages=[MessageParam(role="user", content="Hello")],
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )

        internal = AnthropicFormatter.parse_request(request)

        assert internal.model == "claude-3-opus"
        assert internal.params.max_tokens == 2000
        assert internal.params.temperature == 0.7
        assert internal.params.top_p == 0.9
        assert internal.stream is True


class TestExtractTextContent:
    """Tests for _extract_text_content helper function."""

    def test_string_returns_unchanged(self) -> None:
        """String input returns unchanged."""
        result = _extract_text_content("Hello, world!")
        assert result == "Hello, world!"

    def test_text_block_param_list(self) -> None:
        """List of TextBlockParam returns joined text."""
        content = [
            TextBlockParam(text="First."),
            TextBlockParam(text="Second."),
        ]
        result = _extract_text_content(content)
        assert result == "First. Second."

    def test_dict_content_blocks(self) -> None:
        """Dict representations of content blocks are handled."""
        content = [
            {"type": "text", "text": "Dict text."},
            {"type": "text", "text": "More text."},
        ]
        result = _extract_text_content(content)
        assert result == "Dict text. More text."

    def test_mixed_pydantic_and_dict(self) -> None:
        """Mixed Pydantic models and dicts are handled."""
        content = [
            TextBlockParam(text="Pydantic."),
            {"type": "text", "text": "Dict."},
        ]
        result = _extract_text_content(content)
        assert result == "Pydantic. Dict."

    def test_non_text_blocks_ignored(self) -> None:
        """Non-text blocks are ignored in extraction."""
        content = [
            TextBlockParam(text="Text."),
            ImageBlockParam(source=ImageSource(media_type="image/png", data="base64...")),
        ]
        result = _extract_text_content(content)
        assert result == "Text."

    def test_empty_list_returns_empty_string(self) -> None:
        """Empty list returns empty string."""
        result = _extract_text_content([])
        assert result == ""


class TestStopReasonTranslation:
    """Tests for stop reason bidirectional translation."""

    def test_openai_stop_to_anthropic_stop(self) -> None:
        """OpenAI 'stop' maps to Anthropic 'end_turn'."""
        assert openai_stop_to_anthropic("stop") == "end_turn"

    def test_openai_length_to_anthropic_max_tokens(self) -> None:
        """OpenAI 'length' maps to Anthropic 'max_tokens'."""
        assert openai_stop_to_anthropic("length") == "max_tokens"

    def test_openai_content_filter_to_anthropic(self) -> None:
        """OpenAI 'content_filter' maps to Anthropic 'end_turn'."""
        assert openai_stop_to_anthropic("content_filter") == "end_turn"

    def test_openai_tool_calls_to_anthropic(self) -> None:
        """OpenAI 'tool_calls' maps to Anthropic 'tool_use'."""
        assert openai_stop_to_anthropic("tool_calls") == "tool_use"

    def test_openai_none_to_anthropic(self) -> None:
        """None maps to default 'end_turn'."""
        assert openai_stop_to_anthropic(None) == "end_turn"

    def test_anthropic_end_turn_to_openai_stop(self) -> None:
        """Anthropic 'end_turn' maps to OpenAI 'stop'."""
        assert anthropic_stop_to_openai("end_turn") == "stop"

    def test_anthropic_max_tokens_to_openai_length(self) -> None:
        """Anthropic 'max_tokens' maps to OpenAI 'length'."""
        assert anthropic_stop_to_openai("max_tokens") == "length"

    def test_anthropic_stop_sequence_to_openai(self) -> None:
        """Anthropic 'stop_sequence' maps to OpenAI 'stop'."""
        assert anthropic_stop_to_openai("stop_sequence") == "stop"

    def test_anthropic_tool_use_to_openai(self) -> None:
        """Anthropic 'tool_use' maps to OpenAI 'tool_calls'."""
        assert anthropic_stop_to_openai("tool_use") == "tool_calls"

    def test_anthropic_none_to_openai(self) -> None:
        """None maps to default 'stop'."""
        assert anthropic_stop_to_openai(None) == "stop"

    def test_unknown_stop_reason_defaults(self) -> None:
        """Unknown stop reasons use default values."""
        assert openai_stop_to_anthropic("unknown") == "end_turn"
        assert anthropic_stop_to_openai("unknown") == "stop"


class TestInternalRequest:
    """Tests for InternalRequest BaseModel."""

    def test_model_fields(self) -> None:
        """InternalRequest has expected fields."""
        request = InternalRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            params=InferenceParams(
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
            ),
            stream=True,
            stop=["END"],
        )

        assert request.model == "test-model"
        assert request.messages == [{"role": "user", "content": "Hello"}]
        assert request.params.max_tokens == 100
        assert request.params.temperature == 0.7
        assert request.params.top_p == 0.9
        assert request.stream is True
        assert request.stop == ["END"]

    def test_optional_fields_can_be_none(self) -> None:
        """Optional fields can be None."""
        request = InternalRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            params=InferenceParams(
                max_tokens=100,
                temperature=1.0,
                top_p=None,
            ),
            stream=False,
            stop=None,
        )

        assert request.params.top_p is None
        assert request.stop is None
