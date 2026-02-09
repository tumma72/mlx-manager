"""Tests for composable adapter architecture."""

from __future__ import annotations

from typing import Any

import pytest

from mlx_manager.mlx_server.models.adapters.composable import (
    FAMILY_REGISTRY,
    DefaultAdapter,
    GemmaAdapter,
    GLM4Adapter,
    LlamaAdapter,
    MistralAdapter,
    ModelAdapter,
    QwenAdapter,
    create_adapter,
)
from mlx_manager.mlx_server.parsers import (
    Glm4NativeParser,
    HermesJsonParser,
    LlamaXmlParser,
    NullThinkingParser,
    NullToolParser,
    ThinkTagParser,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self) -> None:
        self.eos_token_id = 0
        self.unk_token_id = -1
        self._special_tokens = {
            "<|im_end|>": 100,
            "<|eot_id|>": 200,
            "<|end_of_turn|>": 300,
            "<|eom_id|>": 400,
            "<|user|>": 500,
            "<|observation|>": 600,
            "<|endoftext|>": 700,
            "<end_of_turn>": 800,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special_tokens.get(token, self.unk_token_id)

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return "mock_template_output"


class TestModelAdapterABC:
    """Test ModelAdapter abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """ModelAdapter cannot be instantiated directly."""
        tokenizer = MockTokenizer()
        with pytest.raises(TypeError, match="abstract"):
            ModelAdapter(tokenizer)  # type: ignore


class TestDefaultAdapter:
    """Test DefaultAdapter."""

    def test_family(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        assert adapter.family == "default"

    def test_default_parsers(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        assert isinstance(adapter.tool_parser, NullToolParser)
        assert isinstance(adapter.thinking_parser, NullThinkingParser)

    def test_supports_tool_calling(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        assert adapter.supports_tool_calling() is False

    def test_supports_native_tools(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        assert adapter.supports_native_tools() is False

    def test_stop_tokens(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        assert adapter.stop_tokens == [0]

    def test_get_stream_markers(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        assert adapter.get_stream_markers() == []

    def test_apply_chat_template(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        result = adapter.apply_chat_template([{"role": "user", "content": "hello"}])
        assert result == "mock_template_output"

    def test_format_tools_for_prompt(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        assert adapter.format_tools_for_prompt([]) == ""

    def test_get_tool_call_stop_tokens(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        assert adapter.get_tool_call_stop_tokens() == []

    def test_convert_messages_tool_role(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "result",
            }
        ]
        converted = adapter.convert_messages(messages)
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert "call_123" in converted[0]["content"]
        assert "result" in converted[0]["content"]

    def test_convert_messages_assistant_with_tool_calls(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        messages = [
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            }
        ]
        converted = adapter.convert_messages(messages)
        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert "Let me check" in converted[0]["content"]
        assert "[Tool Call: get_weather" in converted[0]["content"]

    def test_clean_response(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        text = "<|im_start|>hello<|im_end|>\n\n\n\nworld"
        cleaned = adapter.clean_response(text)
        assert "<|im_start|>" not in cleaned
        assert "<|im_end|>" not in cleaned
        assert "\n\n\n" not in cleaned


class TestQwenAdapter:
    """Test QwenAdapter."""

    def test_family(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        assert adapter.family == "qwen"

    def test_default_parsers(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        assert isinstance(adapter.tool_parser, HermesJsonParser)
        assert isinstance(adapter.thinking_parser, ThinkTagParser)

    def test_supports_tool_calling(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        assert adapter.supports_tool_calling() is True

    def test_supports_native_tools(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        assert adapter.supports_native_tools() is False

    def test_stop_tokens_includes_im_end(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        assert 0 in adapter.stop_tokens
        assert 100 in adapter.stop_tokens  # <|im_end|>

    def test_get_stream_markers(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        markers = adapter.get_stream_markers()
        assert ("<tool_call>", "</tool_call>") in markers
        assert ("<think>", "</think>") in markers

    def test_format_tools_for_prompt(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            }
        ]
        result = adapter.format_tools_for_prompt(tools)
        assert "<tools>" in result
        assert "get_weather" in result
        assert "<tool_call>" in result

    def test_convert_messages_tool_role(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "result",
            }
        ]
        converted = adapter.convert_messages(messages)
        assert converted[0]["role"] == "user"
        assert "[End Tool Result]" in converted[0]["content"]

    def test_convert_messages_assistant_with_tool_calls(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        messages = [
            {
                "role": "assistant",
                "content": "Calling tool",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        }
                    }
                ],
            }
        ]
        converted = adapter.convert_messages(messages)
        assert "<tool_call>" in converted[0]["content"]
        assert "get_weather" in converted[0]["content"]


class TestGLM4Adapter:
    """Test GLM4Adapter."""

    def test_family(self) -> None:
        adapter = GLM4Adapter(MockTokenizer())
        assert adapter.family == "glm4"

    def test_default_parsers(self) -> None:
        adapter = GLM4Adapter(MockTokenizer())
        assert isinstance(adapter.tool_parser, Glm4NativeParser)
        assert isinstance(adapter.thinking_parser, ThinkTagParser)

    def test_supports_tool_calling(self) -> None:
        adapter = GLM4Adapter(MockTokenizer())
        assert adapter.supports_tool_calling() is True

    def test_supports_native_tools(self) -> None:
        adapter = GLM4Adapter(MockTokenizer())
        assert adapter.supports_native_tools() is True

    def test_stop_tokens_includes_special_tokens(self) -> None:
        adapter = GLM4Adapter(MockTokenizer())
        assert 0 in adapter.stop_tokens
        assert 500 in adapter.stop_tokens  # <|user|>
        assert 600 in adapter.stop_tokens  # <|observation|>
        assert 700 in adapter.stop_tokens  # <|endoftext|>

    def test_format_tools_for_prompt(self) -> None:
        adapter = GLM4Adapter(MockTokenizer())
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object"},
                }
            }
        ]
        result = adapter.format_tools_for_prompt(tools)
        assert "<tool>" in result
        assert "<name>get_weather</name>" in result
        assert "<description>" in result

    def test_apply_chat_template_with_tools(self) -> None:
        """GLM4 should pass tools= to tokenizer."""

        class GLM4MockTokenizer(MockTokenizer):
            def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
                if "tools" in kwargs:
                    return "template_with_tools"
                return "template_without_tools"

        adapter = GLM4Adapter(GLM4MockTokenizer())
        result = adapter.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tools=[{"function": {"name": "test"}}],
        )
        assert result == "template_with_tools"

    def test_convert_messages_assistant_with_tool_calls(self) -> None:
        adapter = GLM4Adapter(MockTokenizer())
        messages = [
            {
                "role": "assistant",
                "content": "Calling tool",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        }
                    }
                ],
            }
        ]
        converted = adapter.convert_messages(messages)
        assert "<tool_call>" in converted[0]["content"]


class TestLlamaAdapter:
    """Test LlamaAdapter."""

    def test_family(self) -> None:
        adapter = LlamaAdapter(MockTokenizer())
        assert adapter.family == "llama"

    def test_default_parsers(self) -> None:
        adapter = LlamaAdapter(MockTokenizer())
        assert isinstance(adapter.tool_parser, LlamaXmlParser)
        assert isinstance(adapter.thinking_parser, ThinkTagParser)

    def test_supports_tool_calling(self) -> None:
        adapter = LlamaAdapter(MockTokenizer())
        assert adapter.supports_tool_calling() is True

    def test_supports_native_tools(self) -> None:
        adapter = LlamaAdapter(MockTokenizer())
        assert adapter.supports_native_tools() is False

    def test_stop_tokens_includes_eot_id(self) -> None:
        adapter = LlamaAdapter(MockTokenizer())
        assert 0 in adapter.stop_tokens
        assert 200 in adapter.stop_tokens  # <|eot_id|>

    def test_stop_tokens_includes_end_of_turn(self) -> None:
        adapter = LlamaAdapter(MockTokenizer())
        assert 300 in adapter.stop_tokens  # <|end_of_turn|>

    def test_get_tool_call_stop_tokens(self) -> None:
        adapter = LlamaAdapter(MockTokenizer())
        tool_stop_tokens = adapter.get_tool_call_stop_tokens()
        assert 400 in tool_stop_tokens  # <|eom_id|>

    def test_format_tools_for_prompt(self) -> None:
        adapter = LlamaAdapter(MockTokenizer())
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object"},
                }
            }
        ]
        result = adapter.format_tools_for_prompt(tools)
        assert "get_weather:" in result
        assert "description:" in result
        assert "<function=" in result

    def test_convert_messages_assistant_with_tool_calls(self) -> None:
        adapter = LlamaAdapter(MockTokenizer())
        messages = [
            {
                "role": "assistant",
                "content": "Calling",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        }
                    }
                ],
            }
        ]
        converted = adapter.convert_messages(messages)
        assert "<function=get_weather>" in converted[0]["content"]


class TestGemmaAdapter:
    """Test GemmaAdapter."""

    def test_family(self) -> None:
        adapter = GemmaAdapter(MockTokenizer())
        assert adapter.family == "gemma"

    def test_default_parsers(self) -> None:
        adapter = GemmaAdapter(MockTokenizer())
        assert isinstance(adapter.tool_parser, NullToolParser)
        assert isinstance(adapter.thinking_parser, NullThinkingParser)

    def test_supports_tool_calling(self) -> None:
        adapter = GemmaAdapter(MockTokenizer())
        assert adapter.supports_tool_calling() is False

    def test_stop_tokens_includes_end_of_turn(self) -> None:
        adapter = GemmaAdapter(MockTokenizer())
        assert 0 in adapter.stop_tokens
        assert 800 in adapter.stop_tokens  # <end_of_turn>


class TestMistralAdapter:
    """Test MistralAdapter."""

    def test_family(self) -> None:
        adapter = MistralAdapter(MockTokenizer())
        assert adapter.family == "mistral"

    def test_default_parsers(self) -> None:
        adapter = MistralAdapter(MockTokenizer())
        assert isinstance(adapter.tool_parser, NullToolParser)
        assert isinstance(adapter.thinking_parser, NullThinkingParser)

    def test_supports_tool_calling(self) -> None:
        adapter = MistralAdapter(MockTokenizer())
        assert adapter.supports_tool_calling() is False

    def test_apply_chat_template_system_message_prepend(self) -> None:
        """Mistral prepends system message to first user message."""

        class MistralMockTokenizer(MockTokenizer):
            def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
                first_msg = messages[0]
                return first_msg["content"]

        adapter = MistralAdapter(MistralMockTokenizer())
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = adapter.apply_chat_template(messages)
        assert "You are helpful" in result
        assert "Hello" in result


class TestParserDependencyInjection:
    """Test parser dependency injection."""

    def test_override_tool_parser(self) -> None:
        """Can inject different tool parser."""
        adapter = QwenAdapter(MockTokenizer(), tool_parser=NullToolParser())
        assert isinstance(adapter.tool_parser, NullToolParser)
        assert adapter.supports_tool_calling() is False

    def test_override_thinking_parser(self) -> None:
        """Can inject different thinking parser."""
        adapter = QwenAdapter(MockTokenizer(), thinking_parser=NullThinkingParser())
        assert isinstance(adapter.thinking_parser, NullThinkingParser)

    def test_override_both_parsers(self) -> None:
        """Can inject both parsers."""
        adapter = LlamaAdapter(
            MockTokenizer(),
            tool_parser=HermesJsonParser(),
            thinking_parser=NullThinkingParser(),
        )
        assert isinstance(adapter.tool_parser, HermesJsonParser)
        assert isinstance(adapter.thinking_parser, NullThinkingParser)


class TestModelAdapterFactory:
    """Test create_adapter factory."""

    def test_create_qwen_adapter(self) -> None:
        adapter = create_adapter("qwen", MockTokenizer())
        assert isinstance(adapter, QwenAdapter)
        assert adapter.family == "qwen"

    def test_create_glm4_adapter(self) -> None:
        adapter = create_adapter("glm4", MockTokenizer())
        assert isinstance(adapter, GLM4Adapter)
        assert adapter.family == "glm4"

    def test_create_llama_adapter(self) -> None:
        adapter = create_adapter("llama", MockTokenizer())
        assert isinstance(adapter, LlamaAdapter)
        assert adapter.family == "llama"

    def test_create_gemma_adapter(self) -> None:
        adapter = create_adapter("gemma", MockTokenizer())
        assert isinstance(adapter, GemmaAdapter)
        assert adapter.family == "gemma"

    def test_create_mistral_adapter(self) -> None:
        adapter = create_adapter("mistral", MockTokenizer())
        assert isinstance(adapter, MistralAdapter)
        assert adapter.family == "mistral"

    def test_create_unknown_family_returns_default(self) -> None:
        adapter = create_adapter("unknown", MockTokenizer())
        assert isinstance(adapter, DefaultAdapter)
        assert adapter.family == "default"

    def test_create_with_parser_override(self) -> None:
        adapter = create_adapter("qwen", MockTokenizer(), tool_parser=NullToolParser())
        assert isinstance(adapter, QwenAdapter)
        assert isinstance(adapter.tool_parser, NullToolParser)


class TestComposableFamilyRegistry:
    """Test FAMILY_REGISTRY completeness."""

    def test_registry_contains_all_families(self) -> None:
        expected_families = {
            "qwen",
            "glm4",
            "llama",
            "gemma",
            "mistral",
            "default",
        }
        assert set(FAMILY_REGISTRY.keys()) == expected_families

    def test_registry_values_are_adapter_classes(self) -> None:
        for cls in FAMILY_REGISTRY.values():
            assert issubclass(cls, ModelAdapter)


class TestTokenizerAccess:
    """Test tokenizer property access."""

    def test_tokenizer_property(self) -> None:
        tokenizer = MockTokenizer()
        adapter = DefaultAdapter(tokenizer)
        assert adapter.tokenizer is tokenizer

    def test_processor_wrapped_tokenizer(self) -> None:
        """Test processor with wrapped tokenizer attribute."""

        class MockProcessor:
            def __init__(self) -> None:
                self.tokenizer = MockTokenizer()

        processor = MockProcessor()
        adapter = DefaultAdapter(processor)
        assert adapter._actual_tokenizer is processor.tokenizer


class TestStopTokensPreComputation:
    """Test stop tokens are pre-computed at init."""

    def test_stop_tokens_cached(self) -> None:
        """stop_tokens should be pre-computed, not re-computed on access."""
        tokenizer = MockTokenizer()
        adapter = QwenAdapter(tokenizer)

        # Get stop tokens multiple times
        tokens1 = adapter.stop_tokens
        tokens2 = adapter.stop_tokens

        # Should be the same list instance (cached)
        assert tokens1 is tokens2

    def test_stop_tokens_computed_at_init(self) -> None:
        """_compute_stop_tokens should be called during __init__."""
        call_count = 0

        class TestAdapter(DefaultAdapter):
            def _compute_stop_tokens(self) -> list[int]:
                nonlocal call_count
                call_count += 1
                return super()._compute_stop_tokens()

        adapter = TestAdapter(MockTokenizer())
        assert call_count == 1

        # Accessing stop_tokens shouldn't call _compute_stop_tokens again
        _ = adapter.stop_tokens
        assert call_count == 1


class TestCleanResponse:
    """Test clean_response method."""

    def test_removes_special_tokens(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        text = "<|im_start|>hello<|im_end|><|endoftext|>"
        cleaned = adapter.clean_response(text)
        assert "<|im_start|>" not in cleaned
        assert "<|im_end|>" not in cleaned
        assert "<|endoftext|>" not in cleaned
        assert "hello" in cleaned

    def test_normalizes_multiple_newlines(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        text = "hello\n\n\n\nworld"
        cleaned = adapter.clean_response(text)
        assert "\n\n\n" not in cleaned
        assert "hello\n\nworld" in cleaned

    def test_strips_whitespace(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        text = "  hello  "
        cleaned = adapter.clean_response(text)
        assert cleaned == "hello"


class TestGetStreamMarkers:
    """Test get_stream_markers combines tool and thinking markers."""

    def test_combines_markers(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        markers = adapter.get_stream_markers()

        # Should have both tool and thinking markers
        tool_markers = list(adapter.tool_parser.stream_markers)
        thinking_markers = list(adapter.thinking_parser.stream_markers)
        expected = tool_markers + thinking_markers

        assert markers == expected

    def test_empty_markers_for_null_parsers(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        markers = adapter.get_stream_markers()
        assert markers == []


class TestConvertMessagesEdgeCases:
    """Test convert_messages edge cases."""

    def test_empty_messages(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        assert adapter.convert_messages([]) == []

    def test_regular_messages_unchanged(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        converted = adapter.convert_messages(messages)
        assert converted == messages

    def test_assistant_no_content_with_tool_calls(self) -> None:
        adapter = QwenAdapter(MockTokenizer())
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "test",
                            "arguments": "{}",
                        }
                    }
                ],
            }
        ]
        converted = adapter.convert_messages(messages)
        assert "<tool_call>" in converted[0]["content"]

    def test_tool_message_missing_content(self) -> None:
        adapter = DefaultAdapter(MockTokenizer())
        messages = [{"role": "tool", "tool_call_id": "call_1"}]
        converted = adapter.convert_messages(messages)
        assert converted[0]["role"] == "user"
