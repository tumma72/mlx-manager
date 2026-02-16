"""Tests for composable adapter architecture."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from typing import Any
from unittest.mock import patch

import pytest

from mlx_manager.mlx_server.models.adapters.composable import (
    FAMILY_REGISTRY,
    DefaultAdapter,
    ModelAdapter,
    create_adapter,
)
from mlx_manager.mlx_server.models.adapters.configs import FamilyConfig
from mlx_manager.mlx_server.models.ir import (
    AudioResult,
    EmbeddingResult,
    PreparedInput,
    StreamEvent,
    TextResult,
    TranscriptionResult,
)
from mlx_manager.mlx_server.parsers import (
    Glm4NativeParser,
    HermesJsonParser,
    LlamaXmlParser,
    NullThinkingParser,
    NullToolParser,
    ThinkTagParser,
)


class FakeTokenizer:
    """Lightweight fake tokenizer with real behavior for testing."""

    def __init__(
        self,
        eos_token_id: int = 0,
        special_tokens: dict[str, int] | None = None,
        raise_on_convert: bool = False,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.unk_token_id = -1
        self._special_tokens = special_tokens or {
            "<|im_end|>": 100,
            "<|eot_id|>": 200,
            "<|end_of_turn|>": 300,
            "<|eom_id|>": 400,
            "<|user|>": 500,
            "<|observation|>": 600,
            "<|endoftext|>": 700,
            "<end_of_turn>": 800,
        }
        self._raise_on_convert = raise_on_convert

    def convert_tokens_to_ids(self, token: str) -> int:
        if self._raise_on_convert:
            raise RuntimeError("Tokenizer error")
        return self._special_tokens.get(token, self.unk_token_id)

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>{content}")
        suffix = "<|assistant|>" if kwargs.get("add_generation_prompt") else ""
        tools_info = ""
        if "tools" in kwargs:
            tools_info = "[TOOLS_PASSED]"
        thinking_info = ""
        template_options = kwargs.get("template_options", {})
        if isinstance(template_options, dict) and template_options.get("enable_thinking"):
            thinking_info = "[THINKING_ON]"
        return "".join(parts) + suffix + tools_info + thinking_info

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """Simple token encoding - each character is a token."""
        return list(range(len(text)))


# ── Backward compatibility ────────────────────────────────────────


class TestModelAdapterAlias:
    """Test that DefaultAdapter is an alias for ModelAdapter."""

    def test_default_adapter_is_model_adapter(self) -> None:
        assert DefaultAdapter is ModelAdapter

    def test_can_instantiate_model_adapter_directly(self) -> None:
        """ModelAdapter is concrete (no longer abstract) with FamilyConfig."""
        adapter = ModelAdapter(tokenizer=FakeTokenizer())
        assert adapter.family == "default"


# ── DefaultAdapter (default config) ──────────────────────────────


class TestDefaultAdapter:
    """Test ModelAdapter with default FamilyConfig."""

    def test_family(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        assert adapter.family == "default"

    def test_default_parsers(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        assert isinstance(adapter.tool_parser, NullToolParser)
        assert isinstance(adapter.thinking_parser, NullThinkingParser)

    def test_supports_tool_calling(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        assert adapter.supports_tool_calling() is False

    def test_supports_native_tools(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        assert adapter.supports_native_tools() is False

    def test_stop_tokens(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        assert adapter.stop_tokens == [0]

    def test_get_stream_markers(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        assert adapter.get_stream_markers() == []

    def test_apply_chat_template(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        result = adapter.apply_chat_template([{"role": "user", "content": "hello"}])
        assert "hello" in result

    def test_format_tools_for_prompt(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        assert adapter.format_tools_for_prompt([]) == ""

    def test_get_tool_call_stop_tokens(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        assert adapter.get_tool_call_stop_tokens() == []

    def test_convert_messages_tool_role(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
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
        adapter = create_adapter("default", FakeTokenizer())
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
        adapter = create_adapter("default", FakeTokenizer())
        text = "<|im_start|>hello<|im_end|>\n\n\n\nworld"
        cleaned = adapter.clean_response(text)
        assert "<|im_start|>" not in cleaned
        assert "<|im_end|>" not in cleaned
        assert "\n\n\n" not in cleaned


# ── Qwen adapter ─────────────────────────────────────────────────


class TestQwenAdapter:
    """Test ModelAdapter with qwen FamilyConfig."""

    def test_family(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        assert adapter.family == "qwen"

    def test_default_parsers(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        assert isinstance(adapter.tool_parser, HermesJsonParser)
        assert isinstance(adapter.thinking_parser, ThinkTagParser)

    def test_supports_tool_calling(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        assert adapter.supports_tool_calling() is True

    def test_supports_native_tools(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        assert adapter.supports_native_tools() is False

    def test_stop_tokens_includes_im_end(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        assert 0 in adapter.stop_tokens
        assert 100 in adapter.stop_tokens  # <|im_end|>

    def test_get_stream_markers(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        markers = adapter.get_stream_markers()
        assert ("<tool_call>", "</tool_call>") in markers
        assert ("<think>", "</think>") in markers

    def test_format_tools_for_prompt(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
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
        adapter = create_adapter("qwen", FakeTokenizer())
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
        adapter = create_adapter("qwen", FakeTokenizer())
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


# ── GLM4 adapter ─────────────────────────────────────────────────


class TestGLM4Adapter:
    """Test ModelAdapter with glm4 FamilyConfig."""

    def test_family(self) -> None:
        adapter = create_adapter("glm4", FakeTokenizer())
        assert adapter.family == "glm4"

    def test_default_parsers(self) -> None:
        adapter = create_adapter("glm4", FakeTokenizer())
        assert isinstance(adapter.tool_parser, Glm4NativeParser)
        assert isinstance(adapter.thinking_parser, ThinkTagParser)

    def test_supports_tool_calling(self) -> None:
        adapter = create_adapter("glm4", FakeTokenizer())
        assert adapter.supports_tool_calling() is True

    def test_supports_native_tools(self) -> None:
        adapter = create_adapter("glm4", FakeTokenizer())
        assert adapter.supports_native_tools() is True

    def test_stop_tokens_includes_special_tokens(self) -> None:
        adapter = create_adapter("glm4", FakeTokenizer())
        assert 0 in adapter.stop_tokens
        assert 500 in adapter.stop_tokens  # <|user|>
        assert 600 in adapter.stop_tokens  # <|observation|>
        assert 700 in adapter.stop_tokens  # <|endoftext|>

    def test_format_tools_for_prompt(self) -> None:
        adapter = create_adapter("glm4", FakeTokenizer())
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

        class GLM4FakeTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
                if "tools" in kwargs:
                    return "template_with_tools"
                return "template_without_tools"

        adapter = create_adapter("glm4", GLM4FakeTokenizer())
        result = adapter.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tools=[{"function": {"name": "test"}}],
        )
        assert result == "template_with_tools"

    def test_convert_messages_assistant_with_tool_calls(self) -> None:
        adapter = create_adapter("glm4", FakeTokenizer())
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


# ── Llama adapter ────────────────────────────────────────────────


class TestLlamaAdapter:
    """Test ModelAdapter with llama FamilyConfig."""

    def test_family(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
        assert adapter.family == "llama"

    def test_default_parsers(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
        assert isinstance(adapter.tool_parser, LlamaXmlParser)
        assert isinstance(adapter.thinking_parser, ThinkTagParser)

    def test_supports_tool_calling(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
        assert adapter.supports_tool_calling() is True

    def test_supports_native_tools(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
        assert adapter.supports_native_tools() is False

    def test_stop_tokens_includes_eot_id(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
        assert 0 in adapter.stop_tokens
        assert 200 in adapter.stop_tokens  # <|eot_id|>

    def test_stop_tokens_includes_end_of_turn(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
        assert 300 in adapter.stop_tokens  # <|end_of_turn|>

    def test_get_tool_call_stop_tokens(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
        tool_stop_tokens = adapter.get_tool_call_stop_tokens()
        assert 400 in tool_stop_tokens  # <|eom_id|>

    def test_format_tools_for_prompt(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
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
        adapter = create_adapter("llama", FakeTokenizer())
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


# ── Gemma adapter ────────────────────────────────────────────────


class TestGemmaAdapter:
    """Test ModelAdapter with gemma FamilyConfig."""

    def test_family(self) -> None:
        adapter = create_adapter("gemma", FakeTokenizer())
        assert adapter.family == "gemma"

    def test_default_parsers(self) -> None:
        adapter = create_adapter("gemma", FakeTokenizer())
        assert isinstance(adapter.tool_parser, NullToolParser)
        assert isinstance(adapter.thinking_parser, NullThinkingParser)

    def test_supports_tool_calling(self) -> None:
        adapter = create_adapter("gemma", FakeTokenizer())
        assert adapter.supports_tool_calling() is False

    def test_stop_tokens_includes_end_of_turn(self) -> None:
        adapter = create_adapter("gemma", FakeTokenizer())
        assert 0 in adapter.stop_tokens
        assert 800 in adapter.stop_tokens  # <end_of_turn>


# ── Mistral adapter ──────────────────────────────────────────────


class TestMistralAdapter:
    """Test ModelAdapter with mistral FamilyConfig."""

    def test_family(self) -> None:
        adapter = create_adapter("mistral", FakeTokenizer())
        assert adapter.family == "mistral"

    def test_default_parsers(self) -> None:
        adapter = create_adapter("mistral", FakeTokenizer())
        assert isinstance(adapter.tool_parser, NullToolParser)
        assert isinstance(adapter.thinking_parser, NullThinkingParser)

    def test_supports_tool_calling(self) -> None:
        adapter = create_adapter("mistral", FakeTokenizer())
        assert adapter.supports_tool_calling() is False

    def test_apply_chat_template_system_message_prepend(self) -> None:
        """Mistral prepends system message to first user message."""
        adapter = create_adapter("mistral", FakeTokenizer())
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = adapter.apply_chat_template(messages)
        assert "You are helpful" in result
        assert "Hello" in result


# ── Embeddings adapter ───────────────────────────────────────────


class TestEmbeddingsAdapter:
    """Test ModelAdapter with embeddings FamilyConfig."""

    def test_family(self) -> None:
        adapter = create_adapter("embeddings", FakeTokenizer())
        assert adapter.family == "embeddings"

    def test_default_parsers(self) -> None:
        adapter = create_adapter("embeddings", FakeTokenizer())
        assert isinstance(adapter.tool_parser, NullToolParser)
        assert isinstance(adapter.thinking_parser, NullThinkingParser)

    def test_supports_tool_calling(self) -> None:
        adapter = create_adapter("embeddings", FakeTokenizer())
        assert adapter.supports_tool_calling() is False

    def test_supports_native_tools(self) -> None:
        adapter = create_adapter("embeddings", FakeTokenizer())
        assert adapter.supports_native_tools() is False

    def test_get_stream_markers_empty(self) -> None:
        adapter = create_adapter("embeddings", FakeTokenizer())
        assert adapter.get_stream_markers() == []

    def test_format_tools_for_prompt_empty(self) -> None:
        adapter = create_adapter("embeddings", FakeTokenizer())
        assert adapter.format_tools_for_prompt([]) == ""

    def test_registry_includes_embeddings(self) -> None:
        assert "embeddings" in FAMILY_REGISTRY
        assert isinstance(FAMILY_REGISTRY["embeddings"], FamilyConfig)

    def test_create_adapter_embeddings(self) -> None:
        adapter = create_adapter("embeddings", FakeTokenizer())
        assert adapter.family == "embeddings"


# ── Parser dependency injection ──────────────────────────────────


class TestParserDependencyInjection:
    """Test parser dependency injection."""

    def test_override_tool_parser(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer(), tool_parser=NullToolParser())
        assert isinstance(adapter.tool_parser, NullToolParser)
        assert adapter.supports_tool_calling() is False

    def test_override_thinking_parser(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer(), thinking_parser=NullThinkingParser())
        assert isinstance(adapter.thinking_parser, NullThinkingParser)

    def test_override_both_parsers(self) -> None:
        adapter = create_adapter(
            "llama",
            FakeTokenizer(),
            tool_parser=HermesJsonParser(),
            thinking_parser=NullThinkingParser(),
        )
        assert isinstance(adapter.tool_parser, HermesJsonParser)
        assert isinstance(adapter.thinking_parser, NullThinkingParser)


# ── Factory ──────────────────────────────────────────────────────


class TestModelAdapterFactory:
    """Test create_adapter factory."""

    def test_create_qwen_adapter(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        assert adapter.family == "qwen"

    def test_create_glm4_adapter(self) -> None:
        adapter = create_adapter("glm4", FakeTokenizer())
        assert adapter.family == "glm4"

    def test_create_llama_adapter(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
        assert adapter.family == "llama"

    def test_create_gemma_adapter(self) -> None:
        adapter = create_adapter("gemma", FakeTokenizer())
        assert adapter.family == "gemma"

    def test_create_mistral_adapter(self) -> None:
        adapter = create_adapter("mistral", FakeTokenizer())
        assert adapter.family == "mistral"

    def test_create_unknown_family_returns_default(self) -> None:
        adapter = create_adapter("unknown", FakeTokenizer())
        assert adapter.family == "default"

    def test_create_with_parser_override(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer(), tool_parser=NullToolParser())
        assert isinstance(adapter.tool_parser, NullToolParser)

    def test_create_with_model_id(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer(), model_id="mlx-community/Qwen-7B")
        assert adapter._model_id == "mlx-community/Qwen-7B"


# ── Family registry ──────────────────────────────────────────────


class TestComposableFamilyRegistry:
    """Test FAMILY_REGISTRY completeness."""

    def test_registry_contains_all_families(self) -> None:
        expected_families = {
            "qwen",
            "glm4",
            "llama",
            "gemma",
            "mistral",
            "liquid",
            "whisper",
            "kokoro",
            "audio_default",
            "embeddings",
            "default",
        }
        assert set(FAMILY_REGISTRY.keys()) == expected_families

    def test_registry_values_are_family_configs(self) -> None:
        for config in FAMILY_REGISTRY.values():
            assert isinstance(config, FamilyConfig)


# ── Tokenizer access ─────────────────────────────────────────────


class TestTokenizerAccess:
    """Test tokenizer property access."""

    def test_tokenizer_property(self) -> None:
        tokenizer = FakeTokenizer()
        adapter = create_adapter("default", tokenizer)
        assert adapter.tokenizer is tokenizer

    def test_processor_wrapped_tokenizer(self) -> None:
        """Test processor with wrapped tokenizer attribute."""

        class FakeProcessor:
            def __init__(self) -> None:
                self.tokenizer = FakeTokenizer()

        processor = FakeProcessor()
        adapter = create_adapter("default", processor)
        assert adapter._actual_tokenizer is processor.tokenizer

    def test_none_tokenizer(self) -> None:
        """Audio adapters pass None tokenizer."""
        adapter = create_adapter("kokoro", None)
        assert adapter.tokenizer is None
        assert adapter._actual_tokenizer is None
        # Stop tokens should be empty with no tokenizer
        assert adapter.stop_tokens == []


# ── Stop tokens pre-computation ──────────────────────────────────


class TestStopTokensPreComputation:
    """Test stop tokens are pre-computed at init."""

    def test_stop_tokens_cached(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        tokens1 = adapter.stop_tokens
        tokens2 = adapter.stop_tokens
        assert tokens1 is tokens2

    def test_compute_stop_tokens_exception_path(self) -> None:
        """When convert_tokens_to_ids raises, the token is silently skipped."""
        tokenizer = FakeTokenizer(raise_on_convert=True)
        adapter = create_adapter("qwen", tokenizer)
        # Should only have eos_token_id since convert failed for extra tokens
        assert adapter.stop_tokens == [0]

    def test_compute_stop_tokens_no_eos(self) -> None:
        """Tokenizer without eos_token_id returns empty base list."""
        tokenizer = FakeTokenizer()
        del tokenizer.eos_token_id
        adapter = create_adapter("default", tokenizer)
        assert adapter.stop_tokens == []


# ── Clean response ───────────────────────────────────────────────


class TestCleanResponse:
    """Test clean_response method."""

    def test_removes_special_tokens(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        text = "<|im_start|>hello<|im_end|><|endoftext|>"
        cleaned = adapter.clean_response(text)
        assert "<|im_start|>" not in cleaned
        assert "<|im_end|>" not in cleaned
        assert "<|endoftext|>" not in cleaned
        assert "hello" in cleaned

    def test_normalizes_multiple_newlines(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        text = "hello\n\n\n\nworld"
        cleaned = adapter.clean_response(text)
        assert "\n\n\n" not in cleaned
        assert "hello\n\nworld" in cleaned

    def test_strips_whitespace(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        text = "  hello  "
        cleaned = adapter.clean_response(text)
        assert cleaned == "hello"


# ── Stream markers ───────────────────────────────────────────────


class TestGetStreamMarkers:
    """Test get_stream_markers combines tool and thinking markers."""

    def test_combines_markers(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        markers = adapter.get_stream_markers()
        tool_markers = list(adapter.tool_parser.stream_markers)
        thinking_markers = list(adapter.thinking_parser.stream_markers)
        expected = tool_markers + thinking_markers
        assert markers == expected

    def test_empty_markers_for_null_parsers(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        markers = adapter.get_stream_markers()
        assert markers == []


# ── Convert messages edge cases ──────────────────────────────────


class TestConvertMessagesEdgeCases:
    """Test convert_messages edge cases."""

    def test_empty_messages(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        assert adapter.convert_messages([]) == []

    def test_regular_messages_unchanged(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        converted = adapter.convert_messages(messages)
        assert converted == messages

    def test_assistant_no_content_with_tool_calls(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
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
        adapter = create_adapter("default", FakeTokenizer())
        messages = [{"role": "tool", "tool_call_id": "call_1"}]
        converted = adapter.convert_messages(messages)
        assert converted[0]["role"] == "user"


# ── _prepare_tools ───────────────────────────────────────────────


class TestPrepareTools:
    """Test _prepare_tools with various tool delivery scenarios."""

    def test_no_tools_returns_messages_unchanged(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        msgs = [{"role": "user", "content": "hi"}]
        effective, native = adapter._prepare_tools(msgs, None)
        assert effective is msgs
        assert native is None

    def test_empty_tools_returns_messages_unchanged(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        msgs = [{"role": "user", "content": "hi"}]
        effective, native = adapter._prepare_tools(msgs, [])
        assert effective is msgs
        assert native is None

    def test_native_tools_returns_tools_for_template(self) -> None:
        """GLM4 has native_tools=True, so tools pass through to template."""
        adapter = create_adapter("glm4", FakeTokenizer())
        msgs = [{"role": "user", "content": "hi"}]
        tools = [{"function": {"name": "test"}}]
        effective, native = adapter._prepare_tools(msgs, tools)
        assert effective is msgs
        assert native is tools

    def test_non_native_tools_injects_into_system_message(self) -> None:
        """Qwen injects tools into system message (non-native delivery)."""
        adapter = create_adapter("qwen", FakeTokenizer())
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
        ]
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }
            }
        ]
        effective, native = adapter._prepare_tools(msgs, tools)
        assert native is None
        # The system message should now contain the tool prompt
        assert "get_weather" in effective[0]["content"]
        assert "You are helpful" in effective[0]["content"]

    def test_non_native_tools_creates_system_message_when_missing(self) -> None:
        """When no system message exists, one is created for tool injection."""
        adapter = create_adapter("qwen", FakeTokenizer())
        msgs = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }
            }
        ]
        effective, native = adapter._prepare_tools(msgs, tools)
        assert native is None
        assert effective[0]["role"] == "system"
        assert "get_weather" in effective[0]["content"]

    def test_non_native_tools_with_no_formatter_returns_unchanged(self) -> None:
        """Default adapter has no tool formatter, so tools are not injected."""
        adapter = create_adapter("default", FakeTokenizer())
        msgs = [{"role": "user", "content": "hi"}]
        tools = [{"function": {"name": "test"}}]
        # Default has no tool_format_strategy so format_tools_for_prompt returns ""
        effective, native = adapter._prepare_tools(msgs, tools)
        assert effective is msgs
        assert native is None


# ── apply_chat_template ──────────────────────────────────────────


class TestApplyChatTemplate:
    """Test apply_chat_template with various configurations."""

    def test_default_template_without_tools(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        result = adapter.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
        )
        assert "hello" in result

    def test_native_tools_passed_to_tokenizer(self) -> None:
        """When native_tools, tools= kwarg is passed to apply_chat_template."""
        adapter = create_adapter("glm4", FakeTokenizer())
        result = adapter.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tools=[{"function": {"name": "test"}}],
        )
        # Our FakeTokenizer adds [TOOLS_PASSED] when tools kwarg is present
        assert "[TOOLS_PASSED]" in result

    def test_template_strategy_takes_precedence(self) -> None:
        """When a template_strategy is set, it is used instead of direct tokenizer call."""
        adapter = create_adapter("qwen", FakeTokenizer())
        result = adapter.apply_chat_template(
            [{"role": "user", "content": "hello"}],
        )
        # Qwen template strategy is used
        assert "hello" in result


# ── get_tool_call_stop_tokens ────────────────────────────────────


class TestGetToolCallStopTokens:
    """Test get_tool_call_stop_tokens edge cases."""

    def test_returns_tokens_for_llama(self) -> None:
        adapter = create_adapter("llama", FakeTokenizer())
        tokens = adapter.get_tool_call_stop_tokens()
        assert 400 in tokens  # <|eom_id|>

    def test_exception_in_convert_is_silently_skipped(self) -> None:
        """When convert_tokens_to_ids raises, that token is skipped."""
        tokenizer = FakeTokenizer(raise_on_convert=True)
        adapter = create_adapter("llama", tokenizer)
        tokens = adapter.get_tool_call_stop_tokens()
        assert tokens == []

    def test_no_tokenizer_returns_empty(self) -> None:
        """Audio adapters with no tokenizer return empty list."""
        adapter = create_adapter("kokoro", None)
        assert adapter.get_tool_call_stop_tokens() == []

    def test_no_tool_call_stop_tokens_config_returns_empty(self) -> None:
        """Families without tool_call_stop_tokens return empty."""
        adapter = create_adapter("qwen", FakeTokenizer())
        assert adapter.get_tool_call_stop_tokens() == []


# ── create_stream_processor ──────────────────────────────────────


class TestCreateStreamProcessor:
    """Test create_stream_processor factory method."""

    def test_creates_processor(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        processor = adapter.create_stream_processor()
        assert processor is not None

    def test_detects_thinking_mode_from_prompt(self) -> None:
        """Prompt ending with <think> starts processor in thinking mode."""
        adapter = create_adapter("qwen", FakeTokenizer())
        processor = adapter.create_stream_processor(prompt="some text\n<think>")
        assert processor._in_pattern is True
        assert processor._is_thinking_pattern is True

    def test_no_thinking_mode_for_empty_prompt(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        processor = adapter.create_stream_processor(prompt="")
        assert processor._in_pattern is False

    def test_no_thinking_mode_for_regular_prompt(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        processor = adapter.create_stream_processor(prompt="Hello world")
        assert processor._in_pattern is False


# ── post_load_configure ──────────────────────────────────────────


class TestPostLoadConfigure:
    """Test post_load_configure hook delegation."""

    @pytest.mark.anyio
    async def test_no_hook_does_nothing(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        # Should not raise
        await adapter.post_load_configure(model=object(), model_id="test/model")

    @pytest.mark.anyio
    async def test_delegates_to_config_hook(self) -> None:
        """Post-load hook is called when configured."""
        called_with: list[tuple[Any, str]] = []

        async def fake_hook(model: Any, model_id: str) -> None:
            called_with.append((model, model_id))

        config = FamilyConfig(family="test_hook", post_load_hook=fake_hook)
        adapter = ModelAdapter(config=config, tokenizer=FakeTokenizer())
        model = object()
        await adapter.post_load_configure(model, "test/model")
        assert len(called_with) == 1
        assert called_with[0] == (model, "test/model")


# ── prepare_input (text path) ────────────────────────────────────


class TestPrepareInput:
    """Test prepare_input for text models."""

    def test_basic_text_input(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        result = adapter.prepare_input(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(result, PreparedInput)
        assert "Hello" in result.prompt
        assert 0 in result.stop_token_ids  # eos_token_id
        assert result.pixel_values is None

    def test_text_input_with_tools(self) -> None:
        """Tools are delivered and tool_call_stop_tokens added."""
        adapter = create_adapter("llama", FakeTokenizer())
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }
            }
        ]
        result = adapter.prepare_input(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=tools,
        )
        assert isinstance(result, PreparedInput)
        # Llama has tool_call_stop_tokens=["<|eom_id|>"] -> token ID 400
        assert 400 in result.stop_token_ids

    def test_text_input_tools_without_support_no_injection(self) -> None:
        """Default adapter doesn't support tools, so tools are ignored."""
        adapter = create_adapter("default", FakeTokenizer())
        tools = [{"function": {"name": "test"}}]
        result = adapter.prepare_input(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        assert isinstance(result, PreparedInput)
        # No tool stop tokens added
        assert result.stop_token_ids == [0]

    def test_text_input_tools_with_prompt_injection(self) -> None:
        """enable_prompt_injection forces tool delivery even on default adapter."""
        adapter = create_adapter("default", FakeTokenizer())
        tools = [{"function": {"name": "test"}}]
        result = adapter.prepare_input(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            enable_prompt_injection=True,
        )
        assert isinstance(result, PreparedInput)

    def test_text_input_with_thinking(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        result = adapter.prepare_input(
            messages=[{"role": "user", "content": "Think about this"}],
            template_options={"enable_thinking": True},
        )
        assert isinstance(result, PreparedInput)

    def test_vision_input_delegates_to_vlm(self) -> None:
        """Vision path uses mlx-vlm for template application."""
        import sys
        from unittest.mock import MagicMock

        adapter = create_adapter("default", FakeTokenizer(), model_id="test/vision-model")
        fake_images = ["fake_image_data"]

        # Create fake mlx_vlm modules for local imports inside prepare_input
        fake_vlm_utils = MagicMock()
        fake_vlm_utils.load_config.return_value = {"model_type": "qwen2_vl"}
        fake_vlm_prompt = MagicMock()
        fake_vlm_prompt.apply_chat_template.return_value = "vision_prompt"

        with patch.dict(
            sys.modules,
            {
                "mlx_vlm": MagicMock(),
                "mlx_vlm.utils": fake_vlm_utils,
                "mlx_vlm.prompt_utils": fake_vlm_prompt,
            },
        ):
            result = adapter.prepare_input(
                messages=[{"role": "user", "content": "Describe this image"}],
                images=fake_images,
            )

        assert isinstance(result, PreparedInput)
        assert result.prompt == "vision_prompt"
        assert result.pixel_values == fake_images
        fake_vlm_prompt.apply_chat_template.assert_called_once()

    def test_vision_input_multipart_content(self) -> None:
        """Vision path extracts text from multipart content."""
        import sys
        from unittest.mock import MagicMock

        adapter = create_adapter("default", FakeTokenizer(), model_id="test/vision-model")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]

        fake_vlm_utils = MagicMock()
        fake_vlm_utils.load_config.return_value = {"model_type": "qwen2_vl"}
        fake_vlm_prompt = MagicMock()
        fake_vlm_prompt.apply_chat_template.return_value = "vision_prompt"

        with patch.dict(
            sys.modules,
            {
                "mlx_vlm": MagicMock(),
                "mlx_vlm.utils": fake_vlm_utils,
                "mlx_vlm.prompt_utils": fake_vlm_prompt,
            },
        ):
            result = adapter.prepare_input(
                messages=messages,
                images=["fake_image"],
            )

        assert isinstance(result, PreparedInput)
        # The vlm template should have been called with text extracted from multipart
        call_args = fake_vlm_prompt.apply_chat_template.call_args
        assert "What's in this image?" in call_args[0][2]  # text_prompt arg


# ── process_complete ─────────────────────────────────────────────


class TestProcessComplete:
    """Test process_complete post-processing pipeline."""

    def test_clean_text_no_tools_no_thinking(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        result = adapter.process_complete("Hello world!")
        assert isinstance(result, TextResult)
        assert result.content == "Hello world!"
        assert result.reasoning_content is None
        assert result.tool_calls is None
        assert result.finish_reason == "stop"

    def test_extracts_thinking_content(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        raw = "<think>Let me think about this</think>The answer is 42."
        result = adapter.process_complete(raw)
        assert isinstance(result, TextResult)
        assert result.reasoning_content == "Let me think about this"
        assert "42" in result.content
        assert "<think>" not in result.content

    def test_extracts_tool_calls(self) -> None:
        adapter = create_adapter("qwen", FakeTokenizer())
        raw = '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'
        result = adapter.process_complete(raw)
        assert isinstance(result, TextResult)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.finish_reason == "tool_calls"

    def test_finish_reason_preserved_when_no_tools(self) -> None:
        adapter = create_adapter("default", FakeTokenizer())
        result = adapter.process_complete("Hello", finish_reason="length")
        assert result.finish_reason == "length"


# ── generate() ───────────────────────────────────────────────────


class _FakeStreamResponse:
    """Fake response object matching mlx_lm.stream_generate output."""

    def __init__(self, text: str, token: int) -> None:
        self.text = text
        self.token = token


class TestGenerate:
    """Test generate() with mocked metal thread and GPU libraries."""

    @pytest.mark.anyio
    async def test_generate_text(self) -> None:
        """Text generation with mocked stream_generate."""
        adapter = create_adapter("default", FakeTokenizer())

        responses = [
            _FakeStreamResponse("Hello", 1),
            _FakeStreamResponse(" world", 2),
            _FakeStreamResponse("!", 3),
        ]

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch("mlx_lm.stream_generate", return_value=iter(responses)),
            patch("mlx_lm.sample_utils.make_sampler", return_value=None),
        ):
            result = await adapter.generate(
                model=object(),
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                temperature=0.7,
            )

        assert isinstance(result, TextResult)
        assert "Hello world!" in result.content
        assert result.finish_reason == "length"

    @pytest.mark.anyio
    async def test_generate_text_hits_stop_token(self) -> None:
        """Generation stops when a stop token is encountered."""
        adapter = create_adapter("default", FakeTokenizer())

        responses = [
            _FakeStreamResponse("Hello", 1),
            _FakeStreamResponse("", 0),  # eos_token_id = 0
        ]

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch("mlx_lm.stream_generate", return_value=iter(responses)),
            patch("mlx_lm.sample_utils.make_sampler", return_value=None),
        ):
            result = await adapter.generate(
                model=object(),
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert isinstance(result, TextResult)
        assert result.finish_reason == "stop"

    @pytest.mark.anyio
    async def test_generate_vision(self) -> None:
        """Vision generation uses mlx-vlm."""
        adapter = create_adapter("default", FakeTokenizer(), model_id="test/vision")

        class FakeVLMResponse:
            text = "I see a cat"

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch("mlx_vlm.utils.load_config", return_value={"model_type": "qwen2_vl"}),
            patch(
                "mlx_vlm.prompt_utils.apply_chat_template",
                return_value="vision_prompt",
            ),
            patch("mlx_vlm.generate", return_value=FakeVLMResponse()),
        ):
            result = await adapter.generate(
                model=object(),
                messages=[{"role": "user", "content": "Describe this"}],
                images=["fake_image"],
            )

        assert isinstance(result, TextResult)
        assert "cat" in result.content


# ── generate_step() ──────────────────────────────────────────────


class TestGenerateStep:
    """Test generate_step() streaming generation."""

    @pytest.mark.anyio
    async def test_generate_step_text(self) -> None:
        """Streaming text generation yields events then final TextResult."""
        adapter = create_adapter("default", FakeTokenizer())

        tokens = [
            ("Hello", 1, False),
            (" world", 2, False),
            ("", 0, True),  # stop token
        ]

        async def fake_stream(fn: Any, poll_interval: float = 0.1) -> AsyncGenerator[Any, None]:
            for item in fn():
                yield item

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.stream_from_metal_thread",
                side_effect=fake_stream,
            ),
            patch("mlx_lm.stream_generate") as mock_gen,
            patch("mlx_lm.sample_utils.make_sampler", return_value=None),
        ):
            # Make stream_generate return our tokens
            def make_responses() -> Iterator[_FakeStreamResponse]:
                for text, token_id, _ in tokens:
                    yield _FakeStreamResponse(text, token_id)

            mock_gen.return_value = make_responses()

            events: list[Any] = []
            async for event in adapter.generate_step(
                model=object(),
                messages=[{"role": "user", "content": "Hi"}],
            ):
                events.append(event)

        # Should have content events + final TextResult
        assert len(events) >= 1
        assert isinstance(events[-1], TextResult)
        assert events[-1].finish_reason == "stop"

    @pytest.mark.anyio
    async def test_generate_step_vision(self) -> None:
        """Vision streaming yields single event + TextResult."""
        adapter = create_adapter("default", FakeTokenizer(), model_id="test/vision")

        class FakeVLMResponse:
            text = "A beautiful sunset"

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch("mlx_vlm.utils.load_config", return_value={"model_type": "qwen2_vl"}),
            patch(
                "mlx_vlm.prompt_utils.apply_chat_template",
                return_value="vision_prompt",
            ),
            patch("mlx_vlm.generate", return_value=FakeVLMResponse()),
        ):
            events: list[Any] = []
            async for event in adapter.generate_step(
                model=object(),
                messages=[{"role": "user", "content": "Describe"}],
                images=["fake_image"],
            ):
                events.append(event)

        # First event should be a StreamEvent with content
        assert isinstance(events[0], StreamEvent)
        assert "sunset" in (events[0].content or "")
        # Last event is the processed TextResult
        assert isinstance(events[-1], TextResult)


# ── generate_embeddings() ────────────────────────────────────────


class TestGenerateEmbeddings:
    """Test generate_embeddings with fake model and tokenizer."""

    @pytest.mark.anyio
    async def test_generate_embeddings_basic(self) -> None:
        """Embeddings generation with minimal fakes."""

        class FakeInnerTokenizer:
            """Minimal tokenizer that supports __call__ for batch encoding."""

            def __call__(
                self,
                texts: list[str],
                return_tensors: Any = None,
                padding: bool = True,
                truncation: bool = True,
                max_length: int = 512,
            ) -> dict[str, list[list[int]]]:
                # Return simple token IDs for each text
                max_len = max(len(t.split()) for t in texts) if texts else 1
                input_ids = []
                attention_mask = []
                for t in texts:
                    toks = list(range(len(t.split())))
                    padded = toks + [0] * (max_len - len(toks))
                    mask = [1] * len(toks) + [0] * (max_len - len(toks))
                    input_ids.append(padded)
                    attention_mask.append(mask)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

        class FakeEmbeddingTokenizer:
            """Wrapper tokenizer that exposes _tokenizer."""

            def __init__(self) -> None:
                self._tokenizer = FakeInnerTokenizer()

            def encode(self, text: str, **kwargs: Any) -> list[int]:
                return list(range(len(text.split())))

        class FakeEmbeddingOutput:
            def __init__(self, embeddings: list[list[float]]) -> None:
                self.text_embeds = FakeTensor(embeddings)

        class FakeTensor:
            """Minimal tensor that supports tolist()."""

            def __init__(self, data: list[list[float]]) -> None:
                self._data = data

            def tolist(self) -> list[list[float]]:
                return self._data

        class FakeEmbeddingModel:
            def __call__(self, input_ids: Any, attention_mask: Any = None) -> FakeEmbeddingOutput:
                return FakeEmbeddingOutput([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        tokenizer = FakeEmbeddingTokenizer()
        adapter = create_adapter("embeddings", tokenizer)

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        # Mock run_on_metal_thread and mx.array/mx.eval
        fake_mx = type(
            "FakeMx",
            (),
            {
                "array": staticmethod(lambda x: FakeTensor(x) if isinstance(x, list) else x),
                "eval": staticmethod(lambda x: None),
            },
        )()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch.dict(
                "sys.modules",
                {"mlx": type("", (), {"core": fake_mx}), "mlx.core": fake_mx},
            ),
        ):
            result = await adapter.generate_embeddings(
                model=FakeEmbeddingModel(),
                texts=["hello world", "test sentence"],
            )

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert result.dimensions == 3
        assert result.total_tokens > 0
        assert result.finish_reason == "stop"


# ── generate_speech() ────────────────────────────────────────────


class TestGenerateSpeech:
    """Test generate_speech with fake TTS model."""

    @pytest.mark.anyio
    async def test_generate_speech_basic(self) -> None:
        """TTS generation with minimal fakes."""

        class FakeTensor:
            def __init__(self, data: list[float]) -> None:
                self._data = data

            def tolist(self) -> list[float]:
                return self._data

        class FakeGenResult:
            def __init__(self) -> None:
                self.audio = FakeTensor([0.0, 0.1, -0.1, 0.2])
                self.sample_rate = 24000

        class FakeTTSModel:
            sample_rate = 24000

            def generate(self, **kwargs: Any) -> list[FakeGenResult]:
                return [FakeGenResult()]

        adapter = create_adapter("kokoro", None)

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        fake_mx = type(
            "FakeMx",
            (),
            {
                "eval": staticmethod(lambda x: None),
                "concatenate": staticmethod(lambda segments, axis=0: segments[0]),
            },
        )()

        fake_np_module = type(
            "FakeNp",
            (),
            {
                "array": staticmethod(lambda x: x),
            },
        )()

        class FakeSoundfile:
            @staticmethod
            def write(buffer: Any, data: Any, sample_rate: int, format: str = "WAV") -> None:
                buffer.write(b"RIFF_fake_audio_data")

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch.dict(
                "sys.modules",
                {
                    "mlx": type("", (), {"core": fake_mx}),
                    "mlx.core": fake_mx,
                    "numpy": fake_np_module,
                    "soundfile": FakeSoundfile,
                },
            ),
        ):
            result = await adapter.generate_speech(
                model=FakeTTSModel(),
                text="Hello world",
                voice="af_heart",
                speed=1.0,
                response_format="wav",
            )

        assert isinstance(result, AudioResult)
        assert len(result.audio_bytes) > 0
        assert result.sample_rate == 24000
        assert result.format == "wav"

    @pytest.mark.anyio
    async def test_generate_speech_multiple_segments(self) -> None:
        """TTS with multiple audio segments concatenated."""

        class FakeTensor:
            def __init__(self, data: list[float]) -> None:
                self._data = data

            def tolist(self) -> list[float]:
                return self._data

        class FakeGenResult:
            def __init__(self, data: list[float]) -> None:
                self.audio = FakeTensor(data)
                self.sample_rate = 24000

        class FakeTTSModel:
            sample_rate = 24000

            def generate(self, **kwargs: Any) -> list[FakeGenResult]:
                return [
                    FakeGenResult([0.1, 0.2]),
                    FakeGenResult([0.3, 0.4]),
                ]

        adapter = create_adapter("kokoro", None)

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        # Track if concatenate was called
        concat_called = False

        class FakeConcatTensor(FakeTensor):
            pass

        def fake_concat(segments: list[Any], axis: int = 0) -> FakeConcatTensor:
            nonlocal concat_called
            concat_called = True
            return FakeConcatTensor([0.1, 0.2, 0.3, 0.4])

        fake_mx = type(
            "FakeMx",
            (),
            {
                "eval": staticmethod(lambda x: None),
                "concatenate": staticmethod(fake_concat),
            },
        )()

        fake_np_module = type(
            "FakeNp",
            (),
            {
                "array": staticmethod(lambda x: x),
            },
        )()

        class FakeSoundfile:
            @staticmethod
            def write(buffer: Any, data: Any, sample_rate: int, format: str = "WAV") -> None:
                buffer.write(b"RIFF_fake_audio")

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch.dict(
                "sys.modules",
                {
                    "mlx": type("", (), {"core": fake_mx}),
                    "mlx.core": fake_mx,
                    "numpy": fake_np_module,
                    "soundfile": FakeSoundfile,
                },
            ),
        ):
            result = await adapter.generate_speech(
                model=FakeTTSModel(),
                text="Hello world this is longer text",
            )

        assert isinstance(result, AudioResult)
        assert concat_called

    @pytest.mark.anyio
    async def test_generate_speech_no_output_raises(self) -> None:
        """TTS model producing no audio raises RuntimeError."""

        class FakeTTSModel:
            sample_rate = 24000

            def generate(self, **kwargs: Any) -> list[Any]:
                return []  # No results

        adapter = create_adapter("kokoro", None)

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        fake_mx = type(
            "FakeMx",
            (),
            {
                "eval": staticmethod(lambda x: None),
                "concatenate": staticmethod(lambda x, axis=0: x[0]),
            },
        )()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch.dict(
                "sys.modules",
                {
                    "mlx": type("", (), {"core": fake_mx}),
                    "mlx.core": fake_mx,
                },
            ),
            pytest.raises(RuntimeError, match="TTS model produced no audio output"),
        ):
            await adapter.generate_speech(
                model=FakeTTSModel(),
                text="Hello",
            )


# ── transcribe() ─────────────────────────────────────────────────


class TestTranscribe:
    """Test transcribe with fake STT model."""

    @pytest.mark.anyio
    async def test_transcribe_basic(self) -> None:
        """Basic transcription with minimal fakes."""

        class FakeSTTOutput:
            def __init__(self) -> None:
                self.text = "Hello, this is a test transcription."
                self.segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
                self.language = "en"

        adapter = create_adapter("whisper", None)

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        def fake_generate_transcription(**kwargs: Any) -> FakeSTTOutput:
            return FakeSTTOutput()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch.dict(
                "sys.modules",
                {
                    "mlx_audio": type("", (), {})(),
                    "mlx_audio.stt": type("", (), {})(),
                    "mlx_audio.stt.generate": type(
                        "",
                        (),
                        {
                            "generate_transcription": staticmethod(fake_generate_transcription),
                        },
                    )(),
                },
            ),
        ):
            result = await adapter.transcribe(
                model=object(),
                audio_data=b"fake_wav_data",
                language="en",
            )

        assert isinstance(result, TranscriptionResult)
        assert "test transcription" in result.text
        assert result.language == "en"
        assert result.segments is not None

    @pytest.mark.anyio
    async def test_transcribe_no_language(self) -> None:
        """Transcription without explicit language."""

        class FakeSTTOutput:
            def __init__(self) -> None:
                self.text = "Bonjour"
                self.language = "fr"

        adapter = create_adapter("whisper", None)

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        def fake_generate(**kwargs: Any) -> FakeSTTOutput:
            # Ensure language kwarg is NOT passed
            assert "language" not in kwargs
            return FakeSTTOutput()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch.dict(
                "sys.modules",
                {
                    "mlx_audio": type("", (), {})(),
                    "mlx_audio.stt": type("", (), {})(),
                    "mlx_audio.stt.generate": type(
                        "",
                        (),
                        {
                            "generate_transcription": staticmethod(fake_generate),
                        },
                    )(),
                },
            ),
        ):
            result = await adapter.transcribe(
                model=object(),
                audio_data=b"fake_wav_data",
            )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Bonjour"

    @pytest.mark.anyio
    async def test_transcribe_no_segments(self) -> None:
        """Transcription output without segments attribute."""

        class FakeSTTOutput:
            def __init__(self) -> None:
                self.text = "Simple output"

        adapter = create_adapter("whisper", None)

        async def run_sync(fn: Any, **kwargs: Any) -> Any:
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_sync,
            ),
            patch.dict(
                "sys.modules",
                {
                    "mlx_audio": type("", (), {})(),
                    "mlx_audio.stt": type("", (), {})(),
                    "mlx_audio.stt.generate": type(
                        "",
                        (),
                        {
                            "generate_transcription": staticmethod(lambda **kw: FakeSTTOutput()),
                        },
                    )(),
                },
            ),
        ):
            result = await adapter.transcribe(
                model=object(),
                audio_data=b"data",
            )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Simple output"
        assert result.segments is None
        assert result.language is None


# ── _inject_tools static method ──────────────────────────────────


class TestInjectTools:
    """Test _inject_tools static method directly."""

    def test_appends_to_existing_system_message(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
        ]
        result = ModelAdapter._inject_tools(messages, "TOOL PROMPT HERE")
        assert "You are helpful" in result[0]["content"]
        assert "TOOL PROMPT HERE" in result[0]["content"]
        assert len(result) == 2

    def test_creates_system_message_when_none_exists(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        result = ModelAdapter._inject_tools(messages, "TOOL PROMPT HERE")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "TOOL PROMPT HERE"
        assert len(result) == 2

    def test_does_not_mutate_original_list(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        original_len = len(messages)
        result = ModelAdapter._inject_tools(messages, "tools")
        assert len(messages) == original_len
        assert len(result) == original_len + 1
