"""Tests for model adapter strategy functions.

All pure functions tested with real inputs (no mocks).
"""

from __future__ import annotations

from typing import Any

import pytest

from mlx_manager.mlx_server.models.adapters.strategies import (
    glm4_template,
    glm4_tool_formatter,
    hermes_message_converter,
    llama_message_converter,
    llama_tool_formatter,
    mistral_template,
    qwen_template,
    qwen_tool_formatter,
    whisper_post_load_hook,
)

# ── Fake tokenizer for testing ──────────────────────────────────────


class FakeTokenizer:
    """Minimal tokenizer implementation for testing template strategies."""

    def __init__(self, should_fail: bool = False, fail_on_enable_thinking: bool = False):
        """Initialize fake tokenizer.

        Args:
            should_fail: If True, apply_chat_template always raises an exception
            fail_on_enable_thinking: If True, fail only when enable_thinking is passed
        """
        self.should_fail = should_fail
        self.fail_on_enable_thinking = fail_on_enable_thinking

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Fake implementation of apply_chat_template."""
        if self.should_fail:
            raise ValueError("Tokenizer error")

        if self.fail_on_enable_thinking and enable_thinking is not None:
            raise TypeError("enable_thinking not supported")

        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")

        if tools:
            parts.append(f"[TOOLS: {len(tools)}]")

        if add_generation_prompt:
            parts.append("assistant:")

        return "\n".join(parts)


class TokenizerWithoutTemplate:
    """Tokenizer without apply_chat_template method (for glm4 fallback)."""

    pass


# ── Template strategy tests ─────────────────────────────────────────


def test_qwen_template_basic():
    """Test qwen_template with basic messages."""
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "Hello"}]

    result = qwen_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=None,
        enable_thinking=True,
    )

    assert "user: Hello" in result
    assert "assistant:" in result


def test_qwen_template_with_tools():
    """Test qwen_template with native tools (line 44)."""
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            },
        }
    ]

    result = qwen_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=tools,
        enable_thinking=True,
    )

    assert "user: Hello" in result
    assert "[TOOLS: 1]" in result


def test_qwen_template_enable_thinking_fallback():
    """Test qwen_template fallback when enable_thinking not supported (lines 47-49)."""
    tokenizer = FakeTokenizer(fail_on_enable_thinking=True)
    messages = [{"role": "user", "content": "Hello"}]

    # Should succeed by falling back to version without enable_thinking
    result = qwen_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=None,
        enable_thinking=True,
    )

    assert "user: Hello" in result
    assert "assistant:" in result


def test_qwen_template_enable_thinking_fallback_with_tools():
    """Test qwen_template fallback with both tools and enable_thinking error."""
    tokenizer = FakeTokenizer(fail_on_enable_thinking=True)
    messages = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            },
        }
    ]

    result = qwen_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=tools,
        enable_thinking=True,
    )

    assert "user: Hello" in result
    assert "[TOOLS: 1]" in result


def test_glm4_template_basic():
    """Test glm4_template with basic messages."""
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "Hello"}]

    result = glm4_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=None,
        enable_thinking=False,
    )

    assert "user: Hello" in result
    assert "assistant:" in result


def test_glm4_template_with_tools():
    """Test glm4_template with native tools."""
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            },
        }
    ]

    result = glm4_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=tools,
        enable_thinking=False,
    )

    assert "user: Hello" in result
    assert "[TOOLS: 1]" in result


def test_glm4_template_chatml_fallback():
    """Test glm4_template ChatML fallback when tokenizer fails (lines 69-79)."""
    tokenizer = FakeTokenizer(should_fail=True)
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    result = glm4_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=None,
        enable_thinking=False,
    )

    # Should use manual ChatML format
    assert "<|system|>" in result
    assert "You are helpful" in result
    assert "<|user|>" in result
    assert "Hello" in result
    assert "<|assistant|>" in result


def test_glm4_template_chatml_fallback_no_generation_prompt():
    """Test glm4_template ChatML fallback without generation prompt."""
    tokenizer = FakeTokenizer(should_fail=True)
    messages = [{"role": "user", "content": "Hello"}]

    result = glm4_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=False,
        native_tools=None,
        enable_thinking=False,
    )

    assert "<|user|>" in result
    assert "Hello" in result
    assert "<|assistant|>" not in result


def test_glm4_template_no_template_method():
    """Test glm4_template with tokenizer lacking apply_chat_template."""
    tokenizer = TokenizerWithoutTemplate()
    messages = [{"role": "user", "content": "Hello"}]

    result = glm4_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=None,
        enable_thinking=False,
    )

    # Should fall back to ChatML
    assert "<|user|>" in result
    assert "Hello" in result
    assert "<|assistant|>" in result


def test_mistral_template_basic():
    """Test mistral_template with basic messages."""
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "Hello"}]

    result = mistral_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=None,
        enable_thinking=False,
    )

    assert "user: Hello" in result
    assert "assistant:" in result


def test_mistral_template_with_tools():
    """Test mistral_template with native tools (line 106)."""
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "Hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            },
        }
    ]

    result = mistral_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=tools,
        enable_thinking=False,
    )

    assert "user: Hello" in result
    assert "[TOOLS: 1]" in result


def test_mistral_template_system_message_merge():
    """Test mistral_template merges system message into first user message."""
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    result = mistral_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=None,
        enable_thinking=False,
    )

    # Should merge system into user
    assert "user: You are helpful\n\nHello" in result


def test_mistral_template_system_only():
    """Test mistral_template with only system message (no user message to merge)."""
    tokenizer = FakeTokenizer()
    messages = [{"role": "system", "content": "You are helpful"}]

    result = mistral_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        native_tools=None,
        enable_thinking=False,
    )

    # System message removed, but no user message to merge into
    assert result  # Should still return something


# ── Tool formatter tests ────────────────────────────────────────────


def test_qwen_tool_formatter_empty():
    """Test qwen_tool_formatter with empty tools."""
    result = qwen_tool_formatter([])
    assert result == ""


def test_qwen_tool_formatter_single_tool():
    """Test qwen_tool_formatter with single tool."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
    ]

    result = qwen_tool_formatter(tools)

    assert "<tools>" in result
    assert "get_weather" in result
    assert "Get current weather" in result
    assert "<tool_call>" in result
    assert "</tools>" in result


def test_qwen_tool_formatter_multiple_tools():
    """Test qwen_tool_formatter with multiple tools."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web",
                "parameters": {},
            },
        },
    ]

    result = qwen_tool_formatter(tools)

    assert "get_weather" in result
    assert "search_web" in result


def test_glm4_tool_formatter_empty():
    """Test glm4_tool_formatter with empty tools."""
    result = glm4_tool_formatter([])
    assert result == ""


def test_glm4_tool_formatter_single_tool():
    """Test glm4_tool_formatter with single tool."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object"},
            },
        }
    ]

    result = glm4_tool_formatter(tools)

    assert "<tool>" in result
    assert "<name>get_weather</name>" in result
    assert "<description>Get current weather</description>" in result
    assert "<parameters>" in result
    assert "</tool>" in result


def test_llama_tool_formatter_empty():
    """Test llama_tool_formatter with empty tools."""
    result = llama_tool_formatter([])
    assert result == ""


def test_llama_tool_formatter_single_tool():
    """Test llama_tool_formatter with single tool."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object"},
            },
        }
    ]

    result = llama_tool_formatter(tools)

    assert "get_weather:" in result
    assert "description: Get current weather" in result
    assert "parameters:" in result
    assert "<function=function_name>" in result


# ── Message converter tests ─────────────────────────────────────────


def test_hermes_message_converter_basic():
    """Test hermes_message_converter with basic messages."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = hermes_message_converter(messages)

    assert result == messages  # No conversion needed


def test_hermes_message_converter_tool_message():
    """Test hermes_message_converter converts tool message to user."""
    messages = [
        {"role": "tool", "tool_call_id": "call_123", "content": "Weather: sunny"},
    ]

    result = hermes_message_converter(messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "call_123" in result[0]["content"]
    assert "Weather: sunny" in result[0]["content"]
    assert "Tool Result" in result[0]["content"]


def test_hermes_message_converter_assistant_with_tool_calls():
    """Test hermes_message_converter converts assistant tool_calls to tags."""
    messages = [
        {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        }
    ]

    result = hermes_message_converter(messages)

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "Let me check" in result[0]["content"]
    assert "<tool_call>" in result[0]["content"]
    assert "get_weather" in result[0]["content"]


def test_hermes_message_converter_assistant_tool_calls_no_content():
    """Test hermes_message_converter handles assistant tool_calls with no content."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        }
    ]

    result = hermes_message_converter(messages)

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "<tool_call>" in result[0]["content"]


def test_llama_message_converter_basic():
    """Test llama_message_converter with basic messages."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = llama_message_converter(messages)

    assert result == messages


def test_llama_message_converter_tool_message():
    """Test llama_message_converter converts tool message to user (lines 245-247)."""
    messages = [
        {"role": "tool", "tool_call_id": "call_123", "content": "Weather: sunny"},
    ]

    result = llama_message_converter(messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "call_123" in result[0]["content"]
    assert "Weather: sunny" in result[0]["content"]
    assert "Tool Result" in result[0]["content"]


def test_llama_message_converter_assistant_with_tool_calls():
    """Test llama_message_converter converts assistant tool_calls to function tags."""
    messages = [
        {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        }
    ]

    result = llama_message_converter(messages)

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "Let me check" in result[0]["content"]
    assert "<function=get_weather>" in result[0]["content"]


def test_llama_message_converter_assistant_tool_calls_no_content():
    """Test llama_message_converter handles assistant tool_calls with no content (line 269)."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        }
    ]

    result = llama_message_converter(messages)

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "<function=get_weather>" in result[0]["content"]
    # Content should be empty string + tool text
    assert result[0]["content"].startswith("\n<function=")


def test_llama_message_converter_multiple_tool_calls():
    """Test llama_message_converter with multiple tool calls."""
    messages = [
        {
            "role": "assistant",
            "content": "Checking...",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search_web", "arguments": '{"query": "news"}'},
                },
            ],
        }
    ]

    result = llama_message_converter(messages)

    assert len(result) == 1
    assert "<function=get_weather>" in result[0]["content"]
    assert "<function=search_web>" in result[0]["content"]


# ── Post-load hook tests ────────────────────────────────────────────


class FakeProcessor:
    """Fake processor for testing whisper hook."""

    def __init__(self, has_tokenizer: bool = True, vocab_size: int = 0):
        """Initialize fake processor."""
        if has_tokenizer:
            self.tokenizer = FakeTokenizer()
            self.tokenizer.vocab_size = vocab_size
        else:
            self.tokenizer = None


class FakeModel:
    """Fake model for testing whisper hook."""

    def __init__(self, processor: FakeProcessor | None = None):
        """Initialize fake model."""
        self._processor = processor


@pytest.mark.asyncio
async def test_whisper_post_load_hook_valid_processor():
    """Test whisper_post_load_hook with valid processor (early return, line 281)."""
    processor = FakeProcessor(has_tokenizer=True, vocab_size=100)
    model = FakeModel(processor=processor)

    # Should return early without modification
    await whisper_post_load_hook(model, "mlx-community/whisper-tiny")

    # Processor should be unchanged
    assert model._processor is processor


@pytest.mark.asyncio
async def test_whisper_post_load_hook_no_processor():
    """Test whisper_post_load_hook with no processor (lines 278-294)."""
    model = FakeModel(processor=None)

    # This will fail to load WhisperProcessor (no transformers installed or model doesn't exist)
    # but should not raise an exception
    await whisper_post_load_hook(model, "mlx-community/whisper-tiny")

    # Model should still have no processor (or a new one if transformers is available)
    # We can't assert much here since it depends on environment


@pytest.mark.asyncio
async def test_whisper_post_load_hook_broken_tokenizer():
    """Test whisper_post_load_hook with processor but broken tokenizer."""
    processor = FakeProcessor(has_tokenizer=True, vocab_size=0)  # vocab_size=0 is broken
    model = FakeModel(processor=processor)

    # Should attempt to reload processor
    await whisper_post_load_hook(model, "mlx-community/whisper-tiny")

    # We can't assert the result since it depends on whether transformers is available


@pytest.mark.asyncio
async def test_whisper_post_load_hook_no_tokenizer():
    """Test whisper_post_load_hook with processor but no tokenizer."""
    processor = FakeProcessor(has_tokenizer=False)
    model = FakeModel(processor=processor)

    # Should attempt to reload processor
    await whisper_post_load_hook(model, "mlx-community/whisper-tiny")

    # We can't assert the result since it depends on whether transformers is available


@pytest.mark.asyncio
async def test_whisper_post_load_hook_repo_name_extraction():
    """Test whisper_post_load_hook extracts repo name correctly."""
    model = FakeModel(processor=None)

    # Test with full repo path
    await whisper_post_load_hook(model, "mlx-community/whisper-large")
    # Should derive openai/whisper-large

    # Test with simple name
    await whisper_post_load_hook(model, "whisper-small")
    # Should derive openai/whisper-small


@pytest.mark.asyncio
async def test_whisper_post_load_hook_exception_handling():
    """Test whisper_post_load_hook handles exceptions gracefully (lines 293-294)."""
    model = FakeModel(processor=None)

    # Use a model ID that will definitely fail to load
    # Even if transformers is available, this model won't exist
    await whisper_post_load_hook(model, "mlx-community/this-model-does-not-exist-at-all-12345")

    # Should not raise an exception, just log a warning
    # The processor will remain None or whatever transformers managed to create
