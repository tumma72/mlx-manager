"""Tests for the shared native tool and thinking detection utilities."""

from unittest.mock import MagicMock

import pytest

from mlx_manager.mlx_server.utils.template_tools import (
    clear_native_tools_cache,
    clear_thinking_cache,
    has_native_tool_support,
    has_thinking_support,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear detection caches before each test."""
    clear_native_tools_cache()
    clear_thinking_cache()
    yield
    clear_native_tools_cache()
    clear_thinking_cache()


def _make_tokenizer(
    *,
    template: str | None = None,
    accept_tools: bool = False,
) -> MagicMock:
    """Build a mock tokenizer for testing.

    Args:
        template: Value for ``chat_template``. None means no template.
        accept_tools: If True, ``apply_chat_template`` accepts ``tools=``.
                      If False, it raises TypeError when ``tools=`` is passed.
    """
    tok = MagicMock(spec=["apply_chat_template", "chat_template"])
    tok.chat_template = template

    if accept_tools:
        tok.apply_chat_template.return_value = "<formatted with tools>"
    else:

        def _side_effect(*args, **kwargs):
            if "tools" in kwargs:
                raise TypeError("unexpected keyword argument 'tools'")
            return "<formatted>"

        tok.apply_chat_template.side_effect = _side_effect

    return tok


class TestHasNativeToolSupport:
    """Tests for has_native_tool_support()."""

    def test_no_template_returns_false(self) -> None:
        tok = _make_tokenizer(template=None)
        assert has_native_tool_support(tok) is False

    def test_template_without_tools_keyword_returns_false(self) -> None:
        tok = _make_tokenizer(template="Hello {{ messages }}")
        assert has_native_tool_support(tok) is False

    def test_template_with_tools_and_acceptance_returns_true(self) -> None:
        tok = _make_tokenizer(
            template="{% if tools %}{{ tools }}{% endif %} {{ messages }}",
            accept_tools=True,
        )
        assert has_native_tool_support(tok) is True

    def test_template_with_tools_keyword_but_rejection_returns_false(self) -> None:
        tok = _make_tokenizer(
            template="{% if tools %}{{ tools }}{% endif %} {{ messages }}",
            accept_tools=False,
        )
        assert has_native_tool_support(tok) is False

    def test_result_is_cached(self) -> None:
        tok = _make_tokenizer(
            template="{% if tools %}{{ tools }}{% endif %}",
            accept_tools=True,
        )
        assert has_native_tool_support(tok) is True

        # Change behaviour — but cache should return previous result
        tok.apply_chat_template.side_effect = TypeError("nope")
        assert has_native_tool_support(tok) is True

    def test_processor_wrapper_is_unwrapped(self) -> None:
        """Processor objects wrap the real tokenizer in .tokenizer attribute."""
        inner = _make_tokenizer(
            template="tools are here",
            accept_tools=True,
        )
        processor = MagicMock()
        processor.tokenizer = inner

        assert has_native_tool_support(processor) is True

    def test_clear_cache_resets_detection(self) -> None:
        tok = _make_tokenizer(
            template="tools present",
            accept_tools=True,
        )
        assert has_native_tool_support(tok) is True
        clear_native_tools_cache()

        # Now make it reject — should re-evaluate
        tok.apply_chat_template.side_effect = TypeError("tools")
        assert has_native_tool_support(tok) is False

    def test_generic_exception_returns_false(self) -> None:
        """Non-TypeError exceptions during trial call should return False."""
        tok = _make_tokenizer(
            template="tools keyword present",
        )
        tok.apply_chat_template.side_effect = RuntimeError("template broken")
        assert has_native_tool_support(tok) is False


def _make_thinking_tokenizer(
    *,
    template: str | None = None,
    accept_thinking: bool = False,
) -> MagicMock:
    """Build a mock tokenizer for thinking support testing."""
    tok = MagicMock(spec=["apply_chat_template", "chat_template"])
    tok.chat_template = template

    if accept_thinking:
        tok.apply_chat_template.return_value = "<formatted with thinking>"
    else:

        def _side_effect(*args, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            return "<formatted>"

        tok.apply_chat_template.side_effect = _side_effect

    return tok


class TestHasThinkingSupport:
    """Tests for has_thinking_support()."""

    def test_no_template_returns_false(self) -> None:
        tok = _make_thinking_tokenizer(template=None)
        assert has_thinking_support(tok) is False

    def test_template_without_thinking_keyword_returns_false(self) -> None:
        tok = _make_thinking_tokenizer(template="Hello {{ messages }}")
        assert has_thinking_support(tok) is False

    def test_template_with_thinking_keyword_only_returns_false(self) -> None:
        """Generic 'thinking' keyword without 'enable_thinking' is not enough."""
        tok = _make_thinking_tokenizer(
            template="{% if thinking %}think{% endif %} {{ messages }}",
            accept_thinking=False,  # Won't be called since Phase 1 fails
        )
        assert has_thinking_support(tok) is False

    def test_template_with_enable_thinking_keyword_and_acceptance_returns_true(self) -> None:
        """Qwen-style template with enable_thinking parameter."""
        tok = _make_thinking_tokenizer(
            template="{% if enable_thinking %}think{% endif %}",
            accept_thinking=True,
        )
        assert has_thinking_support(tok) is True

    def test_template_with_keep_past_thinking_only_returns_false(self) -> None:
        """Liquid-style keep_past_thinking without enable_thinking is not supported."""
        tok = _make_thinking_tokenizer(
            template="{% if keep_past_thinking %}{{ thinking }}{% endif %}",
            accept_thinking=False,  # Won't be called since Phase 1 fails
        )
        assert has_thinking_support(tok) is False

    def test_template_with_enable_thinking_keyword_but_rejection_returns_false(self) -> None:
        """Template has enable_thinking but tokenizer rejects the parameter."""
        tok = _make_thinking_tokenizer(
            template="{% if enable_thinking %}think{% endif %} {{ messages }}",
            accept_thinking=False,
        )
        assert has_thinking_support(tok) is False

    def test_result_is_cached(self) -> None:
        tok = _make_thinking_tokenizer(
            template="enable_thinking support here",
            accept_thinking=True,
        )
        assert has_thinking_support(tok) is True

        # Change behaviour — but cache should return previous result
        tok.apply_chat_template.side_effect = TypeError("nope")
        assert has_thinking_support(tok) is True

    def test_processor_wrapper_is_unwrapped(self) -> None:
        """Processor objects wrap the real tokenizer in .tokenizer attribute."""
        inner = _make_thinking_tokenizer(
            template="enable_thinking is here",
            accept_thinking=True,
        )
        processor = MagicMock()
        processor.tokenizer = inner

        assert has_thinking_support(processor) is True

    def test_clear_cache_resets_detection(self) -> None:
        tok = _make_thinking_tokenizer(
            template="enable_thinking present",
            accept_thinking=True,
        )
        assert has_thinking_support(tok) is True
        clear_thinking_cache()

        # Now make it reject — should re-evaluate
        tok.apply_chat_template.side_effect = TypeError("enable_thinking")
        assert has_thinking_support(tok) is False

    def test_generic_exception_returns_false(self) -> None:
        """Non-TypeError exceptions during trial call should return False."""
        tok = _make_thinking_tokenizer(
            template="enable_thinking keyword present",
        )
        tok.apply_chat_template.side_effect = RuntimeError("template broken")
        assert has_thinking_support(tok) is False
