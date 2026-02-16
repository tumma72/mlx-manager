"""Shared native tool detection utility.

Provides a centralized, cached check for whether a tokenizer's Jinja
template natively accepts a ``tools=`` parameter.  This replaces ad-hoc
per-adapter detection (e.g. the GLM4 ``_native_tools_cache``) with a
single source of truth.

Detection is two-phase:
1. Fast filter – does the raw template string mention ``"tools"``?
2. Confirmation – trial ``apply_chat_template(..., tools=[...])`` call.

Results are cached by ``id(tokenizer)`` so each tokenizer is probed at
most once per process lifetime.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Cache: tokenizer id -> supports native tools
_native_tools_cache: dict[int, bool] = {}

# Minimal tool definition used for the trial call
_TRIAL_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "_probe",
            "description": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

_TRIAL_MESSAGES = [{"role": "user", "content": "hi"}]


def has_native_tool_support(tokenizer: Any) -> bool:
    """Check whether *tokenizer* natively supports ``tools=`` in its template.

    The result is cached per tokenizer object identity so repeated calls
    are essentially free.

    Args:
        tokenizer: A HuggingFace tokenizer (or Processor wrapping one).

    Returns:
        ``True`` if ``apply_chat_template(messages, tools=..., tokenize=False)``
        succeeds without error, ``False`` otherwise.
    """
    actual = getattr(tokenizer, "tokenizer", tokenizer)
    tok_id = id(actual)

    cached = _native_tools_cache.get(tok_id)
    if cached is not None:
        return cached

    # Phase 1: fast string check
    template: str | None = getattr(actual, "chat_template", None)
    if template is None or "tools" not in template:
        _native_tools_cache[tok_id] = False
        return False

    # Phase 2: trial call
    try:
        actual.apply_chat_template(
            _TRIAL_MESSAGES,
            tools=_TRIAL_TOOL,
            add_generation_prompt=True,
            tokenize=False,
        )
        _native_tools_cache[tok_id] = True
        logger.debug("Native tool support detected for tokenizer {}", type(actual).__name__)
        return True
    except (TypeError, Exception) as exc:
        if isinstance(exc, TypeError) and "tools" in str(exc):
            logger.debug(
                "Tokenizer {} doesn't accept tools= parameter",
                type(actual).__name__,
            )
        else:
            logger.debug(
                "Tokenizer {} trial tools call failed: {}",
                type(actual).__name__,
                exc,
            )
        _native_tools_cache[tok_id] = False
        return False


def clear_native_tools_cache() -> None:
    """Clear the detection cache.  Useful in tests."""
    _native_tools_cache.clear()


# Cache: tokenizer id -> supports thinking
_thinking_cache: dict[int, bool] = {}

_THINKING_MESSAGES = [{"role": "user", "content": "hi"}]


def has_thinking_support(tokenizer: Any) -> bool:
    """Check whether *tokenizer* supports ``enable_thinking=True`` in its template.

    The result is cached per tokenizer object identity so repeated calls
    are essentially free.

    Args:
        tokenizer: A HuggingFace tokenizer (or Processor wrapping one).

    Returns:
        ``True`` if ``apply_chat_template(messages, enable_thinking=True, tokenize=False)``
        succeeds without error, ``False`` otherwise.
    """
    actual = getattr(tokenizer, "tokenizer", tokenizer)
    tok_id = id(actual)

    cached = _thinking_cache.get(tok_id)
    if cached is not None:
        return cached

    # Phase 1: fast string check
    template: str | None = getattr(actual, "chat_template", None)
    if template is None or "enable_thinking" not in template:
        _thinking_cache[tok_id] = False
        return False

    # Phase 2: trial call
    try:
        actual.apply_chat_template(
            _THINKING_MESSAGES,
            enable_thinking=True,
            add_generation_prompt=True,
            tokenize=False,
        )
        _thinking_cache[tok_id] = True
        logger.debug("Thinking support detected for tokenizer {}", type(actual).__name__)
        return True
    except (TypeError, Exception) as exc:
        logger.debug(
            "Tokenizer {} doesn't support enable_thinking: {}",
            type(actual).__name__,
            exc,
        )
        _thinking_cache[tok_id] = False
        return False


def clear_thinking_cache() -> None:
    """Clear the thinking detection cache. Useful in tests."""
    _thinking_cache.clear()
