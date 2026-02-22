"""Concrete thinking/reasoning parser implementations."""

from __future__ import annotations

import re

from mlx_manager.mlx_server.parsers.base import ThinkingParser


class ThinkTagParser(ThinkingParser):
    """Extracts thinking blocks wrapped in <think>...</think> tags.

    Used by Qwen3, GLM-4.7, and other models that support reasoning mode.
    Also matches <thinking>, <reasoning>, <reflection> variants.
    """

    _PATTERNS = [
        re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
        for tag in ("think", "thinking", "reasoning", "reflection")
    ]

    @property
    def parser_id(self) -> str:
        return "think_tag"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [
            ("<think>", "</think>"),
            ("<thinking>", "</thinking>"),
            ("<reasoning>", "</reasoning>"),
            ("<reflection>", "</reflection>"),
        ]

    @property
    def supports_toggle(self) -> bool:
        return True

    def extract(self, text: str) -> str | None:
        return self._collect_matches(text, self._PATTERNS)

    def remove(self, text: str) -> str:
        return self._remove_patterns(text, self._PATTERNS)


class MistralThinkingParser(ThinkingParser):
    """Extracts thinking blocks wrapped in [THINK]...[/THINK] bracket tokens.

    Used by Mistral v3, Devstral, and other Mistral-family models that use
    bracket-style control tokens for reasoning.
    """

    _PATTERN = re.compile(r"\[THINK\]\s*(.*?)\s*\[/THINK\]", re.DOTALL | re.IGNORECASE)

    @property
    def parser_id(self) -> str:
        return "mistral_think"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("[THINK]", "[/THINK]")]

    def extract(self, text: str) -> str | None:
        return self._collect_matches(text, [self._PATTERN])

    def remove(self, text: str) -> str:
        return self._remove_patterns(text, [self._PATTERN])


class NullThinkingParser(ThinkingParser):
    """No-op parser for models without thinking support."""

    @property
    def parser_id(self) -> str:
        return "null"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return []

    def extract(self, text: str) -> str | None:
        return None

    def remove(self, text: str) -> str:
        return text
