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
        parts: list[str] = []
        for pattern in self._PATTERNS:
            for match in pattern.finditer(text):
                content = match.group(1).strip()
                if content:
                    parts.append(content)
        return "\n\n".join(parts) if parts else None

    def remove(self, text: str) -> str:
        result = text
        for pattern in self._PATTERNS:
            result = pattern.sub("", result)
        # Normalize whitespace
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()


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
