"""Parser base classes for tool call and thinking extraction."""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from mlx_manager.mlx_server.schemas.openai import FunctionCall, ToolCall


class ToolCallParser(ABC):
    """Extracts tool calls from model output.

    Each implementation handles one specific output format.
    Used in both streaming (marker detection) and batch (full extraction).

    Subclasses MUST implement: parser_id, stream_markers, extract().
    Subclasses SHOULD use the concrete helpers: _make_tool_call(),
    _coerce_arguments(), _deduplicate().
    """

    @property
    @abstractmethod
    def parser_id(self) -> str:
        """Unique string identifier for DB storage (e.g., 'hermes_json')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def stream_markers(self) -> list[tuple[str, str]]:
        """(start_marker, end_marker) pairs for streaming detection."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, text: str) -> list[ToolCall]:
        """Extract all tool calls from complete text (batch mode)."""
        raise NotImplementedError

    def validates(self, text: str, expected_fn: str) -> bool:
        """Check if output contains a valid call to expected_fn.
        Used by probe. Delegates to extract() — same code path as inference."""
        return any(tc.function.name == expected_fn for tc in self.extract(text))

    # ── Concrete helpers for subclasses ───────────────────────────

    @staticmethod
    def _make_tool_call(name: str, arguments: str, *, call_id: str | None = None) -> ToolCall:
        """Construct a ToolCall with standardized ID generation."""
        return ToolCall(
            id=call_id or f"call_{uuid.uuid4().hex[:8]}",
            function=FunctionCall(name=name, arguments=arguments),
        )

    @staticmethod
    def _coerce_arguments(arguments: Any) -> str:
        """Normalize arguments to a JSON string.
        Dicts are serialized; other types coerced to str."""
        if isinstance(arguments, dict):
            return json.dumps(arguments)
        return str(arguments)

    @staticmethod
    def _deduplicate(calls: list[ToolCall]) -> list[ToolCall]:
        """Remove duplicate tool calls by (name, arguments) identity."""
        seen: set[str] = set()
        results: list[ToolCall] = []
        for tc in calls:
            key = f"{tc.function.name}:{tc.function.arguments}"
            if key not in seen:
                seen.add(key)
                results.append(tc)
        return results


class ThinkingParser(ABC):
    """Extracts thinking/reasoning blocks from model output.

    Subclasses MUST implement: parser_id, stream_markers, extract(), remove().
    Subclasses SHOULD use the concrete helpers: _normalize_whitespace(),
    _collect_matches(), _remove_patterns().
    """

    @property
    @abstractmethod
    def parser_id(self) -> str:
        """Unique string identifier for DB storage (e.g., 'think_tag')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def stream_markers(self) -> list[tuple[str, str]]:
        """(start_marker, end_marker) pairs for streaming detection."""
        raise NotImplementedError

    @property
    def supports_toggle(self) -> bool:
        """Whether this parser supports enable_thinking parameter in templates.
        Override in subclasses for models that support toggling."""
        return False

    @abstractmethod
    def extract(self, text: str) -> str | None:
        """Extract thinking content from text. Returns None if no thinking."""
        raise NotImplementedError

    @abstractmethod
    def remove(self, text: str) -> str:
        """Remove all thinking blocks from text, return cleaned content."""
        raise NotImplementedError

    # ── Concrete helpers for subclasses ───────────────────────────

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse 3+ consecutive newlines to 2, then strip."""
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    @staticmethod
    def _collect_matches(text: str, patterns: Sequence[re.Pattern[str]]) -> str | None:
        """Accumulate group(1) from all pattern matches, join with double newline.
        Returns None if no non-empty matches found."""
        parts: list[str] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                content = match.group(1).strip()
                if content:
                    parts.append(content)
        return "\n\n".join(parts) if parts else None

    @classmethod
    def _remove_patterns(cls, text: str, patterns: Sequence[re.Pattern[str]]) -> str:
        """Remove all pattern matches from text, then normalize whitespace."""
        result = text
        for pattern in patterns:
            result = pattern.sub("", result)
        return cls._normalize_whitespace(result)
