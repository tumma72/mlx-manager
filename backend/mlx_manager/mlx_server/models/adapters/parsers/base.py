"""Base tool call parser protocol."""

from abc import ABC, abstractmethod
from typing import Any


class ToolCallParser(ABC):
    """Abstract base class for model-specific tool call parsers.

    Each model family outputs tool calls in a different format.
    Parsers normalize these to OpenAI-compatible format:
    [{"id": str, "type": "function", "function": {"name": str, "arguments": str}}]
    """

    @abstractmethod
    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls from model output text.

        Args:
            text: Model output text that may contain tool calls

        Returns:
            List of tool call dicts in OpenAI format:
            [{"id": str, "type": "function", "function": {"name": str, "arguments": str}}]
            Returns empty list if no tool calls found.
        """
        ...

    @abstractmethod
    def format_tools(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions for inclusion in prompt.

        Args:
            tools: List of tool definitions in OpenAI format, e.g.:
                [{"type": "function", "function": {"name": ..., "parameters": ...}}]

        Returns:
            Formatted string to include in system prompt or tool section
        """
        ...
