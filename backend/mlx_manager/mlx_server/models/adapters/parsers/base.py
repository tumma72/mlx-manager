"""Base tool call parser protocol."""

import re
from abc import ABC, abstractmethod
from typing import Any

# Common special tokens across model families
COMMON_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|end|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
]


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

    def clean_response(self, text: str) -> str:
        """Remove tool calls and special tokens from response text.

        Override in subclasses for model-specific cleaning.
        Default implementation removes common special tokens.

        Args:
            text: Raw model output

        Returns:
            Cleaned text suitable for display
        """
        cleaned = text

        # Remove common special tokens
        for token in COMMON_SPECIAL_TOKENS:
            cleaned = cleaned.replace(token, "")

        # Clean up excessive whitespace from removals
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

        return cleaned
