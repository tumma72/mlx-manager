"""Unified response processor for MLX model outputs.

This module provides single-pass extraction and cleaning of model responses:
- Tool calls (Hermes/Qwen, Llama XML, GLM4 XML)
- Thinking/reasoning content
- Special token cleanup

CRITICAL: The processor extracts ALL matches in one scan and removes their spans
from the content, fixing the bug where tool call markers remained in output.
"""

import hashlib
import json
import logging
import re
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Pydantic Models ---


class ToolCallFunction(BaseModel):
    """Function call details within a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call in OpenAI-compatible format."""

    id: str
    type: str = "function"
    function: ToolCallFunction


class ParseResult(BaseModel):
    """Result of processing a model response.

    Contains extracted content, tool calls, and reasoning, with all
    markers/tags removed from the content field.
    """

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: str | None = None


# --- Parser Callback Functions ---


def parse_hermes_tool(match: re.Match[str]) -> ToolCall | None:
    """Parse Qwen/Hermes style: <tool_call>{"name": ..., "arguments": ...}</tool_call>"""
    try:
        json_str = match.group(1).strip()
        data = json.loads(json_str)

        name = data.get("name", "")
        arguments = data.get("arguments", {})

        # Arguments should be a JSON string in OpenAI format
        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments)
        else:
            arguments_str = str(arguments)

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            function=ToolCallFunction(name=name, arguments=arguments_str),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Invalid Hermes tool call: %s", e)
        return None


def parse_llama_tool(match: re.Match[str]) -> ToolCall | None:
    """Parse Llama style: <function=name>{...}</function>"""
    try:
        name = match.group(1)
        args_str = match.group(2).strip()

        # Validate JSON (or keep raw if invalid)
        try:
            json.loads(args_str)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in Llama tool call %s: %s", name, e)
            # Still include with raw arguments

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            function=ToolCallFunction(name=name, arguments=args_str),
        )
    except (IndexError, AttributeError) as e:
        logger.warning("Invalid Llama tool call: %s", e)
        return None


def parse_glm4_tool(match: re.Match[str]) -> ToolCall | None:
    """Parse GLM4 XML style: <tool_call><name>...</name><arguments>...</arguments></tool_call>"""
    try:
        name = match.group(1).strip()
        args_str = match.group(2).strip()

        # Validate JSON (or keep raw if invalid)
        try:
            json.loads(args_str)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in GLM4 tool call %s: %s", name, e)
            # Still include with raw arguments

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            function=ToolCallFunction(name=name, arguments=args_str),
        )
    except (IndexError, AttributeError) as e:
        logger.warning("Invalid GLM4 tool call: %s", e)
        return None


def parse_llama_python_tool(match: re.Match[str]) -> ToolCall | None:
    """Parse Llama Python style: <|python_tag|>module.method(args)<|eom_id|>"""
    try:
        module = match.group(1)
        method = match.group(2)
        args_str = match.group(3).strip()

        # Convert Python-style args to JSON
        args_dict = _parse_python_args(args_str)
        name = f"{module}.{method}"

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            function=ToolCallFunction(name=name, arguments=json.dumps(args_dict)),
        )
    except (IndexError, AttributeError) as e:
        logger.warning("Invalid Llama Python tool call: %s", e)
        return None


def _parse_python_args(args_str: str) -> dict[str, Any]:
    """Parse Python-style function arguments.

    Example: 'query="hello", limit=5' -> {"query": "hello", "limit": 5}
    """
    result: dict[str, Any] = {}
    if not args_str:
        return result

    # Simple parsing for key=value pairs
    pattern = re.compile(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|\d+(?:\.\d+)?|\w+)')
    for match in pattern.finditer(args_str):
        key = match.group(1)
        value_str = match.group(2)

        # Parse value
        if value_str.startswith('"') or value_str.startswith("'"):
            value: Any = value_str[1:-1]  # Remove quotes
        elif value_str.isdigit():
            value = int(value_str)
        elif "." in value_str:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
        elif value_str.lower() == "true":
            value = True
        elif value_str.lower() == "false":
            value = False
        else:
            value = value_str

        result[key] = value

    return result


# --- Response Processor ---


class ResponseProcessor:
    """Single-pass response processor with registered handlers.

    Extracts tool calls, reasoning content, and cleans special tokens
    in one scan. All matched spans are removed from the final content.
    """

    def __init__(self) -> None:
        """Initialize processor with empty pattern lists."""
        self._thinking_patterns: list[re.Pattern[str]] = []
        self._tool_patterns: list[
            tuple[re.Pattern[str], Callable[[re.Match[str]], ToolCall | None]]
        ] = []
        self._cleanup_patterns: list[str] = []

    def register_thinking_tags(self, tags: list[str]) -> None:
        """Register thinking tag names (without brackets).

        Args:
            tags: List of tag names like ["think", "thinking", "reasoning"]
        """
        for tag in tags:
            # Match <tag>content</tag> with optional whitespace and newlines
            pattern = re.compile(
                rf"<{tag}>\s*(.*?)\s*</{tag}>",
                re.DOTALL | re.IGNORECASE,
            )
            self._thinking_patterns.append(pattern)

    def register_tool_pattern(
        self,
        pattern: str,
        parser: Callable[[re.Match[str]], ToolCall | None],
    ) -> None:
        """Register tool pattern with parser callback.

        Args:
            pattern: Regex pattern string
            parser: Callback that receives match and returns ToolCall or None
        """
        compiled = re.compile(pattern, re.DOTALL)
        self._tool_patterns.append((compiled, parser))

    def register_cleanup_patterns(self, patterns: list[str]) -> None:
        """Register special tokens to remove.

        Args:
            patterns: List of literal strings to remove
        """
        self._cleanup_patterns.extend(patterns)

    def process(self, text: str) -> ParseResult:
        """Single-pass extraction and cleaning.

        Finds all matches for thinking tags and tool patterns, extracts
        their content, then removes all matched spans from the text.

        Args:
            text: Raw model output text

        Returns:
            ParseResult with cleaned content, tool calls, and reasoning
        """
        if not text:
            return ParseResult(content="")

        # Collect all matches with their spans
        # Format: (start, end, type, data)
        matches: list[tuple[int, int, str, Any]] = []

        # Find thinking content
        reasoning_parts: list[str] = []
        for pattern in self._thinking_patterns:
            for match in pattern.finditer(text):
                content = match.group(1).strip()
                # Always record span for removal, even if empty
                matches.append((match.start(), match.end(), "thinking", content))
                # Only add to reasoning if there's actual content
                if content:
                    reasoning_parts.append(content)

        # Find tool calls
        tool_calls: list[ToolCall] = []
        seen_hashes: set[str] = set()  # For GLM4 deduplication

        for pattern, parser in self._tool_patterns:
            for match in pattern.finditer(text):
                tool_call = parser(match)
                if tool_call:
                    # Deduplicate by (name, arguments) hash
                    content_hash = hashlib.md5(
                        f"{tool_call.function.name}:{tool_call.function.arguments}".encode()
                    ).hexdigest()
                    if content_hash not in seen_hashes:
                        seen_hashes.add(content_hash)
                        tool_calls.append(tool_call)
                    else:
                        logger.debug(
                            "Skipping duplicate tool call: %s",
                            tool_call.function.name,
                        )

                # Always record span for removal (even if duplicate)
                matches.append((match.start(), match.end(), "tool", None))

        # Merge overlapping spans and remove in reverse order
        spans_to_remove = self._merge_overlapping_spans(
            [(start, end) for start, end, _, _ in matches]
        )

        # Remove spans from text (reverse order to preserve indices)
        cleaned = text
        for start, end in sorted(spans_to_remove, reverse=True):
            cleaned = cleaned[:start] + cleaned[end:]

        # Remove cleanup patterns (special tokens)
        for token in self._cleanup_patterns:
            cleaned = cleaned.replace(token, "")

        # Normalize whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

        # Combine reasoning parts
        reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None

        return ParseResult(
            content=cleaned,
            tool_calls=tool_calls,
            reasoning=reasoning,
        )

    def _merge_overlapping_spans(
        self, spans: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Merge overlapping or adjacent spans.

        Args:
            spans: List of (start, end) tuples

        Returns:
            List of merged (start, end) tuples
        """
        if not spans:
            return []

        # Sort by start position
        sorted_spans = sorted(spans, key=lambda x: x[0])

        merged: list[tuple[int, int]] = []
        current_start, current_end = sorted_spans[0]

        for start, end in sorted_spans[1:]:
            if start <= current_end:
                # Overlapping or adjacent - extend current span
                current_end = max(current_end, end)
            else:
                # Gap - save current and start new
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        # Don't forget the last span
        merged.append((current_start, current_end))

        return merged


# --- Factory and Singleton ---


def create_default_processor() -> ResponseProcessor:
    """Create processor with all standard handlers registered.

    Returns:
        ResponseProcessor with thinking, tool, and cleanup patterns configured
    """
    processor = ResponseProcessor()

    # Thinking tags
    processor.register_thinking_tags(["think", "thinking", "reasoning", "reflection"])

    # Tool patterns with parsers
    # Hermes/Qwen JSON format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    processor.register_tool_pattern(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        parse_hermes_tool,
    )

    # Llama XML format: <function=name>{...}</function>
    processor.register_tool_pattern(
        r"<function=(\w+)>(.*?)</function>",
        parse_llama_tool,
    )

    # GLM4 XML format: <tool_call><name>...</name><arguments>...</arguments></tool_call>
    processor.register_tool_pattern(
        r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
        parse_glm4_tool,
    )

    # Llama Python format: <|python_tag|>module.method(args)<|eom_id|>
    processor.register_tool_pattern(
        r"<\|python_tag\|>\s*(\w+)\.(\w+)\((.*?)\)\s*<\|eom_id\|>",
        parse_llama_python_tool,
    )

    # Cleanup patterns (special tokens)
    processor.register_cleanup_patterns(
        [
            "<|endoftext|>",
            "<|im_end|>",
            "<|im_start|>",
            "<|eot_id|>",
            "<|end|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ]
    )

    return processor


# Singleton instance
_processor: ResponseProcessor | None = None


def get_response_processor() -> ResponseProcessor:
    """Get the singleton ResponseProcessor instance.

    Returns:
        Shared ResponseProcessor with all patterns registered
    """
    global _processor
    if _processor is None:
        _processor = create_default_processor()
    return _processor


def reset_response_processor() -> None:
    """Reset the singleton instance (for testing)."""
    global _processor
    _processor = None
