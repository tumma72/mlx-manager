"""Unified response processor for MLX model outputs.

This module provides single-pass extraction and cleaning of model responses:
- Tool calls (Hermes/Qwen, Llama XML, GLM4 XML)
- Thinking/reasoning content
- Special token cleanup

CRITICAL: The processor extracts ALL matches in one scan and removes their spans
from the content, fixing the bug where tool call markers remained in output.

StreamingProcessor returns OpenAI-compatible StreamEvents with:
- reasoning_content: Content inside <think> tags (for thinking models)
- content: Regular response content
Following OpenAI o1/o3 reasoning model API spec.

Model-Family-Specific Patterns:
The processor can be configured with model-family-specific patterns via
create_processor_for_family(). This prevents pattern conflicts where e.g.
GLM4 attr-style patterns could incorrectly match against Qwen output.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from mlx_manager.mlx_server.schemas.openai import FunctionCall, ToolCall

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.adapters.base import ModelAdapter

# --- Stream Event and Pattern Dataclasses ---


@dataclass
class StreamEvent:
    """Event from streaming processor for OpenAI-compatible streaming.

    Follows OpenAI o1/o3 reasoning model API spec:
    - reasoning_content: Content inside <think> tags (thinking phase)
    - content: Regular response content
    - is_complete: True when a thinking pattern ends (transition point)
    """

    content: str | None = None
    reasoning_content: str | None = None
    is_complete: bool = False


@dataclass
class ToolPatternSpec:
    """Specification for a tool call pattern.

    Used by model adapters to declare model-family-specific patterns.
    The ResponseProcessor factory uses these specs to register only
    the relevant patterns for each model family.

    Attributes:
        pattern: Regex pattern string to match tool calls
        parser_name: Name of the parser function to use (e.g., 'hermes', 'llama_xml')
        description: Human-readable description of the pattern
    """

    pattern: str
    parser_name: str
    description: str = ""


@dataclass
class ModelFamilyPatterns:
    """Pattern configuration for a model family.

    Specifies which tool call patterns and thinking tags to use
    for a specific model family (e.g., 'qwen', 'glm4', 'llama').

    Attributes:
        tool_patterns: List of tool pattern specifications
        thinking_tags: List of thinking tag names (e.g., ['think', 'thinking'])
        cleanup_tokens: List of special tokens to remove from output
    """

    tool_patterns: list[ToolPatternSpec] = field(default_factory=list)
    thinking_tags: list[str] = field(default_factory=list)
    cleanup_tokens: list[str] = field(default_factory=list)


# --- Pydantic Models ---

# ToolCall and FunctionCall are imported from schemas/openai.py (canonical types).
# No duplicate definitions here — see Principle P4 (one canonical type per concept).


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
            function=FunctionCall(name=name, arguments=arguments_str),
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
            function=FunctionCall(name=name, arguments=args_str),
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
            function=FunctionCall(name=name, arguments=args_str),
        )
    except (IndexError, AttributeError) as e:
        logger.warning("Invalid GLM4 tool call: %s", e)
        return None


def parse_glm4_compact_tool(match: re.Match[str]) -> ToolCall | None:
    """Parse GLM4.7 compact style: <tool_call>func_name<param>value</param>..."""
    try:
        content = match.group(1).strip()

        # Extract function name (first word before any tag)
        name_match = re.match(r"(\w+)", content)
        if not name_match:
            return None
        name = name_match.group(1)

        # Extract parameters from individual tags like <location>Rome</location>
        param_pattern = r"<(\w+)>([^<]*)</\1>"
        params = dict(re.findall(param_pattern, content))

        args_str = json.dumps(params) if params else "{}"

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            function=FunctionCall(name=name, arguments=args_str),
        )
    except (IndexError, AttributeError) as e:
        logger.warning("Invalid GLM4.7 compact tool call: %s", e)
        return None


def parse_glm4_attr_tool(match: re.Match[str]) -> ToolCall | None:
    """Parse GLM4.7 malformed attr style: <tool_call>func_name<param="value"</param>...

    GLM-4.7-Flash sometimes outputs a hybrid format where parameter values
    appear as attributes in the opening tag: <location="Rome, Italy"</location>
    """
    try:
        content = match.group(1).strip()

        # Extract function name (first word before any tag)
        name_match = re.match(r"(\w+)", content)
        if not name_match:
            return None
        name = name_match.group(1)

        # Extract parameters from malformed attr tags like <location="value"</location>
        # Pattern: <param="value"</param> or <param="value"/>
        attr_pattern = r'<(\w+)="([^"]*)"(?:</\1>|/>|<)'
        params = dict(re.findall(attr_pattern, content))

        if not params:
            return None

        args_str = json.dumps(params)
        logger.debug(f"Parsed GLM4.7 attr-style tool call: {name}({args_str})")

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            function=FunctionCall(name=name, arguments=args_str),
        )
    except (IndexError, AttributeError) as e:
        logger.warning("Invalid GLM4.7 attr tool call: %s", e)
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
            function=FunctionCall(name=name, arguments=json.dumps(args_dict)),
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


# --- Parser Registry ---

# Maps parser names to parser functions
PARSER_REGISTRY: dict[str, Callable[[re.Match[str]], ToolCall | None]] = {
    "hermes": parse_hermes_tool,
    "llama_xml": parse_llama_tool,
    "llama_python": parse_llama_python_tool,
    "glm4_xml": parse_glm4_tool,
    "glm4_compact": parse_glm4_compact_tool,
    "glm4_attr": parse_glm4_attr_tool,
}


# --- Model Family Pattern Definitions ---

# Common thinking tags used across model families
COMMON_THINKING_TAGS = ["think", "thinking", "reasoning", "reflection"]

# Common cleanup tokens
COMMON_CLEANUP_TOKENS = [
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|eot_id|>",
    "<|end|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
]

# Qwen/Hermes family patterns (JSON inside <tool_call> tags)
QWEN_PATTERNS = ModelFamilyPatterns(
    tool_patterns=[
        # Hermes/Qwen JSON format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        ToolPatternSpec(
            pattern=r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            parser_name="hermes",
            description="Hermes/Qwen JSON format with closing tag",
        ),
        # Hermes/Qwen unclosed format: <tool_call>{"name": ..., "arguments": ...}$
        # Some models (e.g., Qwen3 0.6B) don't output closing </tool_call> tag
        ToolPatternSpec(
            pattern=r"<tool_call>\s*(\{.+\})\s*$",
            parser_name="hermes",
            description="Hermes/Qwen JSON format without closing tag",
        ),
    ],
    thinking_tags=COMMON_THINKING_TAGS,
    cleanup_tokens=COMMON_CLEANUP_TOKENS,
)

# GLM4 family patterns (XML-style and compact/attr variants)
GLM4_PATTERNS = ModelFamilyPatterns(
    tool_patterns=[
        # GLM4 XML format: <tool_call><name>...</name><arguments>...</arguments></tool_call>
        ToolPatternSpec(
            pattern=(
                r"<tool_call>\s*<name>(.*?)</name>\s*"
                r"<arguments>(.*?)</arguments>\s*</tool_call>"
            ),
            parser_name="glm4_xml",
            description="GLM4 full XML format",
        ),
        # GLM4.7 compact format: <tool_call>func_name<param>value</param>...
        ToolPatternSpec(
            pattern=r"<tool_call>([^{<][^<]*(?:<\w+>[^<]*</\w+>)+)(?:</tool_call>|$)",
            parser_name="glm4_compact",
            description="GLM4.7 compact format with param tags",
        ),
        # GLM4.7 malformed attr format: <tool_call>func_name<param="value"</param>...
        ToolPatternSpec(
            pattern=r'<tool_call>(\w+(?:<\w+="[^"]*"</\w+>)+)\s*(?:</tool_call>|$)',
            parser_name="glm4_attr",
            description="GLM4.7 malformed attr-style format",
        ),
    ],
    thinking_tags=COMMON_THINKING_TAGS,
    cleanup_tokens=COMMON_CLEANUP_TOKENS,
)

# Llama family patterns
LLAMA_PATTERNS = ModelFamilyPatterns(
    tool_patterns=[
        # Llama XML format: <function=name>{...}</function>
        ToolPatternSpec(
            pattern=r"<function=(\w+)>(.*?)</function>",
            parser_name="llama_xml",
            description="Llama XML function format",
        ),
        # Llama Python format: <|python_tag|>module.method(args)<|eom_id|>
        ToolPatternSpec(
            pattern=r"<\|python_tag\|>\s*(\w+)\.(\w+)\((.*?)\)\s*<\|eom_id\|>",
            parser_name="llama_python",
            description="Llama Python tag format",
        ),
    ],
    thinking_tags=COMMON_THINKING_TAGS,
    cleanup_tokens=COMMON_CLEANUP_TOKENS,
)

# Default patterns: includes all patterns for backward compatibility
# but orders them from most specific to least specific to reduce conflicts
DEFAULT_PATTERNS = ModelFamilyPatterns(
    tool_patterns=[
        # Most specific first: Hermes JSON (most common, clear format)
        *QWEN_PATTERNS.tool_patterns,
        # Then Llama patterns (distinct markers)
        *LLAMA_PATTERNS.tool_patterns,
        # GLM4 patterns last (can be more ambiguous)
        *GLM4_PATTERNS.tool_patterns,
    ],
    thinking_tags=COMMON_THINKING_TAGS,
    cleanup_tokens=COMMON_CLEANUP_TOKENS,
)

# Registry mapping model family names to their patterns
MODEL_FAMILY_PATTERNS: dict[str, ModelFamilyPatterns] = {
    "qwen": QWEN_PATTERNS,
    "glm4": GLM4_PATTERNS,
    "llama": LLAMA_PATTERNS,
    "default": DEFAULT_PATTERNS,
}


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

        # Debug: Log tool pattern matching
        if "<tool_call>" in text or "<function=" in text:
            logger.debug(
                f"ResponseProcessor.process(): found potential tool markers in text "
                f"(len={len(text)}), checking {len(self._tool_patterns)} patterns"
            )

        for pattern, parser in self._tool_patterns:
            for match in pattern.finditer(text):
                logger.debug(
                    f"Tool pattern matched: {pattern.pattern[:50]}... at pos {match.start()}"
                )
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

    def _merge_overlapping_spans(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
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


def create_processor_from_patterns(patterns: ModelFamilyPatterns) -> ResponseProcessor:
    """Create processor from a ModelFamilyPatterns specification.

    Args:
        patterns: Pattern configuration for the model family

    Returns:
        ResponseProcessor with only the specified patterns registered
    """
    processor = ResponseProcessor()

    # Register thinking tags
    processor.register_thinking_tags(patterns.thinking_tags)

    # Register tool patterns
    for spec in patterns.tool_patterns:
        parser = PARSER_REGISTRY.get(spec.parser_name)
        if parser is None:
            logger.warning(f"Unknown parser '{spec.parser_name}' in pattern spec")
            continue
        processor.register_tool_pattern(spec.pattern, parser)

    # Register cleanup patterns
    processor.register_cleanup_patterns(patterns.cleanup_tokens)

    return processor


def create_processor_for_family(family: str) -> ResponseProcessor:
    """Create a ResponseProcessor with model-family-specific patterns.

    This is the preferred way to create processors for specific models.
    Using family-specific patterns prevents conflicts where e.g. GLM4's
    attr-style patterns could incorrectly match against Qwen output.

    Args:
        family: Model family name (e.g., 'qwen', 'glm4', 'llama')

    Returns:
        ResponseProcessor with only the patterns for that model family
    """
    patterns = MODEL_FAMILY_PATTERNS.get(family.lower(), DEFAULT_PATTERNS)
    logger.debug(
        f"Creating processor for family '{family}' with {len(patterns.tool_patterns)} tool patterns"
    )
    return create_processor_from_patterns(patterns)


def create_processor_for_adapter(adapter: ModelAdapter) -> ResponseProcessor:
    """Create a ResponseProcessor from a model adapter.

    Uses the adapter's family property to select the appropriate patterns.

    Args:
        adapter: Model adapter instance

    Returns:
        ResponseProcessor with patterns for the adapter's model family
    """
    return create_processor_for_family(adapter.family)


def create_default_processor() -> ResponseProcessor:
    """Create processor with all standard handlers registered.

    DEPRECATED: Use create_processor_for_family() for model-specific processing.
    This function creates a processor with ALL patterns, which may cause
    conflicts between different model family formats.

    Kept for backward compatibility.

    Returns:
        ResponseProcessor with thinking, tool, and cleanup patterns configured
    """
    return create_processor_from_patterns(DEFAULT_PATTERNS)


# Singleton instances (per-family cache)
_processor: ResponseProcessor | None = None
_family_processors: dict[str, ResponseProcessor] = {}


def get_response_processor() -> ResponseProcessor:
    """Get the singleton ResponseProcessor instance with ALL patterns.

    DEPRECATED: Prefer get_processor_for_family() for model-specific processing.

    Returns:
        Shared ResponseProcessor with all patterns registered
    """
    global _processor
    if _processor is None:
        _processor = create_default_processor()
    return _processor


def get_processor_for_family(family: str) -> ResponseProcessor:
    """Get a cached ResponseProcessor for a specific model family.

    This is the preferred way to get processors - it ensures only
    the relevant patterns are active, preventing cross-family conflicts.

    Args:
        family: Model family name (e.g., 'qwen', 'glm4', 'llama')

    Returns:
        Cached ResponseProcessor with patterns for that model family
    """
    family_lower = family.lower()
    if family_lower not in _family_processors:
        _family_processors[family_lower] = create_processor_for_family(family_lower)
    return _family_processors[family_lower]


def reset_response_processor() -> None:
    """Reset all singleton instances (for testing)."""
    global _processor, _family_processors
    _processor = None
    _family_processors.clear()


# --- Streaming Processor ---


class StreamingProcessor:
    """Streaming-aware processor that yields OpenAI-compatible StreamEvents.

    Returns StreamEvent objects with either:
    - reasoning_content: Content inside thinking tags (for thinking models)
    - content: Regular response content

    This follows OpenAI o1/o3 reasoning model API spec where thinking
    content goes in delta.reasoning_content and regular content in delta.content.

    Usage:
        processor = StreamingProcessor()
        for token in generation:
            event = processor.feed(token)
            if event.reasoning_content or event.content:
                yield event
        # After generation
        result = processor.finalize()

    Key behaviors:
    - Thinking content (<think>, etc.) streamed as reasoning_content
    - Tool call markers filtered (only in finalize())
    - Regular content streamed as content
    - Final processing extracts tool_calls from accumulated text
    """

    # Thinking pattern markers (content streamed as reasoning_content)
    THINKING_STARTS = [
        "<think>",
        "<thinking>",
        "<reasoning>",
        "<reflection>",
    ]

    # Tool pattern markers (content filtered, extracted in finalize)
    TOOL_STARTS = [
        "<tool_call>",
        "<function=",
        "<|python_tag|>",
    ]

    # All pattern start markers
    PATTERN_STARTS = THINKING_STARTS + TOOL_STARTS

    # Pattern end markers (map start -> end)
    PATTERN_ENDS = {
        "<think>": "</think>",
        "<thinking>": "</thinking>",
        "<reasoning>": "</reasoning>",
        "<reflection>": "</reflection>",
        "<tool_call>": "</tool_call>",
        "<function=": "</function>",
        "<|python_tag|>": "<|eom_id|>",
    }

    # Buffer size for incremental reasoning yield (avoid partial tokens)
    REASONING_BUFFER_SIZE = 10

    def __init__(
        self,
        response_processor: ResponseProcessor | None = None,
        starts_in_thinking: bool = False,
        model_family: str | None = None,
    ) -> None:
        """Initialize streaming processor.

        Args:
            response_processor: Optional ResponseProcessor for final parsing.
                               If provided, takes precedence over model_family.
            starts_in_thinking: If True, treat initial content as reasoning
                               until </think> is found. Used when the prompt
                               already ends with <think> (e.g., GLM-4.7).
            model_family: Optional model family name (e.g., 'qwen', 'glm4', 'llama').
                         If provided, uses family-specific patterns instead of
                         the global processor. This prevents cross-family
                         pattern conflicts.
        """
        if response_processor is not None:
            self._processor = response_processor
        elif model_family is not None:
            self._processor = get_processor_for_family(model_family)
        else:
            # Backward compatible: use global processor with all patterns
            self._processor = get_response_processor()

        self._model_family = model_family
        self._buffer = ""  # Buffer for content within a pattern
        self._pending_buffer = ""  # For partial marker detection
        self._in_pattern = starts_in_thinking
        self._pattern_start = "<think>" if starts_in_thinking else ""
        self._is_thinking_pattern = starts_in_thinking
        self._accumulated = ""  # Full response for final processing
        self._yielded_content = ""  # What we've yielded as content to client

    def feed(self, token: str) -> StreamEvent:
        """Feed a token, get StreamEvent.

        Args:
            token: Next token from generation

        Returns:
            StreamEvent with reasoning_content (inside thinking tags)
            or content (regular text), or empty if buffering
        """
        self._accumulated += token

        if self._in_pattern:
            return self._handle_in_pattern(token)

        # Check if token starts or contains a pattern start
        combined = self._pending_buffer + token
        self._pending_buffer = ""

        # Check for complete pattern starts first
        for start in self.PATTERN_STARTS:
            if start in combined:
                return self._handle_pattern_start(combined, start)

        # Check for partial match at end (e.g., "<tool" might become "<tool_call>")
        for start in self.PATTERN_STARTS:
            for i in range(1, len(start)):
                partial = start[:i]
                if combined.endswith(partial):
                    # Found partial match - buffer it
                    self._pending_buffer = combined[-i:]
                    to_yield = combined[:-i]
                    if to_yield:
                        self._yielded_content += to_yield
                        return StreamEvent(content=to_yield)
                    return StreamEvent()

        # No pattern detected, yield token as content
        self._yielded_content += combined
        return StreamEvent(content=combined)

    def _handle_in_pattern(self, token: str) -> StreamEvent:
        """Handle token while inside a pattern.

        For thinking patterns: yield content as reasoning_content incrementally
        For tool patterns: buffer silently (extracted in finalize)
        """
        self._buffer += token
        end_marker = self.PATTERN_ENDS.get(self._pattern_start, "")

        if end_marker and end_marker in self._buffer:
            # Pattern complete
            self._in_pattern = False
            end_idx = self._buffer.index(end_marker)
            pattern_content = self._buffer[:end_idx]
            after_pattern = self._buffer[end_idx + len(end_marker) :]
            self._buffer = ""
            self._pattern_start = ""
            was_thinking = self._is_thinking_pattern
            self._is_thinking_pattern = False

            if after_pattern:
                # Content after pattern - recurse
                subsequent = self.feed(after_pattern)
                if was_thinking and pattern_content:
                    # Return final reasoning chunk + any subsequent content
                    return StreamEvent(
                        reasoning_content=pattern_content,
                        content=subsequent.content,
                        is_complete=True,
                    )
                return subsequent

            if was_thinking:
                # Return final reasoning chunk
                return StreamEvent(
                    reasoning_content=pattern_content if pattern_content else None,
                    is_complete=True,
                )
            return StreamEvent()

        # Still inside pattern
        if self._is_thinking_pattern:
            # Filter out nested thinking tags from buffer (some models output <think><think>)
            # This handles cases where the model incorrectly outputs duplicate start tags
            for nested_start in self.THINKING_STARTS:
                while nested_start in self._buffer:
                    self._buffer = self._buffer.replace(nested_start, "", 1)

            # Yield reasoning content incrementally, keeping small buffer
            # to avoid partial end markers
            if len(self._buffer) > self.REASONING_BUFFER_SIZE:
                to_yield = self._buffer[: -self.REASONING_BUFFER_SIZE]
                self._buffer = self._buffer[-self.REASONING_BUFFER_SIZE :]
                return StreamEvent(reasoning_content=to_yield)

        # Buffering (tool pattern or small reasoning buffer)
        return StreamEvent()

    def _handle_pattern_start(self, combined: str, start: str) -> StreamEvent:
        """Handle detection of a pattern start marker."""
        idx = combined.index(start)
        before = combined[:idx]
        self._in_pattern = True
        self._pattern_start = start
        self._is_thinking_pattern = start in self.THINKING_STARTS
        # Buffer starts AFTER the start marker (we don't want the marker in output)
        self._buffer = combined[idx + len(start) :]

        # Check if pattern already ends in this combined text
        end_marker = self.PATTERN_ENDS.get(start, "")
        if end_marker and end_marker in self._buffer:
            # Complete pattern in single combined text
            end_idx = self._buffer.index(end_marker)
            pattern_content = self._buffer[:end_idx]
            after_pattern = self._buffer[end_idx + len(end_marker) :]
            self._buffer = ""
            self._pattern_start = ""
            was_thinking = self._is_thinking_pattern
            self._is_thinking_pattern = False
            self._in_pattern = False

            if before:
                self._yielded_content += before
                if after_pattern:
                    subsequent = self.feed(after_pattern)
                    if was_thinking:
                        return StreamEvent(
                            content=before,
                            reasoning_content=pattern_content if pattern_content else None,
                        )
                    return StreamEvent(content=before + (subsequent.content or ""))
                if was_thinking:
                    return StreamEvent(
                        content=before,
                        reasoning_content=pattern_content if pattern_content else None,
                        is_complete=True,
                    )
                return StreamEvent(content=before)
            elif after_pattern:
                if was_thinking:
                    subsequent = self.feed(after_pattern)
                    return StreamEvent(
                        reasoning_content=pattern_content if pattern_content else None,
                        content=subsequent.content,
                        is_complete=True,
                    )
                return self.feed(after_pattern)
            if was_thinking:
                return StreamEvent(
                    reasoning_content=pattern_content if pattern_content else None,
                    is_complete=True,
                )
            return StreamEvent()

        if before:
            self._yielded_content += before
            return StreamEvent(content=before)
        return StreamEvent()

    def finalize(self) -> ParseResult:
        """Finalize and get complete ParseResult.

        Called after all tokens processed. Uses ResponseProcessor
        to extract structured data from accumulated text.

        Returns:
            ParseResult with content, tool_calls, and reasoning
        """
        # Flush any pending buffer (incomplete pattern marker)
        if self._pending_buffer:
            self._yielded_content += self._pending_buffer
            self._pending_buffer = ""

        # Debug: Log accumulated text for tool call analysis
        logger.debug(
            f"StreamingProcessor.finalize(): accumulated={len(self._accumulated)} chars, "
            f"contains_tool_call_marker={'<tool_call>' in self._accumulated}"
        )

        # Process full accumulated text to get structured data
        result = self._processor.process(self._accumulated)

        # Debug: Log extraction results
        tool_count = len(result.tool_calls) if result.tool_calls else 0
        logger.debug(
            f"StreamingProcessor.finalize(): extracted tool_calls={tool_count}, "
            f"has_reasoning={bool(result.reasoning)}, content_len={len(result.content)}"
        )

        return result

    def get_pending_content(self) -> str:
        """Get any buffered content not yet yielded.

        Returns:
            Buffered content (pending marker + pattern buffer)
        """
        return self._pending_buffer + self._buffer

    def get_accumulated_text(self) -> str:
        """Get all accumulated text for logging/metrics.

        Returns:
            Complete accumulated text including pattern content
        """
        return self._accumulated
