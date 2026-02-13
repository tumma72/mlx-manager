"""Streaming response processor for the adapter pipeline.

Layer 2 of the 3-layer adapter pipeline:
  Layer 1 (ModelAdapter)   -> input preparation
  Layer 2 (StreamProcessor) -> token-by-token streaming + finalization
  Layer 3 (ProtocolFormatter) -> protocol-specific responses (future)

Exports:
- StreamProcessor: Token-by-token processor using adapter's parsers
- StreamEvent: IR event type (re-exported from models.ir)
- TextResult: IR result type (re-exported from models.ir)
- ParseResult: Legacy result type (kept for backward compat, removed in Phase 6)
- StreamingProcessor: Backward-compat alias for StreamProcessor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult
from mlx_manager.mlx_server.schemas.openai import ToolCall

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

# --- Legacy Types (kept for backward compat, deleted in Phase 6) ---


class ParseResult(BaseModel):
    """Result of processing a model response.

    .. deprecated::
        Use :class:`TextResult` from ``models.ir`` instead.
        Kept for backward compatibility; will be removed in Phase 6.
    """

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: str | None = None


# --- Stream Processor ---


class StreamProcessor:
    """Streaming-aware processor that yields IR StreamEvents.

    Returns StreamEvent objects with either:
    - reasoning_content: Content inside thinking tags (for thinking models)
    - content: Regular response content

    This follows OpenAI o1/o3 reasoning model API spec where thinking
    content goes in delta.reasoning_content and regular content in
    delta.content.

    Pattern configuration is derived from the adapter's parsers, ensuring
    consistent behavior between streaming and non-streaming paths.

    Usage:
        processor = adapter.create_stream_processor(prompt=prompt)
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

    # Buffer size for incremental reasoning yield (avoid partial tokens)
    REASONING_BUFFER_SIZE = 10

    def __init__(
        self,
        adapter: ModelAdapter,
        starts_in_thinking: bool = False,
    ) -> None:
        """Initialize stream processor.

        Args:
            adapter: Adapter with parsers for tool/thinking extraction
            starts_in_thinking: If True, treat initial content as reasoning
                               until </think> is found. Used when the prompt
                               already ends with <think> (e.g., GLM-4.7).
        """
        self._adapter = adapter

        # Derive streaming markers from adapter's parsers
        # stream_markers is a property returning list[tuple[str, str]]
        thinking_markers = adapter.thinking_parser.stream_markers
        tool_markers = adapter.tool_parser.stream_markers

        # Extract start markers
        self._thinking_starts = [start for start, _ in thinking_markers]
        self._tool_starts = [start for start, _ in tool_markers]
        self._pattern_starts = self._thinking_starts + self._tool_starts

        # Build pattern end mappings from tuples
        self._pattern_ends: dict[str, str] = {}
        for start, end in thinking_markers:
            self._pattern_ends[start] = end
        for start, end in tool_markers:
            self._pattern_ends[start] = end

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
        for start in self._pattern_starts:
            if start in combined:
                return self._handle_pattern_start(combined, start)

        # Check for partial match at end (e.g., "<tool" might become
        # "<tool_call>")
        for start in self._pattern_starts:
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

        For thinking patterns: yield content as reasoning_content
        incrementally. For tool patterns: buffer silently (extracted in
        finalize).
        """
        self._buffer += token
        end_marker = self._pattern_ends.get(self._pattern_start, "")

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
            # Filter out nested thinking tags from buffer (some models output
            # <think><think>). This handles cases where the model incorrectly
            # outputs duplicate start tags.
            for nested_start in self._thinking_starts:
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
        self._is_thinking_pattern = start in self._thinking_starts
        # Buffer starts AFTER the start marker (we don't want the marker in
        # output)
        self._buffer = combined[idx + len(start) :]

        # Check if pattern already ends in this combined text
        end_marker = self._pattern_ends.get(start, "")
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
                            reasoning_content=(pattern_content if pattern_content else None),
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

    def finalize(self) -> TextResult:
        """Finalize and get IR TextResult.

        Called after all tokens processed. Uses adapter's parsers
        to extract structured data from accumulated text.

        Returns:
            TextResult with content, tool_calls, reasoning_content,
            and finish_reason
        """
        # Flush any pending buffer (incomplete pattern marker)
        if self._pending_buffer:
            self._yielded_content += self._pending_buffer
            self._pending_buffer = ""

        # Debug: Log accumulated text for tool call analysis
        logger.debug(f"StreamProcessor.finalize(): accumulated={len(self._accumulated)} chars")

        # Use adapter's parsers for final extraction
        tool_calls_list = self._adapter.tool_parser.extract(self._accumulated)
        reasoning_content = self._adapter.thinking_parser.extract(self._accumulated)
        final_content = self._accumulated
        if reasoning_content:
            final_content = self._adapter.thinking_parser.remove(final_content)
        final_content = self._adapter.clean_response(final_content)

        # Debug: Log extraction results
        tool_count = len(tool_calls_list)
        logger.debug(
            f"StreamProcessor.finalize(): extracted tool_calls={tool_count}, "
            f"has_reasoning={bool(reasoning_content)}, "
            f"content_len={len(final_content)}"
        )

        # Convert ToolCall Pydantic models to dicts for IR
        tool_calls = [tc.model_dump() for tc in tool_calls_list] if tool_calls_list else None
        finish_reason = "tool_calls" if tool_calls else "stop"

        return TextResult(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

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


# Backward-compat alias (deleted in Phase 6)
StreamingProcessor = StreamProcessor
