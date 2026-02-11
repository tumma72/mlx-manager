"""Coverage tests for StreamingProcessor edge/branching paths.

Targets uncovered lines in response_processor.py:
- Lines 171-172: Partial marker match with content before it
- Lines 202-210: Pattern end with subsequent content (_handle_in_pattern)
- Line 227: Nested thinking tag removal
- Lines 254-294: Complete pattern in single combined text (_handle_pattern_start)
- Lines 297-298: Content before pattern start (no complete pattern)
- Lines 312-313: Pending buffer flush in finalize()
- Line 346: get_pending_content()
"""

from unittest.mock import MagicMock

from mlx_manager.mlx_server.models.adapters.composable import create_adapter
from mlx_manager.mlx_server.services.response_processor import StreamingProcessor


def _make_mock_tokenizer(eos_token_id: int = 0) -> MagicMock:
    tok = MagicMock()
    tok.eos_token_id = eos_token_id
    return tok


def _make_qwen_processor() -> StreamingProcessor:
    """Build a StreamingProcessor with QwenAdapter (has <think>/<tool_call> markers)."""
    tok = _make_mock_tokenizer()
    adapter = create_adapter(family="qwen", tokenizer=tok)
    return StreamingProcessor(adapter=adapter)


class TestPartialMarkerMatch:
    """Cover lines 171-172: partial marker match with content to yield."""

    def test_partial_marker_yields_content_before(self) -> None:
        """When text ends with a partial marker, yield content before it."""
        proc = _make_qwen_processor()
        # Feed "Hello<thi" - "Hello" is content, "<thi" is partial "<think>"
        event = proc.feed("Hello<thi")
        assert event.content == "Hello"

    def test_partial_marker_completes_on_next_token(self) -> None:
        """Partial marker resolves to full marker on next feed."""
        proc = _make_qwen_processor()
        # First token: partial "<thi" at the end
        event1 = proc.feed("Hello<thi")
        assert event1.content == "Hello"

        # Next token completes the marker
        event2 = proc.feed("nk>I am thinking")
        # Now we're inside thinking pattern, so reasoning content gets buffered
        assert event2.content is None or event2.content == ""

    def test_partial_marker_no_content_before(self) -> None:
        """Partial marker with no content before yields empty event."""
        proc = _make_qwen_processor()
        event = proc.feed("<thi")
        assert event.content is None


class TestHandleInPatternEnd:
    """Cover lines 202-210: pattern end with subsequent content."""

    def test_thinking_ends_with_subsequent_content(self) -> None:
        """Thinking pattern end with content after yields both."""
        proc = _make_qwen_processor()
        # Enter thinking pattern
        proc.feed("<think>")

        # End thinking with content after
        event = proc.feed("deep thought</think>Here is my answer")
        # Should yield reasoning_content and/or content
        # The "deep thought" goes to reasoning, "Here is my answer" to content
        assert event.reasoning_content is not None or event.content is not None

    def test_thinking_ends_with_subsequent_tool_pattern(self) -> None:
        """After thinking ends, subsequent content can start new pattern."""
        proc = _make_qwen_processor()
        proc.feed("<think>")

        # End thinking then immediately start a tool call
        event = proc.feed("thought</think><tool_call>")
        # The thinking ends and tool call starts
        assert event.is_complete or event.reasoning_content is not None

    def test_tool_pattern_ends_with_subsequent_content(self) -> None:
        """Tool pattern ending with content after yields that content."""
        proc = _make_qwen_processor()
        proc.feed("<tool_call>")
        event = proc.feed('{"name":"fn"}</tool_call>The result is')
        # After tool pattern ends, "The result is" becomes content
        assert event.content is not None


class TestNestedThinkingTag:
    """Cover line 227: nested thinking tag removal."""

    def test_duplicate_think_start_inside_pattern(self) -> None:
        """Model outputting <think><think> should strip nested tag."""
        proc = _make_qwen_processor()
        proc.feed("<think>")

        # Feed another <think> inside the thinking pattern
        # The processor should strip it
        proc.feed("<think>actual reasoning content here")
        # Content should not include the nested <think>
        # It may be buffered due to REASONING_BUFFER_SIZE
        pending = proc.get_pending_content()
        assert "<think>" not in pending

    def test_multiple_nested_thinking_starts(self) -> None:
        """Multiple nested thinking starts are all removed."""
        proc = _make_qwen_processor()
        proc.feed("<think>")

        # Feed content with multiple nested starts
        proc.feed("<think><thinking>real content<think>more content")
        pending = proc.get_pending_content()
        assert "<think>" not in pending


class TestHandlePatternStartComplete:
    """Cover lines 254-294: complete pattern in single combined text."""

    def test_complete_thinking_in_single_feed(self) -> None:
        """Entire thinking block in one feed call."""
        proc = _make_qwen_processor()
        event = proc.feed("<think>quick thought</think>")
        assert event.reasoning_content is not None
        assert event.is_complete is True

    def test_complete_thinking_with_before_content(self) -> None:
        """Content before + complete thinking in one feed."""
        proc = _make_qwen_processor()
        event = proc.feed("Preamble<think>thought</think>")
        # Should have both content (before) and reasoning
        assert event.content is not None
        assert "Preamble" in event.content

    def test_complete_thinking_with_before_and_after(self) -> None:
        """Content before + thinking + content after in one feed."""
        proc = _make_qwen_processor()
        event = proc.feed("Before<think>thought</think>After")
        assert event.content is not None
        assert "Before" in event.content

    def test_complete_tool_pattern_with_before_content(self) -> None:
        """Content before + complete tool pattern in one feed."""
        proc = _make_qwen_processor()
        event = proc.feed('Prefix<tool_call>{"name":"fn"}</tool_call>')
        assert event.content is not None
        assert "Prefix" in event.content

    def test_complete_tool_pattern_with_before_and_after(self) -> None:
        """Content before + tool + content after in one feed."""
        proc = _make_qwen_processor()
        event = proc.feed('Before<tool_call>{"fn":1}</tool_call>After')
        assert event.content is not None

    def test_complete_thinking_no_before_with_after(self) -> None:
        """Complete thinking with no content before but content after."""
        proc = _make_qwen_processor()
        event = proc.feed("<think>thought</think>After content")
        # reasoning_content should be set, after goes to content
        assert event.reasoning_content is not None or event.content is not None

    def test_complete_thinking_no_before_no_after(self) -> None:
        """Complete thinking alone in one feed."""
        proc = _make_qwen_processor()
        event = proc.feed("<think>thought</think>")
        assert event.reasoning_content is not None
        assert event.is_complete is True

    def test_complete_tool_no_before_with_after(self) -> None:
        """Complete tool call with no before but content after."""
        proc = _make_qwen_processor()
        event = proc.feed("<tool_call>data</tool_call>Response here")
        # Tool content is buffered, "Response here" should come through
        assert event.content is not None

    def test_complete_tool_no_before_no_after(self) -> None:
        """Complete tool call alone in one feed."""
        proc = _make_qwen_processor()
        event = proc.feed("<tool_call>data</tool_call>")
        # Tool pattern captured, nothing to yield
        assert event.content is None or event.content == ""


class TestContentBeforePatternStart:
    """Cover lines 297-298: content before pattern start (no complete pattern)."""

    def test_content_before_thinking_start(self) -> None:
        """Regular content followed by thinking start marker."""
        proc = _make_qwen_processor()
        event = proc.feed("Hello world <think>")
        assert event.content is not None
        assert "Hello world " in event.content

    def test_content_before_tool_start(self) -> None:
        """Regular content followed by tool call start marker."""
        proc = _make_qwen_processor()
        event = proc.feed("Let me help<tool_call>")
        assert event.content is not None
        assert "Let me help" in event.content


class TestFinalizeWithPending:
    """Cover lines 312-313: pending buffer flush in finalize()."""

    def test_finalize_flushes_pending_buffer(self) -> None:
        """Pending partial marker is flushed as content in finalize."""
        proc = _make_qwen_processor()
        # Create a partial marker in pending buffer
        proc.feed("Content<thi")
        # Now finalize - the partial should be flushed
        result = proc.finalize()
        # The accumulated text should include the partial
        assert "<thi" in result.content

    def test_finalize_with_no_pending(self) -> None:
        """Finalize with no pending buffer works normally."""
        proc = _make_qwen_processor()
        proc.feed("Hello world")
        result = proc.finalize()
        assert result.content == "Hello world"


class TestGetPendingContent:
    """Cover line 346: get_pending_content()."""

    def test_get_pending_content_with_partial_marker(self) -> None:
        """Returns pending buffer content from partial marker."""
        proc = _make_qwen_processor()
        proc.feed("Text<thi")
        pending = proc.get_pending_content()
        assert "<thi" in pending

    def test_get_pending_content_with_pattern_buffer(self) -> None:
        """Returns buffer content when inside a pattern."""
        proc = _make_qwen_processor()
        proc.feed("<think>some reasoning")
        pending = proc.get_pending_content()
        assert "some reasoning" in pending

    def test_get_pending_content_empty(self) -> None:
        """Returns empty string when nothing is buffered."""
        proc = _make_qwen_processor()
        proc.feed("Hello")
        pending = proc.get_pending_content()
        assert pending == ""


class TestStartsInThinking:
    """Test starts_in_thinking mode for completeness."""

    def test_starts_in_thinking_yields_reasoning(self) -> None:
        """When starting in thinking mode, content is reasoning."""
        tok = _make_mock_tokenizer()
        adapter = create_adapter(family="qwen", tokenizer=tok)
        proc = StreamingProcessor(adapter=adapter, starts_in_thinking=True)

        # Feed enough to exceed REASONING_BUFFER_SIZE
        event = proc.feed("x" * 20)
        assert event.reasoning_content is not None

    def test_starts_in_thinking_ends_on_close_tag(self) -> None:
        """Thinking mode exits on </think>."""
        tok = _make_mock_tokenizer()
        adapter = create_adapter(family="qwen", tokenizer=tok)
        proc = StreamingProcessor(adapter=adapter, starts_in_thinking=True)

        # Feed content and close the thinking
        proc.feed("reasoning")
        event = proc.feed("</think>answer")
        # Should have transitioned out of thinking
        assert event.content is not None or event.reasoning_content is not None


class TestGetAccumulatedText:
    """Test get_accumulated_text() for completeness."""

    def test_accumulated_text_tracks_all_tokens(self) -> None:
        proc = _make_qwen_processor()
        proc.feed("Hello ")
        proc.feed("world")
        assert proc.get_accumulated_text() == "Hello world"

    def test_accumulated_text_includes_patterns(self) -> None:
        proc = _make_qwen_processor()
        proc.feed("<think>thought</think>answer")
        text = proc.get_accumulated_text()
        assert "<think>" in text
        assert "answer" in text
