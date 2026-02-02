"""Golden file integration tests for ResponseProcessor and StreamingProcessor.

Tests validate that:
1. Tool call markers are extracted and removed from content
2. Thinking/reasoning tags are extracted and removed
3. Streaming processor filters patterns across chunk boundaries
4. All model family formats are correctly parsed
"""

import pytest
from pathlib import Path

from mlx_manager.mlx_server.services.response_processor import (
    get_response_processor,
    StreamingProcessor,
    reset_response_processor,
)

GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "golden"


def collect_tool_call_files() -> list[tuple[str, Path]]:
    """Collect all tool call golden files for parametrization."""
    test_cases: list[tuple[str, Path]] = []
    for family_dir in sorted(GOLDEN_DIR.iterdir()):
        if not family_dir.is_dir():
            continue
        tool_file = family_dir / "tool_calls.txt"
        if tool_file.exists():
            test_cases.append((family_dir.name, tool_file))
    return test_cases


def collect_streaming_files() -> list[tuple[str, str, Path]]:
    """Collect all streaming golden files for parametrization."""
    test_cases: list[tuple[str, str, Path]] = []
    for family_dir in sorted(GOLDEN_DIR.iterdir()):
        if not family_dir.is_dir():
            continue
        stream_dir = family_dir / "stream"
        if stream_dir.exists():
            for chunk_file in sorted(stream_dir.glob("*.txt")):
                test_cases.append((family_dir.name, chunk_file.stem, chunk_file))
    return test_cases


@pytest.fixture(autouse=True)
def reset_processor():
    """Reset singleton between tests."""
    reset_response_processor()
    yield
    reset_response_processor()


class TestResponseProcessorToolCalls:
    """Test tool call extraction from golden files."""

    @pytest.mark.parametrize("family,golden_file", collect_tool_call_files())
    def test_tool_calls_extracted(self, family: str, golden_file: Path):
        """Verify tool calls are extracted from model family format."""
        text = golden_file.read_text()
        processor = get_response_processor()
        result = processor.process(text)

        # Tool calls should be extracted
        assert len(result.tool_calls) > 0, f"No tool calls found in {family}"

        # Each tool call should have name and arguments
        for tc in result.tool_calls:
            assert tc.function.name, f"Empty function name in {family}"
            assert tc.function.arguments, f"Empty arguments in {family}"

    @pytest.mark.parametrize("family,golden_file", collect_tool_call_files())
    def test_markers_removed_from_content(self, family: str, golden_file: Path):
        """Verify tool call markers removed from content."""
        text = golden_file.read_text()
        processor = get_response_processor()
        result = processor.process(text)

        # No raw markers in content
        assert "<tool_call>" not in result.content
        assert "</tool_call>" not in result.content
        assert "<function=" not in result.content
        assert "</function>" not in result.content
        assert "<|python_tag|>" not in result.content
        assert "<|eom_id|>" not in result.content

    @pytest.mark.parametrize("family,golden_file", collect_tool_call_files())
    def test_surrounding_text_preserved(self, family: str, golden_file: Path):
        """Verify text before and after tool call markers is preserved."""
        text = golden_file.read_text()
        processor = get_response_processor()
        result = processor.process(text)

        # Content should not be empty (surrounding text preserved)
        assert result.content.strip(), f"Content empty for {family}"

    def test_glm4_deduplication(self):
        """Verify GLM4 duplicate tool calls are deduplicated."""
        dup_file = GOLDEN_DIR / "glm4" / "duplicate_tools.txt"
        if not dup_file.exists():
            pytest.skip("GLM4 duplicate tools file not found")

        text = dup_file.read_text()
        processor = get_response_processor()
        result = processor.process(text)

        # Should have only 1 tool call (deduplicated)
        assert len(result.tool_calls) == 1

    def test_llama_python_tag(self):
        """Verify Llama Python tag format is parsed."""
        py_file = GOLDEN_DIR / "llama" / "python_tag.txt"
        if not py_file.exists():
            pytest.skip("Llama python tag file not found")

        text = py_file.read_text()
        processor = get_response_processor()
        result = processor.process(text)

        assert len(result.tool_calls) == 1
        # Python tag format: module.method -> function name
        assert "." in result.tool_calls[0].function.name


class TestResponseProcessorThinking:
    """Test thinking/reasoning extraction from golden files."""

    def test_thinking_extracted(self):
        """Verify thinking content is extracted."""
        think_file = GOLDEN_DIR / "qwen" / "thinking.txt"
        if not think_file.exists():
            pytest.skip("Thinking file not found")

        text = think_file.read_text()
        processor = get_response_processor()
        result = processor.process(text)

        # Reasoning should be extracted
        assert result.reasoning is not None
        assert "analyze" in result.reasoning.lower()

    def test_thinking_tags_removed(self):
        """Verify thinking tags are removed from content."""
        think_file = GOLDEN_DIR / "qwen" / "thinking.txt"
        if not think_file.exists():
            pytest.skip("Thinking file not found")

        text = think_file.read_text()
        processor = get_response_processor()
        result = processor.process(text)

        # No thinking tags in content
        assert "<think>" not in result.content
        assert "</think>" not in result.content
        assert "<thinking>" not in result.content
        assert "</thinking>" not in result.content
        assert "<reasoning>" not in result.content
        assert "</reasoning>" not in result.content


class TestStreamingProcessor:
    """Test streaming processor filters patterns correctly."""

    @pytest.mark.parametrize("family,format_name,chunk_file", collect_streaming_files())
    def test_streaming_filters_patterns(
        self, family: str, format_name: str, chunk_file: Path
    ):
        """Verify streaming filters patterns correctly."""
        chunks = chunk_file.read_text().splitlines()
        processor = StreamingProcessor()

        yielded: list[str] = []
        for chunk in chunks:
            output, should_yield = processor.feed(chunk)
            if should_yield and output:
                yielded.append(output)

        full_streamed = "".join(yielded)

        # Markers should not appear in streamed output
        assert "<think>" not in full_streamed
        assert "</think>" not in full_streamed
        assert "<tool_call>" not in full_streamed
        assert "</tool_call>" not in full_streamed
        assert "<function=" not in full_streamed
        assert "<|python_tag|>" not in full_streamed

    def test_streaming_extracts_on_finalize(self):
        """Verify finalize() returns complete ParseResult."""
        chunk_file = GOLDEN_DIR / "qwen" / "stream" / "tool_call_chunks.txt"
        if not chunk_file.exists():
            pytest.skip("Streaming chunks file not found")

        chunks = chunk_file.read_text().splitlines()
        processor = StreamingProcessor()

        for chunk in chunks:
            processor.feed(chunk)

        result = processor.finalize()

        # ParseResult should have extracted tool calls
        assert len(result.tool_calls) > 0
        # Content should be clean
        assert "<tool_call>" not in result.content

    def test_streaming_thinking_finalize(self):
        """Verify finalize() extracts reasoning from streamed thinking content."""
        chunk_file = GOLDEN_DIR / "qwen" / "stream" / "thinking_chunks.txt"
        if not chunk_file.exists():
            pytest.skip("Thinking chunks file not found")

        chunks = chunk_file.read_text().splitlines()
        processor = StreamingProcessor()

        for chunk in chunks:
            processor.feed(chunk)

        result = processor.finalize()

        # ParseResult should have extracted reasoning
        assert result.reasoning is not None
        # Content should not have thinking tags
        assert "<think>" not in result.content
        assert "</think>" not in result.content
