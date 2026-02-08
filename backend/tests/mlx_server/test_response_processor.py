"""Comprehensive tests for ResponseProcessor.

Tests cover:
- Pydantic model serialization
- Thinking tag extraction (single, multiple, mixed)
- Tool call extraction (Hermes, Llama, GLM4, Python style)
- Content cleaning (special tokens, whitespace normalization)
- Combined extraction (thinking + tool calls)
- Edge cases (empty, malformed, unicode)
- GLM4 deduplication
"""

import json

import pytest

from mlx_manager.mlx_server.schemas.openai import FunctionCall, ToolCall
from mlx_manager.mlx_server.services.response_processor import (
    GLM4_PATTERNS,
    LLAMA_PATTERNS,
    MODEL_FAMILY_PATTERNS,
    QWEN_PATTERNS,
    ModelFamilyPatterns,
    ParseResult,
    ResponseProcessor,
    StreamEvent,
    StreamingProcessor,
    ToolPatternSpec,
    create_default_processor,
    create_processor_for_family,
    get_processor_for_family,
    get_response_processor,
    reset_response_processor,
)

# --- Pydantic Model Tests ---


class TestPydanticModels:
    """Tests for Pydantic model behavior."""

    def test_tool_call_function_serialization(self) -> None:
        """FunctionCall serializes to dict correctly."""
        func = FunctionCall(name="get_weather", arguments='{"city": "SF"}')
        data = func.model_dump()

        assert data == {"name": "get_weather", "arguments": '{"city": "SF"}'}

    def test_tool_call_serialization(self) -> None:
        """ToolCall serializes to dict correctly."""
        tc = ToolCall(
            id="call_abc123",
            function=FunctionCall(name="search", arguments='{"q": "test"}'),
        )
        data = tc.model_dump()

        assert data["id"] == "call_abc123"
        assert data["type"] == "function"
        assert data["function"]["name"] == "search"
        assert data["function"]["arguments"] == '{"q": "test"}'

    def test_parse_result_defaults(self) -> None:
        """ParseResult has correct default values."""
        result = ParseResult(content="Hello")

        assert result.content == "Hello"
        assert result.tool_calls == []
        assert result.reasoning is None

    def test_parse_result_with_tool_calls(self) -> None:
        """ParseResult correctly holds tool calls."""
        tc = ToolCall(
            id="call_1",
            function=FunctionCall(name="test", arguments="{}"),
        )
        result = ParseResult(content="Result", tool_calls=[tc], reasoning="Thought")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "test"
        assert result.reasoning == "Thought"


# --- Thinking Tag Extraction Tests ---


class TestThinkingExtraction:
    """Tests for thinking/reasoning tag extraction."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_single_think_tag(self) -> None:
        """Extracts content from single <think> tag."""
        processor = get_response_processor()
        text = "<think>Let me analyze this</think>The answer is 42."

        result = processor.process(text)

        assert result.reasoning == "Let me analyze this"
        assert result.content == "The answer is 42."
        assert "<think>" not in result.content
        assert "</think>" not in result.content

    def test_thinking_tag(self) -> None:
        """Extracts content from <thinking> tag."""
        processor = get_response_processor()
        text = "<thinking>Deep thought here</thinking>Result is ready."

        result = processor.process(text)

        assert result.reasoning == "Deep thought here"
        assert result.content == "Result is ready."

    def test_reasoning_tag(self) -> None:
        """Extracts content from <reasoning> tag."""
        processor = get_response_processor()
        text = "<reasoning>Step by step</reasoning>Final answer: yes."

        result = processor.process(text)

        assert result.reasoning == "Step by step"
        assert result.content == "Final answer: yes."

    def test_reflection_tag(self) -> None:
        """Extracts content from <reflection> tag."""
        processor = get_response_processor()
        text = "<reflection>Reviewing my logic</reflection>Confirmed correct."

        result = processor.process(text)

        assert result.reasoning == "Reviewing my logic"
        assert result.content == "Confirmed correct."

    def test_multiple_thinking_tags(self) -> None:
        """Combines content from multiple thinking tags."""
        processor = get_response_processor()
        text = "<think>First thought</think>Middle<think>Second thought</think>End"

        result = processor.process(text)

        assert "First thought" in result.reasoning
        assert "Second thought" in result.reasoning
        assert result.content == "MiddleEnd"

    def test_mixed_tag_types(self) -> None:
        """Handles different thinking tag types in same response."""
        processor = get_response_processor()
        text = "<think>Thought</think>Text<reasoning>Logic</reasoning>More"

        result = processor.process(text)

        # Both should be combined
        assert "Thought" in result.reasoning
        assert "Logic" in result.reasoning
        assert result.content == "TextMore"

    def test_no_thinking_tags(self) -> None:
        """Returns None for reasoning when no tags present."""
        processor = get_response_processor()
        text = "Just regular content without thinking."

        result = processor.process(text)

        assert result.reasoning is None
        assert result.content == "Just regular content without thinking."

    def test_nested_content_preserved(self) -> None:
        """Preserves formatting within thinking tags."""
        processor = get_response_processor()
        text = "<think>Line 1\nLine 2\n- bullet</think>Answer"

        result = processor.process(text)

        assert "Line 1\nLine 2\n- bullet" in result.reasoning

    def test_case_insensitive_tags(self) -> None:
        """Handles mixed case thinking tags."""
        processor = get_response_processor()
        text = "<THINK>Uppercase</THINK>Result"

        result = processor.process(text)

        assert result.reasoning == "Uppercase"
        assert result.content == "Result"


# --- Tool Call Extraction Tests ---


class TestToolCallExtraction:
    """Tests for tool call extraction from various formats."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_hermes_tool_call(self) -> None:
        """Parses Hermes/Qwen style tool calls."""
        processor = get_response_processor()
        text = (
            "Here is the result: "
            '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'
        )

        result = processor.process(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "SF"}
        assert "tool_call" not in result.content
        assert result.content == "Here is the result:"

    def test_llama_xml_tool_call(self) -> None:
        """Parses Llama XML style tool calls."""
        processor = get_response_processor()
        text = 'Result: <function=search>{"query": "test", "limit": 10}</function>'

        result = processor.process(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "query": "test",
            "limit": 10,
        }
        assert "function=" not in result.content
        assert result.content == "Result:"

    def test_glm4_xml_tool_call(self) -> None:
        """Parses GLM4 nested XML style tool calls."""
        processor = get_response_processor()
        text = (
            '<tool_call><name>calculate</name><arguments>{"x": 5, "y": 3}</arguments></tool_call>'
        )

        result = processor.process(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "calculate"
        assert json.loads(result.tool_calls[0].function.arguments) == {"x": 5, "y": 3}
        assert result.content == ""

    def test_llama_python_tool_call(self) -> None:
        """Parses Llama Python style tool calls."""
        processor = get_response_processor()
        text = '<|python_tag|>weather.get(city="NYC", units="metric")<|eom_id|>'

        result = processor.process(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "weather.get"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["city"] == "NYC"
        assert args["units"] == "metric"

    def test_multiple_tool_calls(self) -> None:
        """Handles multiple tool calls in one response."""
        processor = get_response_processor()
        text = """I'll call two functions:
<function=func1>{"a": 1}</function>
<function=func2>{"b": 2}</function>
Done."""

        result = processor.process(text)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "func1"
        assert result.tool_calls[1].function.name == "func2"
        assert "I'll call two functions:" in result.content
        assert "Done." in result.content

    def test_tool_call_generates_unique_ids(self) -> None:
        """Each tool call gets a unique ID."""
        processor = get_response_processor()
        text = '<function=a>{"x": 1}</function><function=b>{"y": 2}</function>'

        result = processor.process(text)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id != result.tool_calls[1].id
        assert result.tool_calls[0].id.startswith("call_")
        assert result.tool_calls[1].id.startswith("call_")

    def test_invalid_json_still_extracted(self) -> None:
        """Tool calls with invalid JSON are still extracted."""
        processor = get_response_processor()
        text = "<function=test>{not valid json}</function>"

        result = processor.process(text)

        # Should still extract - validation is caller's responsibility
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "test"
        assert result.tool_calls[0].function.arguments == "{not valid json}"

    def test_glm4_deduplication(self) -> None:
        """GLM4 duplicate tool calls are deduplicated."""
        processor = get_response_processor()
        # GLM4 sometimes outputs the same tool call multiple times
        text = """<tool_call><name>search</name><arguments>{"q": "test"}</arguments></tool_call>
<tool_call><name>search</name><arguments>{"q": "test"}</arguments></tool_call>
<tool_call><name>search</name><arguments>{"q": "different"}</arguments></tool_call>"""

        result = processor.process(text)

        # Should only have 2 unique tool calls
        assert len(result.tool_calls) == 2
        args_list = [tc.function.arguments for tc in result.tool_calls]
        assert '{"q": "test"}' in args_list
        assert '{"q": "different"}' in args_list


# --- Content Cleaning Tests ---


class TestContentCleaning:
    """Tests for special token and whitespace cleanup."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_removes_special_tokens(self) -> None:
        """Removes common special tokens from content."""
        processor = get_response_processor()
        text = "Hello<|endoftext|>"

        result = processor.process(text)

        assert result.content == "Hello"

    def test_removes_im_tokens(self) -> None:
        """Removes IM start/end tokens."""
        processor = get_response_processor()
        text = "<|im_start|>Hello<|im_end|>"

        result = processor.process(text)

        assert result.content == "Hello"

    def test_removes_eot_id_token(self) -> None:
        """Removes Llama eot_id token."""
        processor = get_response_processor()
        text = "Response text<|eot_id|>"

        result = processor.process(text)

        assert result.content == "Response text"

    def test_removes_header_tokens(self) -> None:
        """Removes header tokens."""
        processor = get_response_processor()
        text = "<|start_header_id|>assistant<|end_header_id|>Hello"

        result = processor.process(text)

        assert "header" not in result.content.lower()
        assert "Hello" in result.content

    def test_normalizes_excessive_newlines(self) -> None:
        """Collapses multiple newlines to max of two."""
        processor = get_response_processor()
        text = "Line 1\n\n\n\n\nLine 2"

        result = processor.process(text)

        # Should have at most 2 newlines
        assert "\n\n\n" not in result.content
        assert "Line 1" in result.content
        assert "Line 2" in result.content

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Strips whitespace from start and end."""
        processor = get_response_processor()
        text = "   \n\nContent here\n\n   "

        result = processor.process(text)

        assert result.content == "Content here"


# --- Combined Extraction Tests ---


class TestCombinedExtraction:
    """Tests for responses with both thinking and tool calls."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_thinking_and_tool_calls(self) -> None:
        """Extracts both thinking content and tool calls."""
        processor = get_response_processor()
        text = (
            '<think>I need to search</think>Searching...<function=search>{"q": "test"}</function>'
        )

        result = processor.process(text)

        assert result.reasoning == "I need to search"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"
        assert result.content == "Searching..."

    def test_order_independent(self) -> None:
        """Order of thinking and tool calls doesn't matter."""
        processor = get_response_processor()
        text = '<function=a>{"x":1}</function><think>Thought</think>Text'

        result = processor.process(text)

        assert result.reasoning == "Thought"
        assert len(result.tool_calls) == 1
        assert result.content == "Text"

    def test_all_markers_removed(self) -> None:
        """All markers are removed from final content."""
        processor = get_response_processor()
        text = """<think>Planning</think>
Starting<|im_end|>
<function=test>{"a": 1}</function>
Done<|endoftext|>"""

        result = processor.process(text)

        # No markers should remain
        assert "<think>" not in result.content
        assert "</think>" not in result.content
        assert "<function=" not in result.content
        assert "</function>" not in result.content
        assert "<|" not in result.content
        assert "|>" not in result.content

        # Content should be clean
        assert "Starting" in result.content
        assert "Done" in result.content


# --- Edge Case Tests ---


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_empty_string(self) -> None:
        """Handles empty string input."""
        processor = get_response_processor()

        result = processor.process("")

        assert result.content == ""
        assert result.tool_calls == []
        assert result.reasoning is None

    def test_no_patterns_match(self) -> None:
        """Handles text with no matching patterns."""
        processor = get_response_processor()
        text = "Just plain text without any special markers."

        result = processor.process(text)

        assert result.content == "Just plain text without any special markers."
        assert result.tool_calls == []
        assert result.reasoning is None

    def test_unclosed_tags(self) -> None:
        """Handles unclosed/malformed tags gracefully."""
        processor = get_response_processor()
        text = "<think>Thought without closing tag"

        result = processor.process(text)

        # Should not extract unclosed tag as reasoning
        assert result.reasoning is None
        # Content should be preserved
        assert "<think>Thought without closing tag" in result.content

    def test_empty_tags(self) -> None:
        """Handles empty tags."""
        processor = get_response_processor()
        text = "<think></think>Content"

        result = processor.process(text)

        # Empty tags should be removed, no reasoning content
        assert result.reasoning is None
        assert result.content == "Content"

    def test_unicode_content(self) -> None:
        """Handles unicode content correctly."""
        processor = get_response_processor()
        text = "<think>Analyzing...</think>Result: 42"

        result = processor.process(text)

        assert result.reasoning == "Analyzing..."
        assert result.content == "Result: 42"

    def test_unicode_in_tool_call(self) -> None:
        """Handles unicode in tool call arguments."""
        processor = get_response_processor()
        text = '<function=translate>{"text": "hello", "to": "ja"}</function>'

        result = processor.process(text)

        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["to"] == "ja"

    def test_deeply_nested_json(self) -> None:
        """Handles deeply nested JSON in tool calls."""
        processor = get_response_processor()
        args = {"nested": {"deep": {"value": [1, 2, {"x": "y"}]}}}
        text = f"<function=complex>{json.dumps(args)}</function>"

        result = processor.process(text)

        assert len(result.tool_calls) == 1
        parsed_args = json.loads(result.tool_calls[0].function.arguments)
        assert parsed_args == args


# --- Processor Factory Tests ---


class TestProcessorFactory:
    """Tests for processor singleton and factory."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_singleton_returns_same_instance(self) -> None:
        """get_response_processor returns same instance."""
        p1 = get_response_processor()
        p2 = get_response_processor()

        assert p1 is p2

    def test_reset_clears_singleton(self) -> None:
        """reset_response_processor clears the singleton."""
        p1 = get_response_processor()
        reset_response_processor()
        p2 = get_response_processor()

        assert p1 is not p2

    def test_create_default_processor_returns_new_instance(self) -> None:
        """create_default_processor always returns new instance."""
        p1 = create_default_processor()
        p2 = create_default_processor()

        assert p1 is not p2

    def test_default_processor_has_all_patterns(self) -> None:
        """Default processor has all expected patterns registered."""
        processor = create_default_processor()

        # Test thinking tags
        result = processor.process("<think>x</think>y")
        assert result.reasoning == "x"

        # Test Hermes format
        result = processor.process('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        assert len(result.tool_calls) == 1

        # Test Llama format
        result = processor.process("<function=f>{}</function>")
        assert len(result.tool_calls) == 1

        # Test GLM4 format
        result = processor.process("<tool_call><name>f</name><arguments>{}</arguments></tool_call>")
        assert len(result.tool_calls) == 1


# --- Model Family Processor Tests ---


class TestModelFamilyProcessors:
    """Tests for model-family-specific processors."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_qwen_family_patterns_defined(self) -> None:
        """Qwen patterns are properly defined."""
        assert "qwen" in MODEL_FAMILY_PATTERNS
        patterns = QWEN_PATTERNS
        assert len(patterns.tool_patterns) >= 2  # At least closed and unclosed formats
        assert len(patterns.thinking_tags) >= 1

    def test_glm4_family_patterns_defined(self) -> None:
        """GLM4 patterns are properly defined."""
        assert "glm4" in MODEL_FAMILY_PATTERNS
        patterns = GLM4_PATTERNS
        assert len(patterns.tool_patterns) >= 3  # XML, compact, attr formats
        assert len(patterns.thinking_tags) >= 1

    def test_llama_family_patterns_defined(self) -> None:
        """Llama patterns are properly defined."""
        assert "llama" in MODEL_FAMILY_PATTERNS
        patterns = LLAMA_PATTERNS
        assert len(patterns.tool_patterns) >= 2  # XML and Python formats
        assert len(patterns.thinking_tags) >= 1

    def test_create_processor_for_qwen(self) -> None:
        """create_processor_for_family creates Qwen-specific processor."""
        processor = create_processor_for_family("qwen")

        # Should match Hermes/Qwen JSON format
        result = processor.process('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        assert len(result.tool_calls) == 1

        # Should NOT match GLM4 XML format (family-specific)
        result = processor.process("<tool_call><name>f</name><arguments>{}</arguments></tool_call>")
        assert len(result.tool_calls) == 0  # Pattern not registered for Qwen

    def test_create_processor_for_glm4(self) -> None:
        """create_processor_for_family creates GLM4-specific processor."""
        processor = create_processor_for_family("glm4")

        # Should match GLM4 XML format
        result = processor.process("<tool_call><name>f</name><arguments>{}</arguments></tool_call>")
        assert len(result.tool_calls) == 1

        # Should NOT match Hermes JSON format (family-specific)
        result = processor.process('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        assert len(result.tool_calls) == 0  # Pattern not registered for GLM4

    def test_create_processor_for_llama(self) -> None:
        """create_processor_for_family creates Llama-specific processor."""
        processor = create_processor_for_family("llama")

        # Should match Llama function format
        result = processor.process("<function=f>{}</function>")
        assert len(result.tool_calls) == 1

        # Should NOT match Hermes JSON format (family-specific)
        result = processor.process('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        assert len(result.tool_calls) == 0  # Pattern not registered for Llama

    def test_get_processor_for_family_caches_instance(self) -> None:
        """get_processor_for_family caches processors per family."""
        p1 = get_processor_for_family("qwen")
        p2 = get_processor_for_family("qwen")
        p3 = get_processor_for_family("glm4")

        assert p1 is p2  # Same family returns same instance
        assert p1 is not p3  # Different families return different instances

    def test_get_processor_for_family_case_insensitive(self) -> None:
        """get_processor_for_family is case-insensitive."""
        p1 = get_processor_for_family("Qwen")
        p2 = get_processor_for_family("QWEN")
        p3 = get_processor_for_family("qwen")

        assert p1 is p2
        assert p2 is p3

    def test_unknown_family_uses_default_patterns(self) -> None:
        """Unknown family falls back to default patterns (all patterns)."""
        processor = create_processor_for_family("unknown_model")

        # Should match multiple formats since default has all patterns
        result = processor.process('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        assert len(result.tool_calls) == 1

        processor2 = create_processor_for_family("unknown_model")
        result2 = processor2.process("<function=f>{}</function>")
        assert len(result2.tool_calls) == 1

    def test_streaming_processor_with_model_family(self) -> None:
        """StreamingProcessor accepts model_family parameter."""
        # Qwen processor - should extract Hermes format
        sp_qwen = StreamingProcessor(model_family="qwen")
        sp_qwen.feed("<tool_call>")
        sp_qwen.feed('{"name": "test", "arguments": {}}')
        sp_qwen.feed("</tool_call>")
        result = sp_qwen.finalize()
        assert len(result.tool_calls) == 1

    def test_streaming_processor_family_specificity(self) -> None:
        """StreamingProcessor uses family-specific patterns."""
        # GLM4 processor should NOT extract Hermes JSON format
        sp_glm4 = StreamingProcessor(model_family="glm4")
        sp_glm4.feed("<tool_call>")
        sp_glm4.feed('{"name": "test", "arguments": {}}')
        sp_glm4.feed("</tool_call>")
        result = sp_glm4.finalize()
        # GLM4 expects XML inside <tool_call>, not JSON
        assert len(result.tool_calls) == 0

    def test_reset_clears_family_caches(self) -> None:
        """reset_response_processor clears family processor caches."""
        p1 = get_processor_for_family("qwen")
        reset_response_processor()
        p2 = get_processor_for_family("qwen")

        assert p1 is not p2


class TestToolPatternSpec:
    """Tests for ToolPatternSpec dataclass."""

    def test_tool_pattern_spec_defaults(self) -> None:
        """ToolPatternSpec has sensible defaults."""
        spec = ToolPatternSpec(pattern=r"<test>", parser_name="hermes")

        assert spec.pattern == r"<test>"
        assert spec.parser_name == "hermes"
        assert spec.description == ""

    def test_tool_pattern_spec_with_description(self) -> None:
        """ToolPatternSpec accepts description."""
        spec = ToolPatternSpec(
            pattern=r"<test>",
            parser_name="hermes",
            description="Test pattern",
        )

        assert spec.description == "Test pattern"


class TestModelFamilyPatterns:
    """Tests for ModelFamilyPatterns dataclass."""

    def test_model_family_patterns_defaults(self) -> None:
        """ModelFamilyPatterns has empty defaults."""
        patterns = ModelFamilyPatterns()

        assert patterns.tool_patterns == []
        assert patterns.thinking_tags == []
        assert patterns.cleanup_tokens == []

    def test_model_family_patterns_with_data(self) -> None:
        """ModelFamilyPatterns accepts full configuration."""
        spec = ToolPatternSpec(pattern=r"<test>", parser_name="hermes")
        patterns = ModelFamilyPatterns(
            tool_patterns=[spec],
            thinking_tags=["think"],
            cleanup_tokens=["<|end|>"],
        )

        assert len(patterns.tool_patterns) == 1
        assert patterns.thinking_tags == ["think"]
        assert patterns.cleanup_tokens == ["<|end|>"]


# --- Custom Processor Tests ---


class TestCustomProcessor:
    """Tests for building custom processors."""

    def test_register_custom_thinking_tag(self) -> None:
        """Can register custom thinking tag."""
        processor = ResponseProcessor()
        processor.register_thinking_tags(["custom"])

        result = processor.process("<custom>My thoughts</custom>Output")

        assert result.reasoning == "My thoughts"
        assert result.content == "Output"

    def test_register_custom_cleanup_pattern(self) -> None:
        """Can register custom cleanup patterns."""
        processor = ResponseProcessor()
        processor.register_cleanup_patterns(["[STOP]", "[END]"])

        result = processor.process("Hello[STOP] world[END]!")

        assert result.content == "Hello world!"

    def test_processor_without_patterns(self) -> None:
        """Processor with no patterns passes through text."""
        processor = ResponseProcessor()

        result = processor.process("<think>test</think>content")

        # No patterns registered, so nothing extracted
        assert result.content == "<think>test</think>content"
        assert result.reasoning is None
        assert result.tool_calls == []


# --- Streaming Processor Tests ---


class TestStreamingProcessorBasic:
    """Basic tests for StreamingProcessor with StreamEvent return type."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_streaming_passthrough_no_patterns(self) -> None:
        """Normal text passes through as content."""
        processor = StreamingProcessor()
        tokens = ["Hello", " ", "world", "!"]
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)

        assert "".join(content_parts) == "Hello world!"

    def test_streaming_think_tags_as_reasoning(self) -> None:
        """Think tags yield reasoning_content, not content."""
        processor = StreamingProcessor()
        tokens = ["Hello", "<think>", "analyzing", "</think>", " world"]
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)
            if event.reasoning_content:
                reasoning_parts.append(event.reasoning_content)

        # Thinking content goes to reasoning_content
        assert "analyzing" in "".join(reasoning_parts)
        # Regular content goes to content
        content = "".join(content_parts)
        assert "Hello" in content
        assert "world" in content
        # No thinking tags in content
        assert "<think>" not in content
        assert "analyzing" not in content

    def test_streaming_filters_tool_call_tags(self) -> None:
        """Tool call tags are filtered (not in content or reasoning)."""
        processor = StreamingProcessor()
        tokens = [
            "Result: ",
            "<tool_call>",
            '{"name": "search", "arguments": {}}',
            "</tool_call>",
            " Done",
        ]
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)
            if event.reasoning_content:
                reasoning_parts.append(event.reasoning_content)

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        assert "<tool_call>" not in content
        assert "</tool_call>" not in content
        assert "Result:" in content
        assert "Done" in content
        # Tool content not in reasoning either
        assert "search" not in reasoning

    def test_streaming_filters_function_tags(self) -> None:
        """Llama function tags are filtered from stream."""
        processor = StreamingProcessor()
        tokens = ["OK ", "<function=test>", '{"x": 1}', "</function>", " bye"]
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)

        content = "".join(content_parts)
        assert "<function=" not in content
        assert "</function>" not in content
        assert "OK" in content
        assert "bye" in content


class TestStreamingProcessorPartialMarkers:
    """Tests for partial marker detection across tokens."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_partial_think_tag_across_tokens(self) -> None:
        """Handles <think> split across multiple tokens."""
        processor = StreamingProcessor()
        tokens = ["Hello ", "<", "think", ">", "thought", "</think>", " end"]
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)
            if event.reasoning_content:
                reasoning_parts.append(event.reasoning_content)

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        # Content should have surrounding text
        assert "Hello" in content
        assert "end" in content
        # Thinking content goes to reasoning
        assert "thought" in reasoning
        # No tags in content
        assert "<think>" not in content

    def test_partial_tool_call_across_tokens(self) -> None:
        """Handles <tool_call> split across tokens."""
        processor = StreamingProcessor()
        tokens = ["Start ", "<tool", "_call>", '{"name": "f", "arguments": {}}', "</tool_call>"]
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)

        content = "".join(content_parts)
        assert "<tool" not in content
        assert "_call>" not in content
        assert "Start" in content

    def test_partial_function_across_tokens(self) -> None:
        """Handles <function= split across tokens."""
        processor = StreamingProcessor()
        tokens = ["Hi ", "<func", "tion=", "test>", "{}", "</function>", " bye"]
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)

        content = "".join(content_parts)
        assert "<function" not in content
        assert "Hi" in content
        assert "bye" in content

    def test_false_partial_marker(self) -> None:
        """Tokens that look like partial markers but aren't are yielded."""
        processor = StreamingProcessor()
        # '<' followed by something that isn't a pattern start
        tokens = ["Compare ", "<", "5", " ", ">", " 3"]
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)

        # After finalize, pending buffer should be flushed
        processor.finalize()
        content = "".join(content_parts)
        # Should include 'Compare ' and eventually the comparison
        assert "Compare" in content


class TestStreamingProcessorMultiplePatterns:
    """Tests for responses with multiple patterns."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_multiple_think_tags(self) -> None:
        """Handles multiple thinking sections with reasoning_content."""
        processor = StreamingProcessor()
        tokens = [
            "<think>",
            "first",
            "</think>",
            "A",
            "<think>",
            "second",
            "</think>",
            "B",
        ]
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)
            if event.reasoning_content:
                reasoning_parts.append(event.reasoning_content)

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        # Thinking content in reasoning
        assert "first" in reasoning
        assert "second" in reasoning
        # Regular content in content
        assert "A" in content
        assert "B" in content
        # No thinking in content
        assert "first" not in content
        assert "second" not in content

    def test_thinking_and_tool_call(self) -> None:
        """Handles both thinking and tool call in same response."""
        processor = StreamingProcessor()
        tokens = [
            "<think>",
            "planning",
            "</think>",
            "Calling: ",
            "<tool_call>",
            '{"name": "f", "arguments": {}}',
            "</tool_call>",
            " done",
        ]
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)
            if event.reasoning_content:
                reasoning_parts.append(event.reasoning_content)

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        # Planning goes to reasoning
        assert "planning" in reasoning
        # Tool call tags filtered from content
        assert "<tool_call>" not in content
        # Regular content preserved
        assert "Calling:" in content
        assert "done" in content


class TestStreamingProcessorFinalize:
    """Tests for finalize() method."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_finalize_extracts_reasoning(self) -> None:
        """finalize() extracts reasoning from accumulated text."""
        processor = StreamingProcessor()
        tokens = ["<think>", "my thoughts", "</think>", "answer"]

        for token in tokens:
            processor.feed(token)

        result = processor.finalize()
        assert result.reasoning == "my thoughts"
        assert result.content == "answer"

    def test_finalize_extracts_tool_calls(self) -> None:
        """finalize() extracts tool calls from accumulated text."""
        processor = StreamingProcessor()
        tokens = ['<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>']

        for token in tokens:
            processor.feed(token)

        result = processor.finalize()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"

    def test_finalize_extracts_both(self) -> None:
        """finalize() extracts both reasoning and tool calls."""
        processor = StreamingProcessor()
        tokens = [
            "<think>",
            "Let me search",
            "</think>",
            '<function=search>{"q": "test"}</function>',
        ]

        for token in tokens:
            processor.feed(token)

        result = processor.finalize()
        assert result.reasoning == "Let me search"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"

    def test_finalize_content_is_clean(self) -> None:
        """finalize() returns clean content without markers."""
        processor = StreamingProcessor()
        tokens = [
            "Hello ",
            "<think>",
            "thinking",
            "</think>",
            "World",
            "<|endoftext|>",
        ]

        for token in tokens:
            processor.feed(token)

        result = processor.finalize()
        assert "<think>" not in result.content
        assert "</think>" not in result.content
        assert "<|endoftext|>" not in result.content
        assert "Hello" in result.content
        assert "World" in result.content


class TestStreamingProcessorEdgeCases:
    """Edge case tests for StreamingProcessor."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset processor before each test."""
        reset_response_processor()

    def test_empty_tokens(self) -> None:
        """Handles empty token strings."""
        processor = StreamingProcessor()
        tokens = ["Hello", "", " ", "", "world"]
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)

        content = "".join(content_parts)
        assert "Hello" in content
        assert "world" in content

    def test_pattern_split_many_tokens(self) -> None:
        """Handles pattern split across many tokens."""
        processor = StreamingProcessor()
        # <tool_call> split char by char
        tokens = ["Start ", "<", "t", "o", "o", "l", "_", "c", "a", "l", "l", ">"]
        tokens += ['{"name": "f", "arguments": {}}', "</tool_call>", " end"]
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)

        content = "".join(content_parts)
        assert "Start" in content
        assert "end" in content
        assert "<tool_call>" not in content

    def test_no_patterns_passthrough(self) -> None:
        """Text without patterns passes through completely."""
        processor = StreamingProcessor()
        tokens = ["Just", " normal", " text", " here", "."]
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)

        assert "".join(content_parts) == "Just normal text here."

    def test_get_accumulated_text(self) -> None:
        """get_accumulated_text() returns all text including patterns."""
        processor = StreamingProcessor()
        tokens = ["Hello", "<think>", "thought", "</think>", "world"]

        for token in tokens:
            processor.feed(token)

        accumulated = processor.get_accumulated_text()
        assert "Hello" in accumulated
        assert "<think>" in accumulated
        assert "thought" in accumulated
        assert "</think>" in accumulated
        assert "world" in accumulated

    def test_get_pending_content(self) -> None:
        """get_pending_content() returns buffered content."""
        processor = StreamingProcessor()

        # Feed partial marker
        processor.feed("Hello ")
        processor.feed("<tool")  # Partial marker

        pending = processor.get_pending_content()
        assert "<tool" in pending

    def test_python_tag_pattern(self) -> None:
        """Filters <|python_tag|>...<|eom_id|> pattern."""
        processor = StreamingProcessor()
        tokens = [
            "Result: ",
            "<|python_tag|>",
            "module.func(x=1)",
            "<|eom_id|>",
            " done",
        ]
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.content:
                content_parts.append(event.content)

        content = "".join(content_parts)
        assert "<|python_tag|>" not in content
        assert "<|eom_id|>" not in content
        assert "Result:" in content
        assert "done" in content

    def test_nested_think_tags_filtered(self) -> None:
        """Filters nested <think> tags from reasoning content (Qwen3 with tools bug)."""
        processor = StreamingProcessor()
        # Real-world case: Qwen3 outputs <think><think> when tools are present
        tokens = [
            "<think>",
            "\n",
            "<think>",
            "\n",
            "Thinking about tools...",
            "\n",
            "</think>",
            "\n",
            "Response",
        ]
        reasoning_parts = []
        content_parts = []

        for token in tokens:
            event = processor.feed(token)
            if event.reasoning_content:
                reasoning_parts.append(event.reasoning_content)
            if event.content:
                content_parts.append(event.content)

        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)

        # Nested <think> should be filtered from reasoning
        assert "<think>" not in reasoning
        # Actual thinking content preserved
        assert "Thinking about tools" in reasoning
        # Content after </think> preserved
        assert "Response" in content

    def test_unclosed_tool_call_extracted(self) -> None:
        """Extracts tool calls without closing </tool_call> tag (Qwen3 behavior)."""
        processor = StreamingProcessor()
        # Real-world case: Qwen3 outputs tool call without closing tag
        text = '<tool_call>{"name": "get_weather", "arguments": {"location": "Bologna"}}'
        tokens = text.split()  # Simple tokenization

        for token in tokens:
            processor.feed(token)

        result = processor.finalize()

        # Tool call should be extracted even without closing tag
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert "Bologna" in result.tool_calls[0].function.arguments

    def test_starts_in_thinking_mode(self) -> None:
        """Handles prompts that end with <think> (GLM-4.7 behavior)."""
        # GLM-4.7 prompt ends with <think>, so model output is already inside thinking
        processor = StreamingProcessor(starts_in_thinking=True)
        # Model output (no opening <think>, continues from prompt)
        text = "Thinking about the question.</think>The answer is 42."
        reasoning_parts = []
        content_parts = []

        for char in text:
            event = processor.feed(char)
            if event.reasoning_content:
                reasoning_parts.append(event.reasoning_content)
            if event.content:
                content_parts.append(event.content)

        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)

        # Reasoning should have the thinking content
        assert "Thinking about" in reasoning
        # Content should have the answer
        assert "42" in content
        # No <think> tags in either
        assert "<think>" not in reasoning
        assert "</think>" not in content

    def test_glm4_compact_tool_call(self) -> None:
        """Extracts GLM-4.7 compact tool format: <tool_call>func<param>val</param>"""
        processor = StreamingProcessor(starts_in_thinking=True)
        # GLM-4.7 output format (prompt ends with <think>)
        text = "Thinking.</think><tool_call>get_weather<location>Rome</location>"

        for char in text:
            processor.feed(char)

        result = processor.finalize()

        # Tool call should be extracted
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert "Rome" in result.tool_calls[0].function.arguments


class TestStreamEventDataclass:
    """Tests for StreamEvent dataclass."""

    def test_stream_event_defaults(self) -> None:
        """StreamEvent has correct default values."""
        event = StreamEvent()

        assert event.content is None
        assert event.reasoning_content is None
        assert event.is_complete is False

    def test_stream_event_with_content(self) -> None:
        """StreamEvent with content only."""
        event = StreamEvent(content="Hello")

        assert event.content == "Hello"
        assert event.reasoning_content is None
        assert event.is_complete is False

    def test_stream_event_with_reasoning(self) -> None:
        """StreamEvent with reasoning_content only."""
        event = StreamEvent(reasoning_content="Thinking...")

        assert event.content is None
        assert event.reasoning_content == "Thinking..."
        assert event.is_complete is False

    def test_stream_event_complete(self) -> None:
        """StreamEvent with is_complete flag."""
        event = StreamEvent(reasoning_content="Final thought", is_complete=True)

        assert event.reasoning_content == "Final thought"
        assert event.is_complete is True

    def test_stream_event_both_fields(self) -> None:
        """StreamEvent can have both content and reasoning_content."""
        event = StreamEvent(content="Result", reasoning_content="Thought")

        assert event.content == "Result"
        assert event.reasoning_content == "Thought"


class TestFamilyAwareStreamingProcessor:
    """Tests for family-aware streaming pattern configuration.

    Verifies that StreamingProcessor uses family-specific markers
    derived from ModelFamilyPatterns (P2: adapter-driven, P5: shared infra).
    """

    @pytest.fixture(autouse=True)
    def reset(self) -> None:
        """Reset singleton cache before each test."""
        reset_response_processor()

    def test_qwen_processor_detects_tool_call_markers(self) -> None:
        """Qwen StreamingProcessor detects <tool_call> markers."""
        sp = StreamingProcessor(model_family="qwen")

        # Feed a tool call pattern
        events = []
        for token in ["Hello ", "<tool_call>", '{"name":"f","arguments":{}}', "</tool_call>"]:
            event = sp.feed(token)
            events.append(event)

        result = sp.finalize()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "f"

    def test_qwen_processor_does_not_detect_llama_markers(self) -> None:
        """Qwen StreamingProcessor ignores <function= markers (Llama-only)."""
        sp = StreamingProcessor(model_family="qwen")

        # Feed a Llama-style function call - should be treated as content
        for token in ['<function=test>{"a":1}', "</function>"]:
            sp.feed(token)

        result = sp.finalize()
        # No tool calls extracted (Qwen processor doesn't have Llama patterns)
        assert len(result.tool_calls) == 0

    def test_llama_processor_detects_function_markers(self) -> None:
        """Llama StreamingProcessor detects <function= markers."""
        sp = StreamingProcessor(model_family="llama")

        for token in ["<function=", "test>", '{"a":1}', "</function>"]:
            sp.feed(token)

        result = sp.finalize()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "test"

    def test_llama_processor_does_not_detect_glm4_markers(self) -> None:
        """Llama StreamingProcessor ignores GLM4 XML format."""
        sp = StreamingProcessor(model_family="llama")

        text = "<tool_call><name>f</name><arguments>{}</arguments></tool_call>"
        for char_chunk in [text[i : i + 5] for i in range(0, len(text), 5)]:
            sp.feed(char_chunk)

        result = sp.finalize()
        # No tool calls (Llama processor doesn't have GLM4 patterns)
        assert len(result.tool_calls) == 0

    def test_default_processor_detects_all_markers(self) -> None:
        """Default StreamingProcessor detects all tool marker types."""
        sp = StreamingProcessor()  # No family = default

        # Feed Hermes-style tool call
        for token in ["<tool_call>", '{"name":"f","arguments":{}}', "</tool_call>"]:
            sp.feed(token)

        result = sp.finalize()
        assert len(result.tool_calls) == 1

    def test_family_aware_thinking_tags(self) -> None:
        """Family-aware processor detects thinking tags from family patterns."""
        sp = StreamingProcessor(model_family="qwen")

        events = []
        for token in ["<think>", "My reasoning", "</think>", "Answer"]:
            events.append(sp.feed(token))

        # Should have streamed reasoning_content
        reasoning_events = [e for e in events if e.reasoning_content]
        assert len(reasoning_events) > 0

    def test_model_family_patterns_streaming_markers(self) -> None:
        """ModelFamilyPatterns derives correct streaming markers."""
        assert QWEN_PATTERNS.get_thinking_starts() == [
            "<think>",
            "<thinking>",
            "<reasoning>",
            "<reflection>",
        ]
        assert "<tool_call>" in [m[0] for m in QWEN_PATTERNS.streaming_tool_markers]
        assert "<function=" in [m[0] for m in LLAMA_PATTERNS.streaming_tool_markers]


# --- Parser Function Edge Case Tests ---


class TestParserFunctions:
    """Direct tests for parser functions to cover edge cases."""

    def test_parse_hermes_tool_string_arguments(self) -> None:
        """parse_hermes_tool handles string arguments (not dict)."""
        import re

        from mlx_manager.mlx_server.services.response_processor import (
            parse_hermes_tool,
        )

        # Arguments as a string instead of dict
        text = '<tool_call>{"name": "test", "arguments": "raw_string"}</tool_call>'
        pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        match = pattern.search(text)
        assert match is not None

        result = parse_hermes_tool(match)

        assert result is not None
        assert result.function.name == "test"
        assert result.function.arguments == "raw_string"

    def test_parse_hermes_tool_invalid_json(self) -> None:
        """parse_hermes_tool returns None for invalid JSON."""
        import re

        from mlx_manager.mlx_server.services.response_processor import (
            parse_hermes_tool,
        )

        text = "<tool_call>{not valid json}</tool_call>"
        pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        match = pattern.search(text)
        assert match is not None

        result = parse_hermes_tool(match)

        assert result is None

    def test_parse_llama_tool_invalid_json_still_returns(self) -> None:
        """parse_llama_tool returns ToolCall even with invalid JSON args."""
        import re

        from mlx_manager.mlx_server.services.response_processor import (
            parse_llama_tool,
        )

        text = "<function=test>{not valid}</function>"
        pattern = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
        match = pattern.search(text)
        assert match is not None

        result = parse_llama_tool(match)

        assert result is not None
        assert result.function.name == "test"
        assert result.function.arguments == "{not valid}"

    def test_parse_llama_tool_error_handling(self) -> None:
        """parse_llama_tool returns None on IndexError/AttributeError."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.services.response_processor import (
            parse_llama_tool,
        )

        mock_match = MagicMock()
        mock_match.group.side_effect = IndexError("no group")

        result = parse_llama_tool(mock_match)
        assert result is None

    def test_parse_glm4_tool_invalid_json_still_returns(self) -> None:
        """parse_glm4_tool returns ToolCall even with invalid JSON args."""
        import re

        from mlx_manager.mlx_server.services.response_processor import (
            parse_glm4_tool,
        )

        text = "<tool_call><name>test</name><arguments>{bad json}</arguments></tool_call>"
        pattern = re.compile(
            r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
            re.DOTALL,
        )
        match = pattern.search(text)
        assert match is not None

        result = parse_glm4_tool(match)

        assert result is not None
        assert result.function.name == "test"
        assert result.function.arguments == "{bad json}"

    def test_parse_glm4_tool_error_handling(self) -> None:
        """parse_glm4_tool returns None on IndexError/AttributeError."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.services.response_processor import (
            parse_glm4_tool,
        )

        mock_match = MagicMock()
        mock_match.group.side_effect = IndexError("no group")

        result = parse_glm4_tool(mock_match)
        assert result is None

    def test_parse_glm4_compact_tool_no_name(self) -> None:
        """parse_glm4_compact_tool returns None when no name found."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.services.response_processor import (
            parse_glm4_compact_tool,
        )

        mock_match = MagicMock()
        mock_match.group.return_value = "   "  # Only whitespace, no name

        result = parse_glm4_compact_tool(mock_match)
        assert result is None

    def test_parse_glm4_compact_tool_error_handling(self) -> None:
        """parse_glm4_compact_tool returns None on error."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.services.response_processor import (
            parse_glm4_compact_tool,
        )

        mock_match = MagicMock()
        mock_match.group.side_effect = IndexError("no group")

        result = parse_glm4_compact_tool(mock_match)
        assert result is None

    def test_parse_glm4_attr_tool_no_name(self) -> None:
        """parse_glm4_attr_tool returns None when no name found."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.services.response_processor import (
            parse_glm4_attr_tool,
        )

        mock_match = MagicMock()
        mock_match.group.return_value = "   "  # Only whitespace

        result = parse_glm4_attr_tool(mock_match)
        assert result is None

    def test_parse_glm4_attr_tool_no_params(self) -> None:
        """parse_glm4_attr_tool returns None when no attr params found."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.services.response_processor import (
            parse_glm4_attr_tool,
        )

        mock_match = MagicMock()
        mock_match.group.return_value = "func_name"  # Name but no attr params

        result = parse_glm4_attr_tool(mock_match)
        assert result is None

    def test_parse_glm4_attr_tool_error_handling(self) -> None:
        """parse_glm4_attr_tool returns None on error."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.services.response_processor import (
            parse_glm4_attr_tool,
        )

        mock_match = MagicMock()
        mock_match.group.side_effect = IndexError("no group")

        result = parse_glm4_attr_tool(mock_match)
        assert result is None

    def test_parse_llama_python_tool_error_handling(self) -> None:
        """parse_llama_python_tool returns None on error."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.services.response_processor import (
            parse_llama_python_tool,
        )

        mock_match = MagicMock()
        mock_match.group.side_effect = IndexError("no group")

        result = parse_llama_python_tool(mock_match)
        assert result is None


class TestParsePythonArgs:
    """Tests for _parse_python_args helper function."""

    def test_empty_string(self) -> None:
        """Empty string returns empty dict."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        assert _parse_python_args("") == {}

    def test_string_value_double_quotes(self) -> None:
        """Parse string value with double quotes."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        result = _parse_python_args('city="New York"')
        assert result == {"city": "New York"}

    def test_string_value_single_quotes(self) -> None:
        """Parse string value with single quotes."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        result = _parse_python_args("city='London'")
        assert result == {"city": "London"}

    def test_integer_value(self) -> None:
        """Parse integer value."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        result = _parse_python_args("limit=10")
        assert result == {"limit": 10}
        assert isinstance(result["limit"], int)

    def test_float_value(self) -> None:
        """Parse float value."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        result = _parse_python_args("temp=3.14")
        assert result == {"temp": 3.14}
        assert isinstance(result["temp"], float)

    def test_boolean_true(self) -> None:
        """Parse boolean True value."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        result = _parse_python_args("verbose=True")
        assert result == {"verbose": True}
        assert isinstance(result["verbose"], bool)

    def test_boolean_false(self) -> None:
        """Parse boolean False value."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        result = _parse_python_args("debug=False")
        assert result == {"debug": False}
        assert isinstance(result["debug"], bool)

    def test_bare_identifier_value(self) -> None:
        """Parse bare identifier as string value."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        result = _parse_python_args("mode=fast")
        assert result == {"mode": "fast"}

    def test_multiple_args(self) -> None:
        """Parse multiple arguments."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        result = _parse_python_args('query="hello", limit=5, verbose=True')
        assert result == {"query": "hello", "limit": 5, "verbose": True}

    def test_float_non_numeric_with_dot(self) -> None:
        """Parse value with dot that is not a valid float."""
        from mlx_manager.mlx_server.services.response_processor import (
            _parse_python_args,
        )

        # The regex won't match "not.a.number" as it expects \d+\.\d+
        # So this specific case won't be in results
        result = _parse_python_args('x="1.2.3"')
        assert result == {"x": "1.2.3"}


class TestModelFamilyPatternsHelpers:
    """Tests for ModelFamilyPatterns helper methods."""

    def test_get_all_pattern_starts(self) -> None:
        """get_all_pattern_starts returns thinking + tool starts."""
        patterns = ModelFamilyPatterns(
            thinking_tags=["think", "reasoning"],
            streaming_tool_markers=[("<tool_call>", "</tool_call>")],
        )

        starts = patterns.get_all_pattern_starts()

        assert "<think>" in starts
        assert "<reasoning>" in starts
        assert "<tool_call>" in starts

    def test_get_all_pattern_ends(self) -> None:
        """get_all_pattern_ends returns thinking + tool end mappings."""
        patterns = ModelFamilyPatterns(
            thinking_tags=["think"],
            streaming_tool_markers=[("<tool_call>", "</tool_call>")],
        )

        ends = patterns.get_all_pattern_ends()

        assert ends["<think>"] == "</think>"
        assert ends["<tool_call>"] == "</tool_call>"


class TestProcessorFromPatternsEdgeCases:
    """Tests for edge cases in processor factory functions."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        reset_response_processor()

    def test_create_processor_from_patterns_unknown_parser(self) -> None:
        """Unknown parser name in pattern spec is skipped with warning."""
        from mlx_manager.mlx_server.services.response_processor import (
            create_processor_from_patterns,
        )

        patterns = ModelFamilyPatterns(
            tool_patterns=[
                ToolPatternSpec(
                    pattern=r"<test>(.*?)</test>",
                    parser_name="nonexistent_parser",
                    description="Test pattern with unknown parser",
                ),
            ],
            thinking_tags=[],
            cleanup_tokens=[],
        )

        processor = create_processor_from_patterns(patterns)

        # Should not crash, just skip the unknown pattern
        result = processor.process("<test>content</test>")
        # Pattern not registered, so no extraction
        assert result.tool_calls == []
        assert "<test>content</test>" in result.content

    def test_create_processor_for_adapter(self) -> None:
        """create_processor_for_adapter uses adapter.family."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.services.response_processor import (
            create_processor_for_adapter,
        )

        mock_adapter = MagicMock()
        mock_adapter.family = "qwen"

        processor = create_processor_for_adapter(mock_adapter)

        # Should work like a Qwen processor
        result = processor.process('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
        assert len(result.tool_calls) == 1


class TestStreamingProcessorCustomRP:
    """Tests for StreamingProcessor with custom ResponseProcessor."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        reset_response_processor()

    def test_streaming_with_custom_response_processor(self) -> None:
        """StreamingProcessor accepts a custom ResponseProcessor."""
        custom_rp = create_default_processor()
        sp = StreamingProcessor(response_processor=custom_rp)

        sp.feed("Hello")
        result = sp.finalize()

        assert result.content == "Hello"

    def test_finalize_flushes_pending_buffer(self) -> None:
        """finalize() flushes pending buffer from incomplete markers."""
        sp = StreamingProcessor()
        sp.feed("Hello ")
        sp.feed("<tool")  # Partial marker - goes to pending buffer

        result = sp.finalize()

        # The finalize should process the full accumulated text
        assert "Hello" in result.content

    def test_handle_pattern_start_with_before_and_after_thinking(self) -> None:
        """_handle_pattern_start with content before and after a complete thinking tag."""
        sp = StreamingProcessor()

        # Feed text that has before + complete thinking pattern in one chunk
        event = sp.feed("Hello<think>thought</think>World")

        # Should get both content and reasoning in this event
        assert event.content is not None
        # The event might contain "Hello" as content
        assert "Hello" in (event.content or "")

        result = sp.finalize()
        assert result.reasoning == "thought"
        assert "World" in result.content

    def test_handle_pattern_start_before_and_after_tool(self) -> None:
        """_handle_pattern_start with content before + complete tool pattern + after."""
        sp = StreamingProcessor()

        # Feed text with before + tool + after all in one token
        sp.feed('Before<tool_call>{"name":"f","arguments":{}}</tool_call>After')

        result = sp.finalize()
        assert len(result.tool_calls) == 1
        assert "Before" in result.content
        assert "After" in result.content

    def test_handle_pattern_start_no_before_with_after_tool(self) -> None:
        """_handle_pattern_start: complete tool pattern with content after but no before."""
        sp = StreamingProcessor()

        sp.feed('<tool_call>{"name":"f","arguments":{}}</tool_call>SomeContent')

        result = sp.finalize()
        assert len(result.tool_calls) == 1
        assert "SomeContent" in result.content

    def test_handle_pattern_start_no_before_no_after_thinking(self) -> None:
        """_handle_pattern_start: complete thinking pattern, no before/after."""
        sp = StreamingProcessor()

        event = sp.feed("<think>just thoughts</think>")

        assert event.reasoning_content is not None
        assert "just thoughts" in (event.reasoning_content or "")
        assert event.is_complete is True

    def test_handle_pattern_start_before_thinking_no_after(self) -> None:
        """_handle_pattern_start: content before + complete thinking, no after."""
        sp = StreamingProcessor()

        event = sp.feed("Preamble<think>thoughts</think>")

        assert "Preamble" in (event.content or "")
        result = sp.finalize()
        assert result.reasoning == "thoughts"

    def test_handle_pattern_start_no_before_thinking_with_after(self) -> None:
        """_handle_pattern_start: complete thinking pattern + content after, no before."""
        sp = StreamingProcessor()

        sp.feed("<think>thoughts</think>Conclusion")

        result = sp.finalize()
        assert result.reasoning == "thoughts"
        assert "Conclusion" in result.content

    def test_partial_marker_yields_nothing_when_at_start(self) -> None:
        """Partial marker at start of combined text yields empty event."""
        sp = StreamingProcessor()

        # Feed just the start of a marker
        event = sp.feed("<")
        # Should buffer and return empty
        assert event.content is None or event.content == ""

    def test_handle_in_pattern_thinking_yields_content(self) -> None:
        """Thinking pattern yields reasoning_content incrementally."""
        sp = StreamingProcessor()

        # Enter thinking mode
        sp.feed("<think>")

        # Feed enough content to exceed REASONING_BUFFER_SIZE
        long_content = "A" * 50
        event = sp.feed(long_content)

        # Should yield reasoning content (part of it, keeping buffer)
        assert event.reasoning_content is not None

    def test_handle_in_pattern_tool_buffers_silently(self) -> None:
        """Tool pattern buffers content without yielding."""
        sp = StreamingProcessor()

        # Enter tool mode
        sp.feed("<tool_call>")

        # Feed tool content
        event = sp.feed('{"name": "test"}')

        # Tool content is buffered, not yielded
        assert event.content is None
        assert event.reasoning_content is None

    def test_partial_marker_with_content_before(self) -> None:
        """Partial marker after content yields the content before the partial."""
        sp = StreamingProcessor()

        # Feed content that ends with partial marker start
        # "Hello<" where '<' could be start of <tool_call>
        event = sp.feed("Hello<")

        # "Hello" should be yielded, "<" should be buffered
        # This tests lines 876-878
        assert event.content == "Hello"

    def test_handle_pattern_start_before_tool_no_after(self) -> None:
        """_handle_pattern_start: content before + complete tool pattern, no after content.

        This covers line 982: return StreamEvent(content=before) when
        there's before content, complete non-thinking pattern, but no after_pattern.
        """
        sp = StreamingProcessor()

        # Text with content before a complete tool call, but nothing after
        event = sp.feed('Result:<tool_call>{"name":"f","arguments":{}}</tool_call>')

        # "Result:" should be yielded as content
        assert event.content is not None
        assert "Result:" in (event.content or "")

        result = sp.finalize()
        assert len(result.tool_calls) == 1
        assert "Result:" in result.content
