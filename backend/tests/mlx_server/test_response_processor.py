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

from mlx_manager.mlx_server.services.response_processor import (
    ParseResult,
    ResponseProcessor,
    StreamEvent,
    StreamingProcessor,
    ToolCall,
    ToolCallFunction,
    create_default_processor,
    get_response_processor,
    reset_response_processor,
)

# --- Pydantic Model Tests ---


class TestPydanticModels:
    """Tests for Pydantic model behavior."""

    def test_tool_call_function_serialization(self) -> None:
        """ToolCallFunction serializes to dict correctly."""
        func = ToolCallFunction(name="get_weather", arguments='{"city": "SF"}')
        data = func.model_dump()

        assert data == {"name": "get_weather", "arguments": '{"city": "SF"}'}

    def test_tool_call_serialization(self) -> None:
        """ToolCall serializes to dict correctly."""
        tc = ToolCall(
            id="call_abc123",
            function=ToolCallFunction(name="search", arguments='{"q": "test"}'),
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
            function=ToolCallFunction(name="test", arguments="{}"),
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
