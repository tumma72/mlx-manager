"""Comprehensive tests for the parsers module.

Tests cover:
- ToolCallParser ABC contract
- ThinkingParser ABC contract
- Each concrete parser's extract() with known inputs
- validates() delegation to extract()
- stream_markers property correctness
- parser_id uniqueness
- Registry resolve_tool_parser() / resolve_thinking_parser()
- Null parsers (no-op behavior)
- ThinkTagParser.supports_toggle, .extract(), .remove()
"""

import json

import pytest

from mlx_manager.mlx_server.parsers import (
    THINKING_PARSERS,
    TOOL_PARSERS,
    Glm4NativeParser,
    Glm4XmlParser,
    HermesJsonParser,
    LlamaPythonParser,
    LlamaXmlParser,
    NullThinkingParser,
    NullToolParser,
    ThinkingParser,
    ThinkTagParser,
    ToolCallParser,
    resolve_thinking_parser,
    resolve_tool_parser,
)

# --- ABC Contract Tests ---


class TestToolCallParserABC:
    """ToolCallParser cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ToolCallParser()  # type: ignore[abstract]


class TestThinkingParserABC:
    """ThinkingParser cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ThinkingParser()  # type: ignore[abstract]


# --- Parser ID Uniqueness ---


class TestParserIdUniqueness:
    """All parser IDs must be unique within their registry."""

    def test_tool_parser_ids_unique(self) -> None:
        ids = [cls().parser_id for cls in TOOL_PARSERS.values()]
        assert len(ids) == len(set(ids))

    def test_thinking_parser_ids_unique(self) -> None:
        ids = [cls().parser_id for cls in THINKING_PARSERS.values()]
        assert len(ids) == len(set(ids))

    def test_parser_id_matches_registry_key(self) -> None:
        for key, cls in TOOL_PARSERS.items():
            assert cls().parser_id == key
        for key, cls in THINKING_PARSERS.items():
            assert cls().parser_id == key


# --- HermesJsonParser Tests ---


class TestHermesJsonParser:
    """Tests for Hermes/Qwen JSON tool call parsing."""

    def setup_method(self) -> None:
        self.parser = HermesJsonParser()

    def test_parser_id(self) -> None:
        assert self.parser.parser_id == "hermes_json"

    def test_stream_markers(self) -> None:
        assert self.parser.stream_markers == [("<tool_call>", "</tool_call>")]

    def test_extract_closed_tag(self) -> None:
        text = '<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"
        args = json.loads(calls[0].function.arguments)
        assert args["location"] == "Tokyo"

    def test_extract_unclosed_tag(self) -> None:
        text = '<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"

    def test_extract_with_surrounding_text(self) -> None:
        text = (
            "I will check the weather.\n"
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"location": "SF"}}</tool_call>\nDone.'
        )
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"

    def test_extract_multiple_calls(self) -> None:
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>'
            '<tool_call>{"name": "get_time", "arguments": {"timezone": "EST"}}</tool_call>'
        )
        calls = self.parser.extract(text)
        assert len(calls) == 2
        assert calls[0].function.name == "get_weather"
        assert calls[1].function.name == "get_time"

    def test_extract_deduplicates(self) -> None:
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>'
            '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>'
        )
        calls = self.parser.extract(text)
        assert len(calls) == 1

    def test_extract_invalid_json(self) -> None:
        text = "<tool_call>not valid json</tool_call>"
        calls = self.parser.extract(text)
        assert len(calls) == 0

    def test_extract_empty_text(self) -> None:
        assert self.parser.extract("") == []

    def test_validates_positive(self) -> None:
        text = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
        assert self.parser.validates(text, "get_weather") is True

    def test_validates_negative_wrong_fn(self) -> None:
        text = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
        assert self.parser.validates(text, "get_time") is False

    def test_validates_negative_no_call(self) -> None:
        assert self.parser.validates("Hello world", "get_weather") is False

    def test_arguments_as_string(self) -> None:
        """Arguments can be a string value, not just a dict."""
        text = '<tool_call>{"name": "search", "arguments": "hello"}</tool_call>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.arguments == "hello"

    def test_golden_hermes_style(self) -> None:
        """From test_response_processor golden inputs."""
        text = (
            "Here is the result: "
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "SF"}}</tool_call>'
        )
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"
        assert json.loads(calls[0].function.arguments) == {"city": "SF"}


# --- Glm4NativeParser Tests ---


class TestGlm4NativeParser:
    """Tests for GLM-4.7 native compact format."""

    def setup_method(self) -> None:
        self.parser = Glm4NativeParser()

    def test_parser_id(self) -> None:
        assert self.parser.parser_id == "glm4_native"

    def test_stream_markers(self) -> None:
        assert self.parser.stream_markers == [("<tool_call>", "</tool_call>")]

    def test_extract_compact_format(self) -> None:
        text = "<tool_call>get_weather<location>Tokyo</location></tool_call>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"
        args = json.loads(calls[0].function.arguments)
        assert args["location"] == "Tokyo"

    def test_extract_compact_multiple_params(self) -> None:
        text = "<tool_call>get_weather<location>Tokyo</location><unit>celsius</unit></tool_call>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        assert args["location"] == "Tokyo"
        assert args["unit"] == "celsius"

    def test_extract_attr_format(self) -> None:
        text = '<tool_call>get_weather<location="Rome, Italy"</location></tool_call>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"
        args = json.loads(calls[0].function.arguments)
        assert args["location"] == "Rome, Italy"

    def test_extract_unclosed_compact(self) -> None:
        """GLM4 often omits the closing </tool_call> tag."""
        text = "<tool_call>get_weather<location>Tokyo</location>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"

    def test_extract_empty_text(self) -> None:
        assert self.parser.extract("") == []

    def test_validates_compact(self) -> None:
        text = "<tool_call>get_weather<location>Tokyo</location></tool_call>"
        assert self.parser.validates(text, "get_weather") is True
        assert self.parser.validates(text, "get_time") is False

    def test_validates_attr_format(self) -> None:
        text = '<tool_call>search<query="test"</query></tool_call>'
        assert self.parser.validates(text, "search") is True
        assert self.parser.validates(text, "get_weather") is False

    def test_extract_deduplicates(self) -> None:
        """GLM4 sometimes duplicates calls."""
        text = "<tool_call>search<q>test</q></tool_call><tool_call>search<q>test</q></tool_call>"
        calls = self.parser.extract(text)
        assert len(calls) == 1


# --- Glm4XmlParser Tests ---


class TestGlm4XmlParser:
    """Tests for GLM4 full XML format."""

    def setup_method(self) -> None:
        self.parser = Glm4XmlParser()

    def test_parser_id(self) -> None:
        assert self.parser.parser_id == "glm4_xml"

    def test_stream_markers(self) -> None:
        assert self.parser.stream_markers == [("<tool_call>", "</tool_call>")]

    def test_extract_xml_format(self) -> None:
        text = (
            "<tool_call><name>get_weather</name>"
            '<arguments>{"location": "Tokyo"}</arguments>'
            "</tool_call>"
        )
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"
        args = json.loads(calls[0].function.arguments)
        assert args["location"] == "Tokyo"

    def test_extract_with_whitespace(self) -> None:
        text = """<tool_call>
            <name>calculate</name>
            <arguments>{"x": 5, "y": 3}</arguments>
        </tool_call>"""
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "calculate"

    def test_extract_invalid_json_still_extracted(self) -> None:
        """Invalid JSON is still extracted - validation is caller's responsibility."""
        text = "<tool_call><name>test</name><arguments>{not valid}</arguments></tool_call>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "test"
        assert calls[0].function.arguments == "{not valid}"

    def test_extract_empty_text(self) -> None:
        assert self.parser.extract("") == []

    def test_validates(self) -> None:
        text = '<tool_call><name>search</name><arguments>{"q": "test"}</arguments></tool_call>'
        assert self.parser.validates(text, "search") is True
        assert self.parser.validates(text, "get_weather") is False

    def test_golden_glm4_xml_style(self) -> None:
        """From test_response_processor golden inputs."""
        text = (
            '<tool_call><name>calculate</name><arguments>{"x": 5, "y": 3}</arguments></tool_call>'
        )
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "calculate"
        assert json.loads(calls[0].function.arguments) == {"x": 5, "y": 3}


# --- LlamaXmlParser Tests ---


class TestLlamaXmlParser:
    """Tests for Llama XML function format."""

    def setup_method(self) -> None:
        self.parser = LlamaXmlParser()

    def test_parser_id(self) -> None:
        assert self.parser.parser_id == "llama_xml"

    def test_stream_markers(self) -> None:
        assert self.parser.stream_markers == [("<function=", "</function>")]

    def test_extract_function_call(self) -> None:
        text = '<function=get_weather>{"location": "Tokyo"}</function>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"
        args = json.loads(calls[0].function.arguments)
        assert args["location"] == "Tokyo"

    def test_extract_multiple_calls(self) -> None:
        text = (
            '<function=get_weather>{"location": "NYC"}</function>'
            '<function=get_time>{"tz": "EST"}</function>'
        )
        calls = self.parser.extract(text)
        assert len(calls) == 2
        assert calls[0].function.name == "get_weather"
        assert calls[1].function.name == "get_time"

    def test_extract_with_surrounding_text(self) -> None:
        text = 'Result: <function=search>{"query": "test", "limit": 10}</function>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "search"

    def test_extract_invalid_json_still_extracted(self) -> None:
        text = "<function=test>{not valid json}</function>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "test"
        assert calls[0].function.arguments == "{not valid json}"

    def test_extract_empty_text(self) -> None:
        assert self.parser.extract("") == []

    def test_validates(self) -> None:
        text = '<function=get_weather>{"location": "Tokyo"}</function>'
        assert self.parser.validates(text, "get_weather") is True
        assert self.parser.validates(text, "get_time") is False

    def test_golden_llama_xml_style(self) -> None:
        """From test_response_processor golden inputs."""
        text = 'Result: <function=search>{"query": "test", "limit": 10}</function>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "search"
        assert json.loads(calls[0].function.arguments) == {"query": "test", "limit": 10}


# --- LlamaPythonParser Tests ---


class TestLlamaPythonParser:
    """Tests for Llama Python tag format."""

    def setup_method(self) -> None:
        self.parser = LlamaPythonParser()

    def test_parser_id(self) -> None:
        assert self.parser.parser_id == "llama_python"

    def test_stream_markers(self) -> None:
        assert self.parser.stream_markers == [("<|python_tag|>", "<|eom_id|>")]

    def test_extract_python_call(self) -> None:
        text = '<|python_tag|>functions.get_weather(location="Tokyo")<|eom_id|>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "functions.get_weather"
        args = json.loads(calls[0].function.arguments)
        assert args["location"] == "Tokyo"

    def test_extract_integer_args(self) -> None:
        text = '<|python_tag|>api.search(query="test", limit=5)<|eom_id|>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        assert args["query"] == "test"
        assert args["limit"] == 5

    def test_extract_mixed_args(self) -> None:
        text = '<|python_tag|>weather.get(city="NYC", units="metric")<|eom_id|>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        assert args["city"] == "NYC"
        assert args["units"] == "metric"

    def test_extract_empty_text(self) -> None:
        assert self.parser.extract("") == []

    def test_validates(self) -> None:
        text = '<|python_tag|>weather.get(city="NYC")<|eom_id|>'
        assert self.parser.validates(text, "weather.get") is True
        assert self.parser.validates(text, "weather.set") is False

    def test_golden_llama_python_style(self) -> None:
        """From test_response_processor golden inputs."""
        text = '<|python_tag|>weather.get(city="NYC", units="metric")<|eom_id|>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "weather.get"
        args = json.loads(calls[0].function.arguments)
        assert args["city"] == "NYC"
        assert args["units"] == "metric"


# --- NullToolParser Tests ---


class TestNullToolParser:
    """Tests for null/no-op tool parser."""

    def setup_method(self) -> None:
        self.parser = NullToolParser()

    def test_parser_id(self) -> None:
        assert self.parser.parser_id == "null"

    def test_stream_markers_empty(self) -> None:
        assert self.parser.stream_markers == []

    def test_extract_always_empty(self) -> None:
        assert self.parser.extract("any text") == []
        assert self.parser.extract('<tool_call>{"name": "test"}</tool_call>') == []

    def test_validates_always_false(self) -> None:
        assert self.parser.validates("anything", "get_weather") is False
        hermes_text = '<tool_call>{"name": "get_weather"}</tool_call>'
        assert self.parser.validates(hermes_text, "get_weather") is False


# --- ThinkTagParser Tests ---


class TestThinkTagParser:
    """Tests for think tag extraction."""

    def setup_method(self) -> None:
        self.parser = ThinkTagParser()

    def test_parser_id(self) -> None:
        assert self.parser.parser_id == "think_tag"

    def test_supports_toggle(self) -> None:
        assert self.parser.supports_toggle is True

    def test_stream_markers(self) -> None:
        markers = self.parser.stream_markers
        assert ("<think>", "</think>") in markers
        assert ("<thinking>", "</thinking>") in markers
        assert ("<reasoning>", "</reasoning>") in markers
        assert ("<reflection>", "</reflection>") in markers

    def test_extract_think_tag(self) -> None:
        text = "<think>Let me think about this carefully.</think>The answer is 42."
        result = self.parser.extract(text)
        assert result == "Let me think about this carefully."

    def test_extract_thinking_tag(self) -> None:
        text = "<thinking>Deep analysis here.</thinking>Result: yes."
        result = self.parser.extract(text)
        assert result == "Deep analysis here."

    def test_extract_reasoning_tag(self) -> None:
        text = "<reasoning>Step by step logic.</reasoning>Final answer."
        result = self.parser.extract(text)
        assert result == "Step by step logic."

    def test_extract_reflection_tag(self) -> None:
        text = "<reflection>Reviewing my approach.</reflection>Confirmed."
        result = self.parser.extract(text)
        assert result == "Reviewing my approach."

    def test_extract_no_thinking(self) -> None:
        text = "Just a normal response."
        result = self.parser.extract(text)
        assert result is None

    def test_extract_empty_thinking(self) -> None:
        text = "<think></think>The answer."
        result = self.parser.extract(text)
        assert result is None

    def test_extract_whitespace_only_thinking(self) -> None:
        text = "<think>   \n  </think>The answer."
        result = self.parser.extract(text)
        assert result is None

    def test_extract_multiple_blocks(self) -> None:
        text = "<think>First thought.</think>Middle.<think>Second thought.</think>End."
        result = self.parser.extract(text)
        assert "First thought." in result
        assert "Second thought." in result

    def test_extract_mixed_tag_types(self) -> None:
        text = "<think>Thought</think>Text<reasoning>Logic</reasoning>More"
        result = self.parser.extract(text)
        assert "Thought" in result
        assert "Logic" in result

    def test_extract_case_insensitive(self) -> None:
        text = "<THINK>Uppercase thinking.</THINK>Answer."
        result = self.parser.extract(text)
        assert result == "Uppercase thinking."

    def test_extract_preserves_formatting(self) -> None:
        text = "<think>Line 1\nLine 2\n- bullet</think>Answer"
        result = self.parser.extract(text)
        assert "Line 1\nLine 2\n- bullet" in result

    def test_remove_think_tags(self) -> None:
        text = "<think>Internal reasoning.</think>The answer is 42."
        result = self.parser.remove(text)
        assert "Internal reasoning" not in result
        assert "The answer is 42." in result

    def test_remove_multiple_tags(self) -> None:
        text = "<think>A</think>Hello <think>B</think>World"
        result = self.parser.remove(text)
        assert "Hello" in result
        assert "World" in result
        assert "<think>" not in result

    def test_remove_mixed_tag_types(self) -> None:
        text = "<think>A</think>X<thinking>B</thinking>Y<reasoning>C</reasoning>Z"
        result = self.parser.remove(text)
        assert "A" not in result
        assert "B" not in result
        assert "C" not in result
        assert "X" in result
        assert "Y" in result
        assert "Z" in result

    def test_remove_preserves_content(self) -> None:
        text = "Just content without thinking."
        result = self.parser.remove(text)
        assert result == "Just content without thinking."

    def test_remove_normalizes_whitespace(self) -> None:
        text = "Start<think>removed</think>\n\n\n\nEnd"
        result = self.parser.remove(text)
        assert "\n\n\n" not in result
        assert "Start" in result
        assert "End" in result

    def test_remove_case_insensitive(self) -> None:
        text = "<THINK>Remove me</THINK>Keep this"
        result = self.parser.remove(text)
        assert "Remove me" not in result
        assert "Keep this" in result

    def test_golden_single_think_tag(self) -> None:
        """From test_response_processor golden inputs."""
        text = "<think>Let me analyze this</think>The answer is 42."
        result = self.parser.extract(text)
        assert result == "Let me analyze this"

    def test_golden_multiple_thinking_tags(self) -> None:
        """From test_response_processor golden inputs."""
        text = "<think>First thought</think>Middle<think>Second thought</think>End"
        result = self.parser.extract(text)
        assert "First thought" in result
        assert "Second thought" in result


# --- NullThinkingParser Tests ---


class TestNullThinkingParser:
    """Tests for null/no-op thinking parser."""

    def setup_method(self) -> None:
        self.parser = NullThinkingParser()

    def test_parser_id(self) -> None:
        assert self.parser.parser_id == "null"

    def test_supports_toggle_false(self) -> None:
        assert self.parser.supports_toggle is False

    def test_stream_markers_empty(self) -> None:
        assert self.parser.stream_markers == []

    def test_extract_always_none(self) -> None:
        assert self.parser.extract("any text") is None
        assert self.parser.extract("<think>thinking</think>") is None

    def test_remove_identity(self) -> None:
        text = "some text with <think>tags</think> in it"
        assert self.parser.remove(text) == text


# --- Registry Tests ---


class TestRegistry:
    """Tests for parser registry resolution."""

    def test_resolve_tool_parser_hermes(self) -> None:
        parser = resolve_tool_parser("hermes_json")
        assert isinstance(parser, HermesJsonParser)

    def test_resolve_tool_parser_glm4_native(self) -> None:
        parser = resolve_tool_parser("glm4_native")
        assert isinstance(parser, Glm4NativeParser)

    def test_resolve_tool_parser_glm4_xml(self) -> None:
        parser = resolve_tool_parser("glm4_xml")
        assert isinstance(parser, Glm4XmlParser)

    def test_resolve_tool_parser_llama_xml(self) -> None:
        parser = resolve_tool_parser("llama_xml")
        assert isinstance(parser, LlamaXmlParser)

    def test_resolve_tool_parser_llama_python(self) -> None:
        parser = resolve_tool_parser("llama_python")
        assert isinstance(parser, LlamaPythonParser)

    def test_resolve_tool_parser_null(self) -> None:
        parser = resolve_tool_parser("null")
        assert isinstance(parser, NullToolParser)

    def test_resolve_tool_parser_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown tool parser"):
            resolve_tool_parser("nonexistent")

    def test_resolve_thinking_parser_think_tag(self) -> None:
        parser = resolve_thinking_parser("think_tag")
        assert isinstance(parser, ThinkTagParser)

    def test_resolve_thinking_parser_null(self) -> None:
        parser = resolve_thinking_parser("null")
        assert isinstance(parser, NullThinkingParser)

    def test_resolve_thinking_parser_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown thinking parser"):
            resolve_thinking_parser("nonexistent")

    def test_all_tool_parsers_registered(self) -> None:
        expected = {"hermes_json", "glm4_native", "glm4_xml", "llama_xml", "llama_python", "null"}
        assert set(TOOL_PARSERS.keys()) == expected

    def test_all_thinking_parsers_registered(self) -> None:
        expected = {"think_tag", "null"}
        assert set(THINKING_PARSERS.keys()) == expected

    def test_resolve_returns_new_instances(self) -> None:
        """Each resolve call returns a new parser instance."""
        parser1 = resolve_tool_parser("hermes_json")
        parser2 = resolve_tool_parser("hermes_json")
        assert parser1 is not parser2

    def test_registry_completeness(self) -> None:
        """All parser classes are registered."""
        tool_classes = {
            HermesJsonParser,
            Glm4NativeParser,
            Glm4XmlParser,
            LlamaXmlParser,
            LlamaPythonParser,
            NullToolParser,
        }
        registered_tool_classes = set(TOOL_PARSERS.values())
        assert tool_classes == registered_tool_classes

        thinking_classes = {ThinkTagParser, NullThinkingParser}
        registered_thinking_classes = set(THINKING_PARSERS.values())
        assert thinking_classes == registered_thinking_classes


# --- Integration Tests ---


class TestParserIntegration:
    """Tests for parser behavior in realistic scenarios."""

    def test_all_tool_parsers_have_consistent_interface(self) -> None:
        """All tool parsers implement the same interface."""
        for parser_id, cls in TOOL_PARSERS.items():
            parser = cls()
            assert parser.parser_id == parser_id
            assert isinstance(parser.stream_markers, list)
            assert callable(parser.extract)
            assert callable(parser.validates)

    def test_all_thinking_parsers_have_consistent_interface(self) -> None:
        """All thinking parsers implement the same interface."""
        for parser_id, cls in THINKING_PARSERS.items():
            parser = cls()
            assert parser.parser_id == parser_id
            assert isinstance(parser.stream_markers, list)
            assert isinstance(parser.supports_toggle, bool)
            assert callable(parser.extract)
            assert callable(parser.remove)

    def test_tool_call_ids_are_unique_per_call(self) -> None:
        """Tool calls get unique IDs even from same parser."""
        parser = HermesJsonParser()
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        calls = parser.extract(text)
        assert len(calls) == 2
        assert calls[0].id != calls[1].id

    def test_validates_uses_extract_code_path(self) -> None:
        """validates() delegates to extract() - same code path."""
        parser = HermesJsonParser()
        text = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

        # Both should agree
        extracted = parser.extract(text)
        validates = parser.validates(text, "get_weather")

        assert len(extracted) == 1
        assert extracted[0].function.name == "get_weather"
        assert validates is True

    def test_thinking_extract_and_remove_consistency(self) -> None:
        """Thinking parser extract() and remove() are consistent."""
        parser = ThinkTagParser()
        text = "<think>Internal thought</think>External content"

        extracted = parser.extract(text)
        removed = parser.remove(text)

        assert extracted == "Internal thought"
        assert "Internal thought" not in removed
        assert "External content" in removed


# --- Error Handling and Edge Case Coverage Tests ---


class TestHermesJsonParserErrorHandling:
    """Tests for HermesJsonParser error handling."""

    def setup_method(self) -> None:
        self.parser = HermesJsonParser()

    def test_malformed_json_missing_name(self) -> None:
        """Handles JSON missing required 'name' field."""
        text = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
        calls = self.parser.extract(text)
        # Should extract with empty name
        assert len(calls) == 1
        assert calls[0].function.name == ""

    def test_malformed_json_missing_arguments(self) -> None:
        """Handles JSON missing 'arguments' field."""
        text = '<tool_call>{"name": "test"}</tool_call>'
        calls = self.parser.extract(text)
        # Should extract with empty dict as arguments
        assert len(calls) == 1
        assert calls[0].function.arguments == "{}"

    def test_completely_invalid_json(self) -> None:
        """Handles completely invalid JSON."""
        text = "<tool_call>{totally broken json!!!</tool_call>"
        calls = self.parser.extract(text)
        # Should return empty list (logged warning)
        assert len(calls) == 0


class TestGlm4NativeParserErrorHandling:
    """Tests for Glm4NativeParser error handling."""

    def setup_method(self) -> None:
        self.parser = Glm4NativeParser()

    def test_compact_no_function_name(self) -> None:
        """Handles compact format with no extractable function name."""
        text = "<tool_call><param>value</param></tool_call>"
        calls = self.parser.extract(text)
        # Should return empty (no valid function name)
        assert len(calls) == 0

    def test_attr_no_function_name(self) -> None:
        """Handles attr format with no extractable function name."""
        text = '<tool_call><param="value"</param></tool_call>'
        calls = self.parser.extract(text)
        # Should return empty (no valid function name)
        assert len(calls) == 0

    def test_attr_no_params(self) -> None:
        """Handles attr format with function name but no params."""
        text = "<tool_call>func_name</tool_call>"
        calls = self.parser.extract(text)
        # Should return empty (attr parser requires params)
        assert len(calls) == 0

    def test_compact_malformed_params(self) -> None:
        """Handles compact format with malformed parameter tags."""
        text = "<tool_call>test<param>unclosed</tool_call>"
        calls = self.parser.extract(text)
        # Should extract with empty params
        assert len(calls) == 1
        assert calls[0].function.name == "test"
        assert calls[0].function.arguments == "{}"


class TestGlm4XmlParserErrorHandling:
    """Tests for Glm4XmlParser error handling."""

    def setup_method(self) -> None:
        self.parser = Glm4XmlParser()

    def test_missing_name_tag(self) -> None:
        """Handles XML missing <name> tag."""
        text = '<tool_call><arguments>{"x": 1}</arguments></tool_call>'
        calls = self.parser.extract(text)
        # Should not match pattern
        assert len(calls) == 0

    def test_missing_arguments_tag(self) -> None:
        """Handles XML missing <arguments> tag."""
        text = "<tool_call><name>test</name></tool_call>"
        calls = self.parser.extract(text)
        # Should not match pattern
        assert len(calls) == 0


class TestLlamaXmlParserErrorHandling:
    """Tests for LlamaXmlParser error handling."""

    def setup_method(self) -> None:
        self.parser = LlamaXmlParser()

    def test_missing_function_name(self) -> None:
        """Handles XML missing function name."""
        text = "<function=>{}</function>"
        calls = self.parser.extract(text)
        # Pattern requires \w+ so empty name won't match
        assert len(calls) == 0

    def test_empty_arguments(self) -> None:
        """Handles empty arguments."""
        text = "<function=test></function>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.arguments == ""


class TestGlm4NativeParserRegexEdgeCases:
    """Tests for GLM4NativeParser regex edge cases."""

    def setup_method(self) -> None:
        self.parser = Glm4NativeParser()

    def test_compact_format_no_name_only_tags(self) -> None:
        """Pattern that matches but has no function name."""
        # This should match the compact pattern but fail name extraction
        text = "<tool_call><param>value</param></tool_call>"
        calls = self.parser.extract(text)
        # Pattern won't actually match because [^{<][^<]* requires non-tag start
        assert len(calls) == 0


class TestLlamaPythonParserErrorHandling:
    """Tests for LlamaPythonParser error handling."""

    def setup_method(self) -> None:
        self.parser = LlamaPythonParser()

    def test_missing_module(self) -> None:
        """Handles missing module name."""
        text = "<|python_tag|>method(x=1)<|eom_id|>"
        calls = self.parser.extract(text)
        # Pattern requires module.method so won't match
        assert len(calls) == 0

    def test_missing_method(self) -> None:
        """Handles missing method name."""
        text = "<|python_tag|>module.(x=1)<|eom_id|>"
        calls = self.parser.extract(text)
        # Pattern requires module.method so won't match
        assert len(calls) == 0

    def test_empty_arguments(self) -> None:
        """Handles empty arguments."""
        text = "<|python_tag|>api.test()<|eom_id|>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        assert calls[0].function.name == "api.test"
        assert calls[0].function.arguments == "{}"

    def test_complex_argument_parsing(self) -> None:
        """Handles various argument types."""
        text = '<|python_tag|>api.call(str="text", num=42, dec=3.14, flag=true)<|eom_id|>'
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        assert args["str"] == "text"
        assert args["num"] == 42
        assert args["dec"] == 3.14
        assert args["flag"] is True

    def test_single_quotes_in_arguments(self) -> None:
        """Handles single-quoted strings in arguments."""
        text = "<|python_tag|>api.call(text='hello')<|eom_id|>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        assert args["text"] == "hello"

    def test_boolean_arguments(self) -> None:
        """Handles boolean arguments."""
        text = "<|python_tag|>api.call(enabled=true, disabled=false)<|eom_id|>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        assert args["enabled"] is True
        assert args["disabled"] is False

    def test_float_arguments(self) -> None:
        """Handles float arguments with decimal points."""
        text = "<|python_tag|>api.call(value=2.5)<|eom_id|>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        assert args["value"] == 2.5

    def test_invalid_float_fallback(self) -> None:
        """Handles value that looks like a float (pattern matches partial)."""
        text = "<|python_tag|>api.call(value=2.5.5)<|eom_id|>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        # Regex matches first valid float part (2.5)
        assert args["value"] == 2.5

    def test_non_numeric_identifier_arguments(self) -> None:
        """Handles identifier arguments that aren't true/false."""
        text = "<|python_tag|>api.call(value=undefined)<|eom_id|>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        # Should fallback to string
        assert args["value"] == "undefined"

    def test_truly_invalid_float_with_exception(self) -> None:
        """Handles float with characters that cause ValueError."""
        # The regex pattern doesn't actually allow this case easily
        # but we test the fallback path by ensuring non-digit non-bool identifiers work
        text = "<|python_tag|>api.call(name=some_var)<|eom_id|>"
        calls = self.parser.extract(text)
        assert len(calls) == 1
        args = json.loads(calls[0].function.arguments)
        assert args["name"] == "some_var"
