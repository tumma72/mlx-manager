"""Coverage tests for tool_call.py error/edge paths.

These tests target specific uncovered branches (exception handlers,
no-match returns) in the parser _parse_match / _parse_compact / _parse_attr
static methods.
"""

import json
import re
from unittest.mock import MagicMock, patch

from mlx_manager.mlx_server.parsers.tool_call import (
    DevstralArgsParser,
    FunctionGemmaParser,
    Glm4NativeParser,
    Glm4XmlParser,
    HermesJsonParser,
    LiquidPythonParser,
    LlamaPythonParser,
    LlamaXmlParser,
    MistralNativeParser,
    OpenAIJsonParser,
    Qwen3CoderXmlParser,
    ToolCodePythonParser,
    _parse_python_args,
)


class TestHermesJsonParserErrors:
    """Cover lines 64-66: JSONDecodeError/KeyError in _parse_match."""

    def test_invalid_json_returns_none(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.return_value = "not valid json {{"
        assert HermesJsonParser()._parse_match(match) is None

    def test_key_error_returns_none(self) -> None:
        """Force KeyError during json.loads to cover the except branch."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = '{"x": 1}'
        with patch(
            "mlx_manager.mlx_server.parsers.tool_call.json.loads",
            side_effect=KeyError("forced"),
        ):
            assert HermesJsonParser()._parse_match(match) is None


class TestGlm4NativeParserErrors:
    """Cover lines 122, 131-133, 141, 146, 152-154."""

    def test_compact_no_name_match_returns_none(self) -> None:
        """Line 122: content with no word chars -> name_match is None."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = "   "  # no word characters
        assert Glm4NativeParser()._parse_compact(match) is None

    def test_compact_exception_returns_none(self) -> None:
        """Lines 131-133: IndexError/AttributeError in _parse_compact."""
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad match")
        assert Glm4NativeParser()._parse_compact(match) is None

    def test_attr_no_name_match_returns_none(self) -> None:
        """Line 141: content with no word chars -> name_match is None."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = "   "
        assert Glm4NativeParser()._parse_attr(match) is None

    def test_attr_no_params_returns_none(self) -> None:
        """Line 146: name found but no attr-style params."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = "funcname_without_attrs"
        assert Glm4NativeParser()._parse_attr(match) is None

    def test_attr_exception_returns_none(self) -> None:
        """Lines 152-154: IndexError/AttributeError in _parse_attr."""
        match = MagicMock(spec=re.Match)
        match.group.side_effect = IndexError("no group")
        assert Glm4NativeParser()._parse_attr(match) is None


class TestGlm4XmlParserErrors:
    """Cover lines 194-196: IndexError/AttributeError in _parse_match."""

    def test_exception_returns_none(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad")
        assert Glm4XmlParser()._parse_match(match) is None


class TestLlamaXmlParserErrors:
    """Cover lines 233-235: IndexError/AttributeError in _parse_match."""

    def test_exception_returns_none(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.side_effect = IndexError("bad")
        assert LlamaXmlParser()._parse_match(match) is None


class TestLlamaPythonParserErrors:
    """Cover lines 271-273: IndexError/AttributeError in _parse_match."""

    def test_exception_returns_none(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad")
        assert LlamaPythonParser()._parse_match(match) is None


class TestLiquidPythonParserErrors:
    """Cover error paths in LiquidPythonParser._parse_match."""

    def test_empty_content_returns_empty(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.return_value = "   "
        assert LiquidPythonParser()._parse_match(match) == []

    def test_invalid_syntax_returns_empty(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.return_value = "[get_weather(city=]"
        assert LiquidPythonParser()._parse_match(match) == []

    def test_exception_returns_empty(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad")
        assert LiquidPythonParser()._parse_match(match) == []

    def test_unsupported_function_type(self) -> None:
        """Test call with non-Name function (e.g., attribute access)."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = "[obj.method()]"
        result = LiquidPythonParser()._parse_match(match)
        # Should skip unsupported function types
        assert len(result) == 0


class TestParsePythonArgsEdgeCases:
    """Cover lines 312-313: ValueError in float parsing."""

    def test_dotted_non_float_triggers_valueerror_fallback(self) -> None:
        """Lines 312-313: '.' in value but float() raises ValueError."""
        # Patch the builtins float to trigger ValueError for "3.14"
        original_float = float
        with patch(
            "builtins.float",
            side_effect=lambda v: (
                (_ for _ in ()).throw(ValueError("forced")) if v == "3.14" else original_float(v)
            ),
        ):
            result = _parse_python_args("rate=3.14")
            assert result["rate"] == "3.14"  # Falls back to string

    def test_integer_parsing(self) -> None:
        result = _parse_python_args("count=42")
        assert result["count"] == 42

    def test_float_parsing(self) -> None:
        result = _parse_python_args("rate=3.14")
        assert result["rate"] == 3.14

    def test_boolean_true(self) -> None:
        result = _parse_python_args("flag=True")
        assert result["flag"] is True

    def test_boolean_false(self) -> None:
        result = _parse_python_args("flag=False")
        assert result["flag"] is False

    def test_bare_word_value(self) -> None:
        result = _parse_python_args("mode=fast")
        assert result["mode"] == "fast"


class TestGlm4XmlParserEmptyArguments:
    """Cover lines 158-160: empty arguments string returns None."""

    def test_empty_arguments_returns_none(self) -> None:
        """GLM4 XML with empty <arguments></arguments> returns None."""
        parser = Glm4XmlParser()
        text = "<tool_call><name>get_weather</name><arguments>  </arguments></tool_call>"
        result = parser.extract(text)
        assert result == []

    def test_whitespace_only_arguments_returns_none(self) -> None:
        """GLM4 XML with whitespace-only arguments returns None."""
        parser = Glm4XmlParser()
        text = "<tool_call><name>get_weather</name><arguments></arguments></tool_call>"
        result = parser.extract(text)
        assert result == []


class TestLiquidPythonParserKwargNone:
    """Cover lines 317-318, 322-324: kw.arg is None + ValueError/SyntaxError fallback."""

    def test_kwarg_none_is_skipped(self) -> None:
        """Line 317-318: **kwargs-style argument (kw.arg is None) is skipped."""
        parser = LiquidPythonParser()
        # **kwargs produces kw.arg = None in AST
        text = "<|tool_call_start|>get_weather(**opts)<|tool_call_end|>"
        result = parser.extract(text)
        # The call is parsed but the **opts kwarg is skipped
        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        args = json.loads(result[0].function.arguments)
        assert args == {}

    def test_literal_eval_fallback_to_unparse(self) -> None:
        """Lines 322-324: value that ast.literal_eval can't handle falls back to ast.unparse."""
        parser = LiquidPythonParser()
        # x + 1 is a BinOp, not a literal — literal_eval raises ValueError
        text = "<|tool_call_start|>compute(expr=x + 1)<|tool_call_end|>"
        result = parser.extract(text)
        assert len(result) == 1
        args = json.loads(result[0].function.arguments)
        # ast.unparse produces the string representation
        assert args["expr"] == "x + 1"


class TestMistralNativeParserEdgePaths:
    """Cover lines 366-372, 374, 391-393."""

    def test_ast_literal_eval_fallback_on_single_quotes(self) -> None:
        """Lines 364-365: JSON fails, ast.literal_eval succeeds with Python dicts."""
        parser = MistralNativeParser()
        # Single-quoted Python dicts (invalid JSON but valid Python)
        text = "[TOOL_CALLS] [{'name': 'get_weather', 'arguments': {'city': 'Paris'}}]"
        result = parser.extract(text)
        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        args = json.loads(result[0].function.arguments)
        assert args["city"] == "Paris"

    def test_ast_literal_eval_also_fails(self) -> None:
        """Lines 366-372: both json.loads and ast.literal_eval fail."""
        parser = MistralNativeParser()
        # The regex requires [...] brackets, and the content must be
        # invalid for both json.loads and ast.literal_eval
        text = "[TOOL_CALLS] [not valid {json} or python !!!]"
        result = parser.extract(text)
        assert result == []

    def test_not_a_list_returns_empty(self) -> None:
        """Line 374: data is valid JSON but not a list."""
        parser = MistralNativeParser()
        # The regex captures [{"name"...}] but here it's just {..} — however
        # the regex requires [...], so we need to trick it. The regex matches
        # \[.*\] so we need a string that parses as non-list.
        # Actually the regex requires '[' and ']' so anything inside will parse as a list.
        # We can force this with a patched json.loads that returns a dict.
        with patch(
            "mlx_manager.mlx_server.parsers.tool_call.json.loads",
            return_value={"name": "get_weather"},
        ):
            result = parser.extract('[TOOL_CALLS] [{"name": "get_weather"}]')
            assert result == []

    def test_parse_item_empty_name(self) -> None:
        """Line 386: _parse_item returns None when name is empty."""
        parser = MistralNativeParser()
        result = parser._parse_item({"name": "", "arguments": {}})
        assert result is None

    def test_parse_item_missing_name(self) -> None:
        """Line 386: _parse_item returns None when name key is missing."""
        parser = MistralNativeParser()
        result = parser._parse_item({"arguments": {"x": 1}})
        assert result is None

    def test_parse_item_type_error(self) -> None:
        """Lines 391-393: TypeError in _parse_item handler."""
        parser = MistralNativeParser()
        with patch.object(
            MistralNativeParser,
            "_coerce_arguments",
            side_effect=TypeError("forced"),
        ):
            result = parser._parse_item({"name": "test", "arguments": {}})
            assert result is None

    def test_parse_item_key_error(self) -> None:
        """Lines 391-393: KeyError in _parse_item handler."""
        parser = MistralNativeParser()
        with patch.object(
            MistralNativeParser,
            "_coerce_arguments",
            side_effect=KeyError("forced"),
        ):
            result = parser._parse_item({"name": "test", "arguments": {}})
            assert result is None


class TestDevstralArgsParserFallback:
    """Cover lines 427, 441: empty name skip + fallback JSON parsing."""

    def test_empty_name_is_skipped(self) -> None:
        """Line 427: empty function name after strip is skipped.

        The regex [^\\[\\s]+ prevents empty names from matching,
        so this line is structurally unreachable. We verify the regex
        behavior produces no results when there's no valid name.
        """
        parser = DevstralArgsParser()
        text = '[TOOL_CALLS]   [ARGS]{"key": "val"}'
        result = parser.extract(text)
        assert result == []

    def test_fallback_json_parsing_after_raw_decode_failure(self) -> None:
        """Lines 434-441: raw_decode fails, fallback grabs chunk and parses.

        We mock JSONDecoder to return an instance whose raw_decode always
        fails, forcing the fallback path. json.loads is NOT affected because
        it creates its own decoder internally.
        """
        parser = DevstralArgsParser()

        failing_decoder = MagicMock(spec=json.JSONDecoder)
        failing_decoder.raw_decode.side_effect = json.JSONDecodeError("forced", "", 0)

        with patch(
            "mlx_manager.mlx_server.parsers.tool_call.json.JSONDecoder",
            return_value=failing_decoder,
        ):
            text = '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}'
            result = parser.extract(text)
            assert len(result) == 1
            assert result[0].function.name == "get_weather"
            args = json.loads(result[0].function.arguments)
            assert args["city"] == "Paris"

    def test_fallback_also_fails_logs_warning(self) -> None:
        """Lines 442-448: both raw_decode and json.loads fail, logs warning."""
        parser = DevstralArgsParser()

        failing_decoder = MagicMock(spec=json.JSONDecoder)
        failing_decoder.raw_decode.side_effect = json.JSONDecodeError("forced", "", 0)

        with patch(
            "mlx_manager.mlx_server.parsers.tool_call.json.JSONDecoder",
            return_value=failing_decoder,
        ):
            # Chunk after [ARGS] is not valid JSON at all
            text = "[TOOL_CALLS]get_weather[ARGS]not valid json at all"
            result = parser.extract(text)
            assert result == []

    def test_fallback_json_parsing_with_multiple_calls(self) -> None:
        """Fallback with [TOOL_CALLS] delimiter splits chunks correctly."""
        parser = DevstralArgsParser()

        failing_decoder = MagicMock(spec=json.JSONDecoder)
        failing_decoder.raw_decode.side_effect = json.JSONDecodeError("forced", "", 0)

        with patch(
            "mlx_manager.mlx_server.parsers.tool_call.json.JSONDecoder",
            return_value=failing_decoder,
        ):
            text = '[TOOL_CALLS]fn1[ARGS]{"a": 1}[TOOL_CALLS]fn2[ARGS]{"b": 2}'
            result = parser.extract(text)
            assert len(result) == 2
            assert result[0].function.name == "fn1"
            assert result[1].function.name == "fn2"


class TestOpenAIJsonParserNonDict:
    """Cover line 501: non-dict JSON object is skipped."""

    def test_json_array_is_skipped(self) -> None:
        """Line 501: raw_decode returns a non-dict object — continue.

        While '{' naturally produces dict from raw_decode, we mock to verify
        the guard clause works if the decoder somehow returns a non-dict.
        """
        parser = OpenAIJsonParser()

        def patched_raw_decode(self_decoder: json.JSONDecoder, s: str, idx: int = 0) -> tuple:
            # Return a non-dict to hit line 500-501
            return ([1, 2, 3], idx + 5)

        with patch.object(json.JSONDecoder, "raw_decode", patched_raw_decode):
            result = parser.extract("some text {stuff}")
            assert result == []


class TestToolCodePythonParserEdgePaths:
    """Cover lines 578-580, 588, 594, 598-600, 606-608."""

    def test_syntax_error_returns_empty(self) -> None:
        """Lines 565-571: invalid Python syntax in tool_code block."""
        parser = ToolCodePythonParser()
        text = "```tool_code\nget_weather(city=\n```"
        result = parser.extract(text)
        assert result == []

    def test_list_of_calls_in_tool_code(self) -> None:
        """Lines 577-580: body is ast.List containing ast.Call elements."""
        parser = ToolCodePythonParser()
        text = '```tool_code\n[get_weather(city="Paris"), get_time(tz="UTC")]\n```'
        result = parser.extract(text)
        assert len(result) == 2
        assert result[0].function.name == "get_weather"
        assert result[1].function.name == "get_time"

    def test_non_name_function_is_skipped(self) -> None:
        """Line 588: function call with non-Name func (e.g., attribute access) is skipped."""
        parser = ToolCodePythonParser()
        text = '```tool_code\nobj.method(x="val")\n```'
        result = parser.extract(text)
        assert result == []

    def test_kwarg_none_is_skipped(self) -> None:
        """Line 593-594: **kwargs-style (kw.arg is None) is skipped."""
        parser = ToolCodePythonParser()
        text = "```tool_code\nget_weather(**opts)\n```"
        result = parser.extract(text)
        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        args = json.loads(result[0].function.arguments)
        assert args == {}

    def test_literal_eval_fallback_to_unparse(self) -> None:
        """Lines 598-600: value not evaluable by literal_eval falls back to ast.unparse."""
        parser = ToolCodePythonParser()
        text = "```tool_code\ncompute(expr=x + 1)\n```"
        result = parser.extract(text)
        assert len(result) == 1
        args = json.loads(result[0].function.arguments)
        assert args["expr"] == "x + 1"

    def test_index_error_in_outer_handler(self) -> None:
        """Lines 606-608: IndexError/AttributeError handler."""
        parser = ToolCodePythonParser()
        match = MagicMock(spec=re.Match)
        match.group.side_effect = IndexError("bad")
        result = parser._parse_match(match)
        assert result == []

    def test_attribute_error_in_outer_handler(self) -> None:
        """Lines 606-608: AttributeError handler."""
        parser = ToolCodePythonParser()
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad")
        result = parser._parse_match(match)
        assert result == []


class TestQwen3CoderXmlParserExceptionHandler:
    """Cover lines 700-702: exception handler in _parse_match."""

    def test_index_error_returns_none(self) -> None:
        """Lines 700-702: IndexError in _parse_match."""
        parser = Qwen3CoderXmlParser()
        match = MagicMock(spec=re.Match)
        match.group.side_effect = IndexError("bad")
        result = parser._parse_match(match)
        assert result is None

    def test_attribute_error_returns_none(self) -> None:
        """Lines 700-702: AttributeError in _parse_match."""
        parser = Qwen3CoderXmlParser()
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad")
        result = parser._parse_match(match)
        assert result is None


class TestFunctionGemmaParserEdgePaths:
    """Cover lines 735-737, 741-755."""

    def test_empty_args_block_returns_tool_call_with_empty_args(self) -> None:
        """Lines 745-746: empty args_block produces tool call with '{}'."""
        parser = FunctionGemmaParser()
        text = "<start_function_call>call:get_time{}<end_function_call>"
        result = parser.extract(text)
        assert len(result) == 1
        assert result[0].function.name == "get_time"
        assert result[0].function.arguments == "{}"

    def test_normal_parse_with_escape_args(self) -> None:
        """Lines 748-752: normal parsing with <escape>-delimited args."""
        parser = FunctionGemmaParser()
        text = (
            "<start_function_call>call:get_weather{"
            "city:<escape>Paris<escape>"
            "units:<escape>metric<escape>"
            "}<end_function_call>"
        )
        result = parser.extract(text)
        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        args = json.loads(result[0].function.arguments)
        assert args == {"city": "Paris", "units": "metric"}

    def test_unclosed_variant(self) -> None:
        """Unclosed variant also works (unclosed pattern fallback)."""
        parser = FunctionGemmaParser()
        text = "<start_function_call>call:get_weather{city:<escape>Paris<escape>}"
        result = parser.extract(text)
        assert len(result) == 1
        assert result[0].function.name == "get_weather"

    def test_exception_handler_index_error(self) -> None:
        """Lines 753-755: IndexError in _parse_match."""
        parser = FunctionGemmaParser()
        match = MagicMock(spec=re.Match)
        match.group.side_effect = IndexError("bad")
        result = parser._parse_match(match)
        assert result is None

    def test_exception_handler_attribute_error(self) -> None:
        """Lines 753-755: AttributeError in _parse_match."""
        parser = FunctionGemmaParser()
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad")
        result = parser._parse_match(match)
        assert result is None
