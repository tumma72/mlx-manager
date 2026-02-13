"""Coverage tests for tool_call.py error/edge paths.

These tests target specific uncovered branches (exception handlers,
no-match returns) in the parser _parse_match / _parse_compact / _parse_attr
static methods.
"""

import re
from unittest.mock import MagicMock, patch

from mlx_manager.mlx_server.parsers.tool_call import (
    Glm4NativeParser,
    Glm4XmlParser,
    HermesJsonParser,
    LiquidPythonParser,
    LlamaPythonParser,
    LlamaXmlParser,
    _parse_python_args,
)


class TestHermesJsonParserErrors:
    """Cover lines 64-66: JSONDecodeError/KeyError in _parse_match."""

    def test_invalid_json_returns_none(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.return_value = "not valid json {{"
        assert HermesJsonParser._parse_match(match) is None

    def test_key_error_returns_none(self) -> None:
        """Force KeyError during json.loads to cover the except branch."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = '{"x": 1}'
        with patch(
            "mlx_manager.mlx_server.parsers.tool_call.json.loads",
            side_effect=KeyError("forced"),
        ):
            assert HermesJsonParser._parse_match(match) is None


class TestGlm4NativeParserErrors:
    """Cover lines 122, 131-133, 141, 146, 152-154."""

    def test_compact_no_name_match_returns_none(self) -> None:
        """Line 122: content with no word chars -> name_match is None."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = "   "  # no word characters
        assert Glm4NativeParser._parse_compact(match) is None

    def test_compact_exception_returns_none(self) -> None:
        """Lines 131-133: IndexError/AttributeError in _parse_compact."""
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad match")
        assert Glm4NativeParser._parse_compact(match) is None

    def test_attr_no_name_match_returns_none(self) -> None:
        """Line 141: content with no word chars -> name_match is None."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = "   "
        assert Glm4NativeParser._parse_attr(match) is None

    def test_attr_no_params_returns_none(self) -> None:
        """Line 146: name found but no attr-style params."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = "funcname_without_attrs"
        assert Glm4NativeParser._parse_attr(match) is None

    def test_attr_exception_returns_none(self) -> None:
        """Lines 152-154: IndexError/AttributeError in _parse_attr."""
        match = MagicMock(spec=re.Match)
        match.group.side_effect = IndexError("no group")
        assert Glm4NativeParser._parse_attr(match) is None


class TestGlm4XmlParserErrors:
    """Cover lines 194-196: IndexError/AttributeError in _parse_match."""

    def test_exception_returns_none(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad")
        assert Glm4XmlParser._parse_match(match) is None


class TestLlamaXmlParserErrors:
    """Cover lines 233-235: IndexError/AttributeError in _parse_match."""

    def test_exception_returns_none(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.side_effect = IndexError("bad")
        assert LlamaXmlParser._parse_match(match) is None


class TestLlamaPythonParserErrors:
    """Cover lines 271-273: IndexError/AttributeError in _parse_match."""

    def test_exception_returns_none(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad")
        assert LlamaPythonParser._parse_match(match) is None


class TestLiquidPythonParserErrors:
    """Cover error paths in LiquidPythonParser._parse_match."""

    def test_empty_content_returns_empty(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.return_value = "   "
        assert LiquidPythonParser._parse_match(match) == []

    def test_invalid_syntax_returns_empty(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.return_value = "[get_weather(city=]"
        assert LiquidPythonParser._parse_match(match) == []

    def test_exception_returns_empty(self) -> None:
        match = MagicMock(spec=re.Match)
        match.group.side_effect = AttributeError("bad")
        assert LiquidPythonParser._parse_match(match) == []

    def test_unsupported_function_type(self) -> None:
        """Test call with non-Name function (e.g., attribute access)."""
        match = MagicMock(spec=re.Match)
        match.group.return_value = "[obj.method()]"
        result = LiquidPythonParser._parse_match(match)
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
