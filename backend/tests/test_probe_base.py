"""Tests for probe base classes to improve coverage.

Focuses on GenerativeProbe edge cases and error paths.
"""

from unittest.mock import MagicMock

from mlx_manager.services.probe.base import (
    _build_marker_to_parsers,
    _detect_all_tags,
    _detect_unknown_thinking_tags,
    _detect_unknown_xml_tags,
    _discover_and_map_tags,
    _find_unclosed_thinking_tag,
    _scan_for_known_markers,
)
from mlx_manager.services.probe.steps import TagDiscovery


def test_detect_unknown_xml_tags_finds_unknown_tags():
    """Test that _detect_unknown_xml_tags identifies tags not in known set."""
    output = "<custom_tag>Some content</custom_tag><think>Known tag</think>"

    unknown = _detect_unknown_xml_tags(output)

    assert "custom_tag" in unknown
    assert "think" not in unknown  # think is a known tag


def test_detect_unknown_xml_tags_requires_closing_tag():
    """Test that _detect_unknown_xml_tags only reports tags with closing tags."""
    output = "<opening_only>No closing tag"

    unknown = _detect_unknown_xml_tags(output)

    # Should not report tags without closing tags
    assert "opening_only" not in unknown


def test_detect_unknown_xml_tags_case_insensitive():
    """Test that _detect_unknown_xml_tags is case-insensitive for known tags."""
    output = "<THINK>Uppercase thinking</THINK><Custom>Unknown tag</Custom>"

    unknown = _detect_unknown_xml_tags(output)

    assert "think" not in unknown  # THINK is normalized to lowercase and matches known set
    assert "custom" in unknown  # Custom is unknown


def test_detect_unknown_xml_tags_with_attributes():
    """Test that _detect_unknown_xml_tags handles tags with attributes."""
    output = '<custom_tag id="123">Content</custom_tag>'

    unknown = _detect_unknown_xml_tags(output)

    assert "custom_tag" in unknown


def test_detect_unknown_xml_tags_empty_output():
    """Test that _detect_unknown_xml_tags handles empty output."""
    unknown = _detect_unknown_xml_tags("")

    assert len(unknown) == 0


# ---------------------------------------------------------------------------
# Tests for _detect_all_tags
# ---------------------------------------------------------------------------


class TestDetectAllTags:
    """Tests for the generic tag detection function."""

    def test_detects_xml_tags(self) -> None:
        output = "<tool_call>some content</tool_call>"
        tags = _detect_all_tags(output)
        xml_tags = [t for t in tags if t.style == "xml"]
        assert any(t.name == "tool_call" and t.paired for t in xml_tags)

    def test_detects_bracket_tags(self) -> None:
        output = '[TOOL_CALLS] [{"name": "test"}]'
        tags = _detect_all_tags(output)
        bracket_tags = [t for t in tags if t.style == "bracket"]
        assert any(t.name == "TOOL_CALLS" for t in bracket_tags)

    def test_detects_paired_bracket_tags(self) -> None:
        output = "[THINK]Some reasoning here[/THINK]The answer."
        tags = _detect_all_tags(output)
        bracket_tags = [t for t in tags if t.style == "bracket"]
        assert any(t.name == "THINK" and t.paired for t in bracket_tags)

    def test_detects_unpaired_bracket_tags(self) -> None:
        output = "[TOOL_CALLS] some output"
        tags = _detect_all_tags(output)
        bracket_tags = [t for t in tags if t.style == "bracket"]
        assert any(t.name == "TOOL_CALLS" and not t.paired for t in bracket_tags)

    def test_detects_mixed_styles(self) -> None:
        output = "<think>reasoning</think>[TOOL_CALLS] some output"
        tags = _detect_all_tags(output)
        xml_tags = [t for t in tags if t.style == "xml"]
        bracket_tags = [t for t in tags if t.style == "bracket"]
        assert len(xml_tags) >= 1
        assert len(bracket_tags) >= 1

    def test_deduplicates_tags(self) -> None:
        output = "<tool_call>a</tool_call><tool_call>b</tool_call>"
        tags = _detect_all_tags(output)
        tool_call_tags = [t for t in tags if t.name == "tool_call"]
        assert len(tool_call_tags) == 1

    def test_empty_output(self) -> None:
        tags = _detect_all_tags("")
        assert tags == []

    def test_no_tags(self) -> None:
        tags = _detect_all_tags("Just normal text without any tags")
        assert tags == []

    def test_xml_tags_with_attributes(self) -> None:
        output = '<custom id="123">content</custom>'
        tags = _detect_all_tags(output)
        assert any(t.name == "custom" and t.paired for t in tags)

    def test_xml_tag_case_normalization(self) -> None:
        output = "<THINK>content</THINK>"
        tags = _detect_all_tags(output)
        # XML tags are normalized to lowercase
        assert any(t.name == "think" for t in tags)

    def test_bracket_tag_preserves_case(self) -> None:
        output = "[TOOL_CALLS] content"
        tags = _detect_all_tags(output)
        # Bracket tags preserve case
        assert any(t.name == "TOOL_CALLS" for t in tags)


# ---------------------------------------------------------------------------
# Tests for family detection with devstral
# ---------------------------------------------------------------------------


def test_detect_model_family_devstral():
    """Test that devstral models are detected as mistral family."""
    from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

    assert detect_model_family("mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit") == "mistral"
    assert detect_model_family("some-org/devstral-mini") == "mistral"
    # Existing mistral patterns still work
    assert detect_model_family("mlx-community/Mistral-7B-Instruct-v0.3-4bit") == "mistral"
    assert detect_model_family("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit") == "mistral"


# ---------------------------------------------------------------------------
# Tests for bracket-aware unclosed thinking tag detection
# ---------------------------------------------------------------------------


def test_find_unclosed_thinking_tag_bracket_style():
    """Test that _find_unclosed_thinking_tag detects unclosed [THINK] tags."""
    # Unclosed bracket-style
    assert _find_unclosed_thinking_tag("[THINK]Some reasoning here") == "THINK"
    # Closed bracket-style returns None
    assert _find_unclosed_thinking_tag("[THINK]Some reasoning[/THINK]") is None
    # Still works for XML-style
    assert _find_unclosed_thinking_tag("<think>Some reasoning") == "think"
    assert _find_unclosed_thinking_tag("<think>Some reasoning</think>") is None


def test_detect_unknown_thinking_tags_bracket_style():
    """Test that _detect_unknown_thinking_tags detects bracket-style thinking."""
    # Bracket-style at start of output with closing tag
    result = _detect_unknown_thinking_tags("[CUSTOM]thinking content[/CUSTOM]rest")
    assert result == "CUSTOM"
    # Known benign tags are not flagged
    assert _detect_unknown_thinking_tags("[THINK]reasoning[/THINK]answer") is None


# ---------------------------------------------------------------------------
# Tests for special token detection in _detect_all_tags
# ---------------------------------------------------------------------------


class TestDetectAllTagsSpecialTokens:
    """Tests for special token pattern detection (<|token|>)."""

    def test_detects_special_tokens(self) -> None:
        output = "<|python_tag|>some code<|eom_id|>"
        tags = _detect_all_tags(output)
        special_tags = [t for t in tags if t.style == "special"]
        assert any(t.name == "python_tag" for t in special_tags)
        assert any(t.name == "eom_id" for t in special_tags)

    def test_special_token_paired_detection(self) -> None:
        output = "<|tool_call_start|>call data<|tool_call_start|>more"
        tags = _detect_all_tags(output)
        special_tags = [t for t in tags if t.style == "special" and t.name == "tool_call_start"]
        assert len(special_tags) == 1
        assert special_tags[0].paired is True  # Two occurrences = paired

    def test_special_token_unpaired(self) -> None:
        output = "<|python_tag|>some code"
        tags = _detect_all_tags(output)
        special_tags = [t for t in tags if t.style == "special" and t.name == "python_tag"]
        assert len(special_tags) == 1
        assert special_tags[0].paired is False

    def test_special_token_deduplication(self) -> None:
        output = "<|tok|>a<|tok|>b<|tok|>c"
        tags = _detect_all_tags(output)
        tok_tags = [t for t in tags if t.name == "tok" and t.style == "special"]
        assert len(tok_tags) == 1


# ---------------------------------------------------------------------------
# Tests for _build_marker_to_parsers
# ---------------------------------------------------------------------------


class TestBuildMarkerToParsers:
    """Tests for the marker-to-parser lookup builder."""

    def test_builds_lookup_from_parsers(self) -> None:
        mock_parser_a = MagicMock()
        mock_parser_a_inst = MagicMock()
        mock_parser_a_inst.stream_markers = [("<tool_call>", "</tool_call>")]
        mock_parser_a.return_value = mock_parser_a_inst

        mock_parser_b = MagicMock()
        mock_parser_b_inst = MagicMock()
        mock_parser_b_inst.stream_markers = [("[TOOL_CALLS]", "")]
        mock_parser_b.return_value = mock_parser_b_inst

        parsers = {"null": MagicMock, "parser_a": mock_parser_a, "parser_b": mock_parser_b}
        result = _build_marker_to_parsers(parsers)

        assert "<tool_call>" in result
        assert "parser_a" in result["<tool_call>"]
        assert "[TOOL_CALLS]" in result
        assert "parser_b" in result["[TOOL_CALLS]"]

    def test_shared_markers_multiple_parsers(self) -> None:
        """Multiple parsers sharing same marker are all listed."""
        mock_p1 = MagicMock()
        mock_p1_inst = MagicMock()
        mock_p1_inst.stream_markers = [("<tool_call>", "</tool_call>")]
        mock_p1.return_value = mock_p1_inst

        mock_p2 = MagicMock()
        mock_p2_inst = MagicMock()
        mock_p2_inst.stream_markers = [("<tool_call>", "</tool_call>")]
        mock_p2.return_value = mock_p2_inst

        parsers = {"null": MagicMock, "hermes": mock_p1, "glm4": mock_p2}
        result = _build_marker_to_parsers(parsers)

        assert "<tool_call>" in result
        assert set(result["<tool_call>"]) == {"hermes", "glm4"}

    def test_skips_null_parser(self) -> None:
        mock_null = MagicMock()
        parsers = {"null": mock_null}
        result = _build_marker_to_parsers(parsers)
        assert result == {}

    def test_skips_empty_markers(self) -> None:
        mock_parser = MagicMock()
        mock_parser_inst = MagicMock()
        mock_parser_inst.stream_markers = [("", "")]
        mock_parser.return_value = mock_parser_inst

        parsers = {"null": MagicMock, "empty": mock_parser}
        result = _build_marker_to_parsers(parsers)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests for _scan_for_known_markers
# ---------------------------------------------------------------------------


class TestScanForKnownMarkers:
    """Tests for direct marker string scanning."""

    def test_finds_matching_markers(self) -> None:
        output = "[TOOL_CALLS] some tool call data"
        marker_map = {"[TOOL_CALLS]": ["mistral_native"], "<tool_call>": ["hermes_json"]}
        result = _scan_for_known_markers(output, marker_map)
        assert result == {"mistral_native"}

    def test_finds_multiple_markers(self) -> None:
        output = "<tool_call>data</tool_call>[TOOL_CALLS] more"
        marker_map = {"[TOOL_CALLS]": ["mistral"], "<tool_call>": ["hermes"]}
        result = _scan_for_known_markers(output, marker_map)
        assert result == {"mistral", "hermes"}

    def test_no_markers_found(self) -> None:
        output = "Just plain text"
        marker_map = {"[TOOL_CALLS]": ["mistral"], "<tool_call>": ["hermes"]}
        result = _scan_for_known_markers(output, marker_map)
        assert result == set()

    def test_empty_output(self) -> None:
        result = _scan_for_known_markers("", {"<tool_call>": ["hermes"]})
        assert result == set()

    def test_special_token_markers(self) -> None:
        output = "<|python_tag|>import os\n<|eom_id|>"
        marker_map = {"<|python_tag|>": ["llama_python"]}
        result = _scan_for_known_markers(output, marker_map)
        assert result == {"llama_python"}


# ---------------------------------------------------------------------------
# Tests for _discover_and_map_tags
# ---------------------------------------------------------------------------


class TestDiscoverAndMapTags:
    """Tests for the full discovery pipeline."""

    def test_xml_tag_mapped_to_parser(self) -> None:
        output = "<tool_call>some data</tool_call>"

        mock_parser = MagicMock()
        mock_parser_inst = MagicMock()
        mock_parser_inst.stream_markers = [("<tool_call>", "</tool_call>")]
        mock_parser.return_value = mock_parser_inst

        parsers = {"null": MagicMock, "hermes": mock_parser}
        tags = _discover_and_map_tags(output, parsers)

        tool_call_tags = [t for t in tags if t.name == "tool_call"]
        assert len(tool_call_tags) >= 1
        assert "hermes" in tool_call_tags[0].matched_parsers

    def test_bracket_tag_mapped_to_parser(self) -> None:
        output = "[TOOL_CALLS] some data"

        mock_parser = MagicMock()
        mock_parser_inst = MagicMock()
        mock_parser_inst.stream_markers = [("[TOOL_CALLS]", "")]
        mock_parser.return_value = mock_parser_inst

        parsers = {"null": MagicMock, "mistral": mock_parser}
        tags = _discover_and_map_tags(output, parsers)

        tool_tags = [t for t in tags if t.name == "TOOL_CALLS"]
        assert len(tool_tags) >= 1
        assert "mistral" in tool_tags[0].matched_parsers

    def test_unmatched_tags_have_empty_parsers(self) -> None:
        output = "<custom_tag>data</custom_tag>"

        parsers = {"null": MagicMock}  # No real parsers
        tags = _discover_and_map_tags(output, parsers)

        custom_tags = [t for t in tags if t.name == "custom_tag"]
        assert len(custom_tags) == 1
        assert custom_tags[0].matched_parsers == []

    def test_direct_scan_adds_parsers_not_found_by_regex(self) -> None:
        """Parsers with non-standard markers (like <function=) found by direct scan."""
        output = "<function=get_weather>{}</function>"

        mock_parser = MagicMock()
        mock_parser_inst = MagicMock()
        mock_parser_inst.stream_markers = [("<function=", "</function>")]
        mock_parser.return_value = mock_parser_inst

        parsers = {"null": MagicMock, "llama_xml": mock_parser}
        tags = _discover_and_map_tags(output, parsers)

        # The regex detects "function" as an XML tag, but the direct scan
        # also finds <function= and maps it to llama_xml.
        # Either way, llama_xml should be in some tag's matched_parsers
        all_matched = set()
        for t in tags:
            all_matched.update(t.matched_parsers)
        assert "llama_xml" in all_matched

    def test_returns_tag_discovery_objects(self) -> None:
        output = "<tool_call>data</tool_call>"
        parsers = {"null": MagicMock}
        tags = _discover_and_map_tags(output, parsers)

        assert all(isinstance(t, TagDiscovery) for t in tags)

    def test_empty_output(self) -> None:
        parsers = {"null": MagicMock}
        tags = _discover_and_map_tags("", parsers)
        assert tags == []
