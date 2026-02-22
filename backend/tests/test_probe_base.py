"""Tests for probe base classes to improve coverage.

Focuses on GenerativeProbe edge cases and error paths.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.services.probe.base import (
    GenerativeProbe,
    _build_marker_to_parsers,
    _detect_all_tags,
    _detect_unknown_thinking_tags,
    _detect_unknown_xml_tags,
    _discover_and_map_tags,
    _find_unclosed_thinking_tag,
    _scan_for_known_markers,
)
from mlx_manager.services.probe.steps import ProbeResult, TagDiscovery


class ConcreteProbe(GenerativeProbe):
    """Concrete implementation for testing GenerativeProbe."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.TEXT_GEN

    async def _generate(
        self,
        loaded,
        messages: list[dict],
        tools: list[dict] | None = None,
        template_options: dict[str, Any] | None = None,
        max_tokens: int = 800,
    ):
        """Mock implementation returns predefined output as TextResult."""
        from mlx_manager.mlx_server.models.ir import TextResult

        return TextResult(content=self._mock_output)

    def set_mock_output(self, output: str):
        """Helper to set output for test assertions."""
        self._mock_output = output

    async def probe(self, model_id: str, loaded, result: ProbeResult):
        """Required implementation for BaseProbe."""
        async for step in self._probe_generative_capabilities(model_id, loaded, result):
            yield step


# ---------------------------------------------------------------------------
# Tests for thinking verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_thinking_sweeps_all_parsers():
    """Test that thinking verification sweeps all parsers on raw output."""
    probe = ConcreteProbe()
    probe.set_mock_output("<think>Some thinking here</think>")

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer

    # Mock adapter (not used for parser lookup in new design)
    mock_adapter = MagicMock()

    # Mock a parser that will match during sweep
    mock_sweep_parser = MagicMock()
    mock_sweep_parser_instance = MagicMock()
    mock_sweep_parser_instance.extract.return_value = "Some thinking here"
    mock_sweep_parser.return_value = mock_sweep_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_sweep_parser},
    ):
        supports, parser_id, diagnostics = await probe._verify_thinking_support(
            mock_loaded, mock_adapter
        )

        assert supports is True
        assert parser_id == "think_tag"
        assert diagnostics == []
        mock_sweep_parser_instance.extract.assert_called_once()


@pytest.mark.asyncio
async def test_verify_thinking_fallback_when_no_parser_matches():
    """Test thinking reports unverified when template supports but no tags found in output."""
    probe = ConcreteProbe()
    probe.set_mock_output("4")  # Plain output without thinking tags

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer

    mock_adapter = MagicMock()

    # Mock sweep parsers that don't match
    mock_sweep_parser = MagicMock()
    mock_sweep_parser_instance = MagicMock()
    mock_sweep_parser_instance.extract.return_value = None
    mock_sweep_parser.return_value = mock_sweep_parser_instance

    mock_other_parser = MagicMock()
    mock_other_parser_instance = MagicMock()
    mock_other_parser_instance.extract.return_value = None
    mock_other_parser.return_value = mock_other_parser_instance

    with patch(
        "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
        {"null": MagicMock, "think_tag": mock_sweep_parser, "reasoning_tag": mock_other_parser},
    ):
        supports, parser_id, diagnostics = await probe._verify_thinking_support(
            mock_loaded, mock_adapter, template_params={"enable_thinking": {"default": True}}
        )

        assert supports is False
        assert parser_id == "null"
        # Should produce a diagnostic about unverified thinking
        assert len(diagnostics) == 1
        assert diagnostics[0].level.value == "warning"
        assert diagnostics[0].category.value == "thinking_dialect"


@pytest.mark.asyncio
async def test_verify_thinking_exception_handling():
    """Test that thinking verification handles generation exceptions gracefully."""
    probe = ConcreteProbe()

    # Make _generate raise an exception
    async def failing_generate(*args, **kwargs):
        raise RuntimeError("Generation failed")

    probe._generate = failing_generate

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer

    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.parser_id = "think_tag"
    mock_adapter.thinking_parser = mock_thinking_parser

    supports, parser_id, diagnostics = await probe._verify_thinking_support(
        mock_loaded, mock_adapter, template_params={"enable_thinking": {"default": True}}
    )

    # Generation failed — both paths fail, report as unverified
    assert supports is False
    assert parser_id == "null"
    # Should produce a diagnostic about generation failure
    assert len(diagnostics) == 1
    assert diagnostics[0].level.value == "warning"
    assert diagnostics[0].category.value == "thinking_dialect"
    assert "generation error" in diagnostics[0].message.lower()


# ---------------------------------------------------------------------------
# Tests for tool verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_tool_template_delivery_no_parser_match():
    """Test logging when template delivery produces output but no parser matches."""
    probe = ConcreteProbe()
    probe.set_mock_output("I'll check the weather for you")  # Natural language, no tool call

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = True

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.validates = MagicMock(return_value=False)
    mock_parser_cls.return_value = mock_parser_instance

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock, "hermes_json": mock_parser_cls},
        ),
    ):
        tool_format, parser_id, diagnostics = await probe._verify_tool_support(
            mock_loaded, mock_adapter
        )

        # Should return None when no parser matches
        assert tool_format is None
        assert parser_id is None
        assert diagnostics == []


@pytest.mark.asyncio
async def test_verify_tool_adapter_delivery_no_parser_match():
    """Test logging when adapter delivery produces output but no parser matches."""
    probe = ConcreteProbe()
    probe.set_mock_output("I'll check the weather for you")

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = False

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.validates = MagicMock(return_value=False)
    mock_parser_cls.return_value = mock_parser_instance

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock, "hermes_json": mock_parser_cls},
        ),
    ):
        tool_format, parser_id, diagnostics = await probe._verify_tool_support(
            mock_loaded, mock_adapter
        )

        assert tool_format is None
        assert parser_id is None
        assert diagnostics == []


@pytest.mark.asyncio
async def test_verify_tool_partial_markers_warning():
    """Test that tool-like output with get_weather hint produces ACTION_NEEDED diagnostic."""
    probe = ConcreteProbe()
    # Output has a tool_call tag AND get_weather hint → detected as unknown tool dialect
    probe.set_mock_output("<tool_call>get_weather incomplete without closing tag")

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = False

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.validates = MagicMock(return_value=False)
    mock_parser_cls.return_value = mock_parser_instance

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock, "hermes_json": mock_parser_cls},
        ),
    ):
        tool_format, parser_id, diagnostics = await probe._verify_tool_support(
            mock_loaded, mock_adapter
        )

        assert tool_format is None
        assert parser_id is None
        assert len(diagnostics) == 1
        assert diagnostics[0].level.value == "action_needed"
        assert diagnostics[0].category.value == "tool_dialect"


@pytest.mark.asyncio
async def test_verify_tool_unknown_xml_tags_warning():
    """Test that unknown XML tags produce a WARNING diagnostic."""
    probe = ConcreteProbe()
    probe.set_mock_output("<custom_tag>Some content</custom_tag>")

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = False

    mock_parser_cls = MagicMock()
    mock_parser_instance = MagicMock()
    mock_parser_instance.validates = MagicMock(return_value=False)
    mock_parser_cls.return_value = mock_parser_instance

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
            {"null": MagicMock, "hermes_json": mock_parser_cls},
        ),
    ):
        tool_format, parser_id, diagnostics = await probe._verify_tool_support(
            mock_loaded, mock_adapter
        )

        assert tool_format is None
        assert parser_id is None
        assert len(diagnostics) == 1
        assert diagnostics[0].level.value == "warning"
        assert diagnostics[0].category.value == "tool_dialect"


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
# Integration tests for _probe_generative_capabilities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_generative_capabilities_no_adapter():
    """Test that probing skips thinking/tools when adapter is None."""
    probe = ConcreteProbe()

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer
    mock_loaded.adapter = None  # No adapter

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="unknown",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"default": MagicMock},
        ),
    ):
        steps = []
        async for step in probe._probe_generative_capabilities("test/model", mock_loaded, result):
            steps.append(step)

        step_names = [s.step for s in steps]
        # detect_family (running + completed) + test_thinking (skipped) + test_tools (skipped)
        assert "detect_family" in step_names
        skipped = [s for s in steps if s.status == "skipped"]
        assert {s.step for s in skipped} == {"test_thinking", "test_tools"}


@pytest.mark.asyncio
async def test_probe_generative_capabilities_no_tokenizer():
    """Test that probing skips thinking/tools when tokenizer is None."""
    probe = ConcreteProbe()

    mock_loaded = MagicMock()
    mock_loaded.tokenizer = None  # No tokenizer
    mock_loaded.adapter = MagicMock()

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
    ):
        steps = []
        async for step in probe._probe_generative_capabilities("test/model", mock_loaded, result):
            steps.append(step)

        skipped = [s for s in steps if s.status == "skipped"]
        assert {s.step for s in skipped} == {"test_thinking", "test_tools"}


@pytest.mark.asyncio
async def test_probe_generative_capabilities_thinking_test_exception():
    """Test that thinking test failure is handled gracefully."""
    probe = ConcreteProbe()

    # Make thinking verification raise
    async def failing_verify(*args, **kwargs):
        raise RuntimeError("Thinking test crashed")

    probe._verify_thinking_support = failing_verify

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer
    mock_loaded.adapter = MagicMock()

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
        patch.object(probe, "_verify_tool_support", new_callable=AsyncMock),
    ):
        steps = []
        async for step in probe._probe_generative_capabilities("test/model", mock_loaded, result):
            steps.append(step)

        # Should have failed thinking step and completed/failed tools step
        thinking_steps = [s for s in steps if s.step == "test_thinking"]
        assert len(thinking_steps) == 2  # running + failed
        failed = next(s for s in thinking_steps if s.status == "failed")
        assert "Thinking test crashed" in failed.error


@pytest.mark.asyncio
async def test_probe_generative_capabilities_tool_test_exception():
    """Test that tool test failure is handled gracefully."""
    probe = ConcreteProbe()

    # Make tool verification raise
    async def failing_verify(*args, **kwargs):
        raise RuntimeError("Tool test crashed")

    probe._verify_tool_support = failing_verify

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer
    mock_loaded.adapter = MagicMock()

    result = ProbeResult()

    with (
        patch(
            "mlx_manager.mlx_server.models.adapters.detect_model_family",
            return_value="qwen",
        ),
        patch(
            "mlx_manager.mlx_server.models.adapters.FAMILY_REGISTRY",
            {"qwen": MagicMock},
        ),
        patch.object(
            probe,
            "_verify_thinking_support",
            new_callable=AsyncMock,
            return_value=(False, "null", []),
        ),
    ):
        steps = []
        async for step in probe._probe_generative_capabilities("test/model", mock_loaded, result):
            steps.append(step)

        # Should have failed tools step
        tools_steps = [s for s in steps if s.step == "test_tools"]
        assert len(tools_steps) == 2  # running + failed
        failed = next(s for s in tools_steps if s.status == "failed")
        assert "Tool test crashed" in failed.error


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
