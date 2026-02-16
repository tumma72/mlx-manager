"""Tests for probe base classes to improve coverage.

Focuses on GenerativeProbe edge cases and error paths.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.services.probe.base import (
    GenerativeProbe,
    _detect_unknown_xml_tags,
    _find_matching_parser,
    _validate_tool_output,
)
from mlx_manager.services.probe.steps import ProbeResult


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
    ) -> str:
        """Mock implementation returns predefined output."""
        return self._mock_output

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
    """Test that thinking verification sweeps all parsers when adapter parser fails."""
    probe = ConcreteProbe()
    probe.set_mock_output("<think>Some thinking here</think>")

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer

    # Mock adapter with null parser
    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.parser_id = "null"
    mock_adapter.thinking_parser = mock_thinking_parser

    # Mock a parser that will match during sweep
    mock_sweep_parser = MagicMock()
    mock_sweep_parser_instance = MagicMock()
    mock_sweep_parser_instance.extract.return_value = "Some thinking here"
    mock_sweep_parser.return_value = mock_sweep_parser_instance

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_thinking_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
            {"null": MagicMock, "think_tag": mock_sweep_parser},
        ),
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

    # Mock adapter with non-null parser that doesn't match
    mock_adapter = MagicMock()
    mock_thinking_parser = MagicMock()
    mock_thinking_parser.parser_id = "think_tag"
    mock_thinking_parser.extract.return_value = None  # Doesn't match
    mock_adapter.thinking_parser = mock_thinking_parser

    # Mock sweep parser that also doesn't match
    mock_sweep_parser = MagicMock()
    mock_sweep_parser_instance = MagicMock()
    mock_sweep_parser_instance.extract.return_value = None
    mock_sweep_parser.return_value = mock_sweep_parser_instance

    # Mock another parser that also doesn't match
    mock_other_parser = MagicMock()
    mock_other_parser_instance = MagicMock()
    mock_other_parser_instance.extract.return_value = None
    mock_other_parser.return_value = mock_other_parser_instance

    with (
        patch(
            "mlx_manager.mlx_server.parsers.THINKING_PARSERS",
            {"null": MagicMock, "think_tag": mock_sweep_parser, "reasoning_tag": mock_other_parser},
        ),
    ):
        supports, parser_id, diagnostics = await probe._verify_thinking_support(
            mock_loaded, mock_adapter, template_params={"enable_thinking": {"default": True}}
        )

        # Probe couldn't verify with enable_thinking, fallback to always-thinks path also failed
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
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"
    mock_adapter.tool_parser.validates.return_value = False
    mock_adapter.format_tools_for_prompt.return_value = None

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=True,
        ),
        patch(
            "mlx_manager.services.probe.base._validate_tool_output",
            return_value=None,
        ),
        patch("mlx_manager.services.probe.base._find_matching_parser", return_value=None),
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
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"
    mock_adapter.format_tools_for_prompt.return_value = "Available tools: get_weather"

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base._validate_tool_output",
            return_value=None,
        ),
        patch("mlx_manager.services.probe.base._find_matching_parser", return_value=None),
    ):
        tool_format, parser_id, diagnostics = await probe._verify_tool_support(
            mock_loaded, mock_adapter
        )

        assert tool_format is None
        assert parser_id is None
        assert diagnostics == []


@pytest.mark.asyncio
async def test_verify_tool_partial_markers_warning():
    """Test that partial tool markers produce an ACTION_NEEDED diagnostic."""
    probe = ConcreteProbe()
    probe.set_mock_output("<tool_call>Incomplete tool call without closing tag")

    mock_loaded = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.chat_template = None
    mock_tokenizer.tokenizer = None  # Prevent processor.tokenizer fallback
    mock_loaded.tokenizer = mock_tokenizer

    mock_adapter = MagicMock()
    mock_adapter.supports_native_tools.return_value = False
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"
    mock_adapter.format_tools_for_prompt.return_value = None

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base._validate_tool_output",
            return_value=None,
        ),
        patch("mlx_manager.services.probe.base._find_matching_parser", return_value=None),
    ):
        tool_format, parser_id, diagnostics = await probe._verify_tool_support(
            mock_loaded, mock_adapter
        )

        assert tool_format is None
        assert parser_id is None
        assert len(diagnostics) == 1
        assert diagnostics[0].level.value == "action_needed"
        assert diagnostics[0].category.value == "tool_dialect"
        assert "<tool_call>" in diagnostics[0].details["found_markers"]


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
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"
    mock_adapter.format_tools_for_prompt.return_value = None

    with (
        patch(
            "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
            return_value=False,
        ),
        patch(
            "mlx_manager.services.probe.base._validate_tool_output",
            return_value=None,
        ),
        patch("mlx_manager.services.probe.base._find_matching_parser", return_value=None),
    ):
        tool_format, parser_id, diagnostics = await probe._verify_tool_support(
            mock_loaded, mock_adapter
        )

        assert tool_format is None
        assert parser_id is None
        assert len(diagnostics) == 1
        assert diagnostics[0].level.value == "warning"
        assert diagnostics[0].category.value == "tool_dialect"
        assert "custom_tag" in diagnostics[0].details["unknown_tags"]


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------


def test_find_matching_parser_excludes_null():
    """Test that _find_matching_parser skips 'null' parser."""
    mock_parser_cls = MagicMock()
    mock_parser = MagicMock()
    mock_parser.validates.return_value = True
    mock_parser_cls.return_value = mock_parser

    with patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock, "xml_tag": mock_parser_cls},
    ):
        result = _find_matching_parser("<tool_call>get_weather</tool_call>", "get_weather")

        assert result == "xml_tag"
        # null parser should not be instantiated
        mock_parser.validates.assert_called_once()


def test_find_matching_parser_excludes_specified_parser():
    """Test that _find_matching_parser skips the excluded parser."""
    mock_parser1_cls = MagicMock()
    mock_parser1 = MagicMock()
    mock_parser1.validates.return_value = True
    mock_parser1_cls.return_value = mock_parser1

    mock_parser2_cls = MagicMock()
    mock_parser2 = MagicMock()
    mock_parser2.validates.return_value = False
    mock_parser2_cls.return_value = mock_parser2

    with patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {
            "null": MagicMock,
            "xml_tag": mock_parser1_cls,
            "json_tag": mock_parser2_cls,
        },
    ):
        # Exclude xml_tag, should try json_tag but it returns False
        result = _find_matching_parser(
            "<tool_call>get_weather</tool_call>",
            "get_weather",
            exclude_parser_id="xml_tag",
        )

        assert result is None
        # xml_tag should not be called
        mock_parser1.validates.assert_not_called()
        # json_tag should be called
        mock_parser2.validates.assert_called_once()


def test_find_matching_parser_returns_none_when_no_match():
    """Test that _find_matching_parser returns None when no parser matches."""
    mock_parser_cls = MagicMock()
    mock_parser = MagicMock()
    mock_parser.validates.return_value = False
    mock_parser_cls.return_value = mock_parser

    with patch(
        "mlx_manager.mlx_server.parsers.TOOL_PARSERS",
        {"null": MagicMock, "xml_tag": mock_parser_cls},
    ):
        result = _find_matching_parser("No tool call here", "get_weather")

        assert result is None


def test_validate_tool_output_uses_adapter_parser_first():
    """Test that _validate_tool_output tries adapter parser before sweep."""
    mock_adapter = MagicMock()
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "xml_tag"
    mock_adapter.tool_parser.validates.return_value = True

    with patch("mlx_manager.services.probe.base._find_matching_parser") as mock_find:
        result = _validate_tool_output(
            "<tool_call>get_weather</tool_call>",
            "get_weather",
            mock_adapter,
        )

        assert result == "xml_tag"
        # Should not call sweep if adapter parser matches
        mock_find.assert_not_called()


def test_validate_tool_output_sweeps_when_adapter_parser_null():
    """Test that _validate_tool_output sweeps when adapter has null parser."""
    mock_adapter = MagicMock()
    mock_adapter.tool_parser = MagicMock()
    mock_adapter.tool_parser.parser_id = "null"

    with patch(
        "mlx_manager.services.probe.base._find_matching_parser",
        return_value="json_tag",
    ) as mock_find:
        result = _validate_tool_output(
            '{"name": "get_weather"}',
            "get_weather",
            mock_adapter,
        )

        assert result == "json_tag"
        mock_find.assert_called_once_with(
            '{"name": "get_weather"}',
            "get_weather",
            exclude_parser_id=None,
        )


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
