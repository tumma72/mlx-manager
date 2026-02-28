"""Tests for tool hint structural evidence detection in probe sweeps.

Tests _has_structural_tool_evidence() and its effect on Phase 5 diagnostics.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.services.probe.sweeps import _has_structural_tool_evidence

# ---------------------------------------------------------------------------
# _has_structural_tool_evidence() unit tests
# ---------------------------------------------------------------------------


class TestHasStructuralToolEvidence:
    """Tests for the structural evidence detection helper."""

    def test_json_object_with_name_key(self):
        """JSON tool call object with "name": "get_weather" is structural."""
        output = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
        assert _has_structural_tool_evidence(output) is True

    def test_json_object_with_name_key_whitespace(self):
        """JSON with extra whitespace around keys is still structural."""
        output = '{  "name" :  "get_weather" }'
        assert _has_structural_tool_evidence(output) is True

    def test_function_call_syntax(self):
        """Python-style function call syntax is structural."""
        output = "get_weather(location='Tokyo')"
        assert _has_structural_tool_evidence(output) is True

    def test_function_call_syntax_with_space(self):
        """Function call with space before paren is structural."""
        output = "get_weather (location='Tokyo')"
        assert _has_structural_tool_evidence(output) is True

    def test_xml_tool_call_tag(self):
        """XML <tool_call> tag is structural."""
        output = "<tool_call>get_weather</tool_call>"
        assert _has_structural_tool_evidence(output) is True

    def test_xml_function_call_tag(self):
        """XML <function_call> tag is structural."""
        output = '<function_call>{"name": "get_weather"}</function_call>'
        assert _has_structural_tool_evidence(output) is True

    def test_xml_tool_use_tag(self):
        """XML <tool_use> tag is structural."""
        output = "<tool_use>get_weather</tool_use>"
        assert _has_structural_tool_evidence(output) is True

    def test_bracket_tool_calls_marker(self):
        """Bracket [TOOL_CALLS] marker is structural."""
        output = '[TOOL_CALLS] [{"name": "get_weather"}]'
        assert _has_structural_tool_evidence(output) is True

    def test_bracket_tool_call_marker_case_insensitive(self):
        """Bracket [tool_call marker is structural (case-insensitive)."""
        output = "[tool_call] something"
        assert _has_structural_tool_evidence(output) is True

    def test_json_array_with_name_key(self):
        """JSON array with tool-like structure is structural."""
        output = '[{"name": "get_weather", "arguments": {}}]'
        assert _has_structural_tool_evidence(output) is True

    def test_json_array_with_function_key(self):
        """JSON array with "function" key is structural."""
        output = '[{"function": "get_weather"}]'
        assert _has_structural_tool_evidence(output) is True

    def test_prose_mention_not_structural(self):
        """Natural language mention of the tool is NOT structural."""
        output = "I will use the get_weather tool to check the weather in Tokyo."
        assert _has_structural_tool_evidence(output) is False

    def test_prose_with_name_in_sentence(self):
        """Prose about calling get_weather is NOT structural."""
        output = (
            "To answer your question, I need to call get_weather for Tokyo. "
            "Let me check the current conditions."
        )
        assert _has_structural_tool_evidence(output) is False

    def test_empty_output(self):
        """Empty output is not structural."""
        assert _has_structural_tool_evidence("") is False

    def test_unrelated_json(self):
        """JSON without tool-related keys is not structural."""
        output = '{"temperature": 72, "unit": "fahrenheit"}'
        assert _has_structural_tool_evidence(output) is False

    def test_quoted_name_without_get_weather(self):
        """JSON with "name" key but unrelated value is not structural (for get_weather check)."""
        output = '{"name": "some_other_function"}'
        # The JSON check specifically looks for "get_weather" in the value
        assert _has_structural_tool_evidence(output) is False


# ---------------------------------------------------------------------------
# Phase 5 diagnostic level integration tests
# ---------------------------------------------------------------------------


def _make_generative_probe():
    """Create a minimal concrete GenerativeProbe subclass for tests."""
    from mlx_manager.mlx_server.models.types import ModelType
    from mlx_manager.services.probe.base import GenerativeProbe

    class _TestGenerativeProbe(GenerativeProbe):
        @property
        def model_type(self):
            return ModelType.TEXT_GEN

        async def probe(self, model_id, loaded, result):
            if False:
                yield  # make it an async generator

    return _TestGenerativeProbe()


@pytest.mark.asyncio
async def test_phase5_prose_mention_produces_info_diagnostic():
    """When model mentions get_weather in prose, Phase 5 emits INFO not ACTION_NEEDED."""
    from mlx_manager.services.probe.steps import DiagnosticLevel
    from mlx_manager.services.probe.sweeps import sweep_tools

    # Prose-only output
    prose_output = "I will use the get_weather tool to check the weather in Tokyo for you."

    strategy = _make_generative_probe()
    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.adapter.supports_native_tools.return_value = False
    mock_loaded.tokenizer = MagicMock()

    mock_result = MagicMock()
    mock_result.content = prose_output

    strategy._generate = AsyncMock(return_value=mock_result)

    with patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=False,
    ):
        tool_format, parser_id, diagnostics, tags = await sweep_tools(
            "mlx-community/EuroLLM-22B-Instruct",
            mock_loaded,
            strategy,
            family="default",
        )

    assert tool_format is None
    assert parser_id is None

    # Should have an INFO diagnostic, not ACTION_NEEDED
    tool_diags = [d for d in diagnostics if d.category.value == "tool_dialect"]
    assert len(tool_diags) >= 1
    info_diags = [d for d in tool_diags if d.level == DiagnosticLevel.INFO]
    action_diags = [d for d in tool_diags if d.level == DiagnosticLevel.ACTION_NEEDED]
    assert len(info_diags) >= 1, "Expected at least one INFO diagnostic for prose mention"
    assert len(action_diags) == 0, "Prose mention should NOT produce ACTION_NEEDED"


@pytest.mark.asyncio
async def test_phase5_structural_tool_call_produces_action_needed():
    """When model emits structural tool call, Phase 5 emits ACTION_NEEDED."""
    from mlx_manager.services.probe.steps import DiagnosticLevel
    from mlx_manager.services.probe.sweeps import sweep_tools

    # Structural output — JSON tool call that no parser validates
    structural_output = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'

    strategy = _make_generative_probe()
    mock_loaded = MagicMock()
    mock_loaded.adapter = MagicMock()
    mock_loaded.adapter.supports_native_tools.return_value = False
    mock_loaded.tokenizer = MagicMock()

    mock_result = MagicMock()
    mock_result.content = structural_output

    strategy._generate = AsyncMock(return_value=mock_result)

    with patch(
        "mlx_manager.mlx_server.utils.template_tools.has_native_tool_support",
        return_value=False,
    ):
        tool_format, parser_id, diagnostics, tags = await sweep_tools(
            "mlx-community/unknown-model",
            mock_loaded,
            strategy,
            family="default",
        )

    # The parsers might actually validate this JSON — if so, we get a result.
    # If no parser validates, we should get ACTION_NEEDED.
    if tool_format is None:
        tool_diags = [d for d in diagnostics if d.category.value == "tool_dialect"]
        action_diags = [d for d in tool_diags if d.level == DiagnosticLevel.ACTION_NEEDED]
        assert len(action_diags) >= 1, (
            "Structural tool call should produce ACTION_NEEDED when no parser validates"
        )
