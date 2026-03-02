"""Tests for family-aware parser prioritization in probe system."""

from __future__ import annotations

# ── Family Detection Tests ────────────────────────────────────────


class TestFamilyDetection:
    """Verify specific model variants map to correct families."""

    def test_qwen3_coder_maps_to_nemotron(self):
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        assert detect_model_family("mlx-community/Qwen3-Coder-7B-4bit") == "nemotron"

    def test_qwen3_coder_next_maps_to_nemotron(self):
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        assert detect_model_family("mlx-community/Qwen3-Coder-Next-8B-4bit") == "nemotron"

    def test_nemotron_maps_to_nemotron(self):
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        assert detect_model_family("nvidia/Nemotron3-8B-Instruct-4bit") == "nemotron"

    def test_base_qwen3_still_maps_to_qwen(self):
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        assert detect_model_family("mlx-community/Qwen3-0.6B-4bit-DWQ") == "qwen"

    def test_qwen_vl_still_maps_to_qwen(self):
        from mlx_manager.mlx_server.models.adapters.registry import detect_model_family

        assert detect_model_family("mlx-community/Qwen2-VL-2B-Instruct-4bit") == "qwen"


# ── Parser Prioritization Tests ───────────────────────────────────


class TestParserPrioritization:
    """Verify family-declared parsers are tried first."""

    def test_family_parser_first_in_order(self):
        from mlx_manager.services.probe.base import _prioritize_parsers

        candidates = {"hermes_json", "glm4_xml", "qwen3_coder_xml"}
        ordered = _prioritize_parsers(candidates, "qwen3_coder_xml")
        assert ordered[0] == "qwen3_coder_xml"

    def test_family_parser_not_in_candidates(self):
        from mlx_manager.services.probe.base import _prioritize_parsers

        candidates = {"hermes_json", "glm4_xml"}
        ordered = _prioritize_parsers(candidates, "qwen3_coder_xml")
        assert ordered == sorted(candidates)

    def test_no_family_parser_falls_back_to_sorted(self):
        from mlx_manager.services.probe.base import _prioritize_parsers

        candidates = {"hermes_json", "glm4_xml", "qwen3_coder_xml"}
        ordered = _prioritize_parsers(candidates, None)
        assert ordered == sorted(candidates)

    def test_empty_candidates(self):
        from mlx_manager.services.probe.base import _prioritize_parsers

        assert _prioritize_parsers(set(), "hermes_json") == []

    def test_remaining_are_sorted_after_family(self):
        from mlx_manager.services.probe.base import _prioritize_parsers

        candidates = {"hermes_json", "glm4_native", "glm4_xml", "qwen3_coder_xml"}
        ordered = _prioritize_parsers(candidates, "qwen3_coder_xml")
        assert ordered == ["qwen3_coder_xml", "glm4_native", "glm4_xml", "hermes_json"]


# ── Family Parser ID Lookup Tests ─────────────────────────────────


class TestFamilyParserLookup:
    """Verify FamilyConfig parser ID extraction."""

    def test_nemotron_tool_parser(self):
        from mlx_manager.services.probe.base import get_family_tool_parser_id

        assert get_family_tool_parser_id("nemotron") == "qwen3_coder_xml"

    def test_qwen_tool_parser(self):
        from mlx_manager.services.probe.base import get_family_tool_parser_id

        assert get_family_tool_parser_id("qwen") == "hermes_json"

    def test_llama_tool_parser(self):
        from mlx_manager.services.probe.base import get_family_tool_parser_id

        assert get_family_tool_parser_id("llama") == "llama_xml"

    def test_default_family_returns_none(self):
        from mlx_manager.services.probe.base import get_family_tool_parser_id

        assert get_family_tool_parser_id("default") is None

    def test_none_family_returns_none(self):
        from mlx_manager.services.probe.base import get_family_tool_parser_id

        assert get_family_tool_parser_id(None) is None

    def test_unknown_family_returns_none(self):
        from mlx_manager.services.probe.base import get_family_tool_parser_id

        assert get_family_tool_parser_id("nonexistent") is None

    def test_nemotron_thinking_parser(self):
        from mlx_manager.services.probe.base import get_family_thinking_parser_id

        assert get_family_thinking_parser_id("nemotron") == "think_tag"

    def test_mistral_thinking_parser(self):
        from mlx_manager.services.probe.base import get_family_thinking_parser_id

        assert get_family_thinking_parser_id("mistral") == "mistral_think"

    def test_default_thinking_returns_none(self):
        from mlx_manager.services.probe.base import get_family_thinking_parser_id

        assert get_family_thinking_parser_id("default") is None
