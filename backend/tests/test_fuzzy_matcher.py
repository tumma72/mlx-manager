"""Tests for the fuzzy matcher module.

NOTE: Parser options are deprecated with the embedded MLX Server.
The fuzzy matcher code is kept for backwards compatibility but the
main entry points (find_parser_options, get_parser_options) return
empty results.

These tests ensure the internal methods still function correctly
for any future use cases.
"""

from unittest.mock import patch


class TestGetParserOptions:
    """Test get_parser_options returns empty (deprecated)."""

    def test_returns_empty_dict(self):
        """Parser options are deprecated, should return empty."""
        from mlx_manager.utils.fuzzy_matcher import get_parser_options

        result = get_parser_options()
        assert result == {
            "tool_call_parsers": [],
            "reasoning_parsers": [],
            "message_converters": [],
        }


class TestFindParserOptions:
    """Test find_parser_options returns empty (deprecated)."""

    def test_returns_empty_dict_for_any_model(self):
        """find_parser_options should always return empty dict."""
        from mlx_manager.utils.fuzzy_matcher import find_parser_options

        # All models now return empty dict
        assert find_parser_options("mlx-community/Qwen3-Coder-7B-4bit") == {}
        assert find_parser_options("mlx-community/Qwen3-8B-4bit") == {}
        assert find_parser_options("mlx-community/Llama-3.1-70B-4bit") == {}
        assert find_parser_options("mlx-community/MiniMax-M2.1-3bit") == {}
        assert find_parser_options("mlx-community/GLM-4-MoE") == {}


class TestMatcherClasses:
    """Test that matcher classes still exist for backwards compatibility."""

    def test_rapidfuzz_matcher_exists(self):
        """RapidfuzzMatcher class should exist."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        assert matcher is not None

    def test_difflib_matcher_exists(self):
        """DifflibMatcher class should exist."""
        from mlx_manager.utils.fuzzy_matcher import DifflibMatcher

        matcher = DifflibMatcher()
        assert matcher is not None

    def test_get_matcher_returns_rapidfuzz(self):
        """get_matcher should return RapidfuzzMatcher."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher, get_matcher

        matcher = get_matcher()
        assert isinstance(matcher, RapidfuzzMatcher)


# ============================================================================
# Internal Matcher Method Tests
# ============================================================================


class TestNormalizeForMatching:
    """Tests for _normalize_for_matching method."""

    def test_lowercase_conversion(self):
        """Converts to lowercase."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        assert matcher._normalize_for_matching("QWEN3") == "qwen3"
        assert matcher._normalize_for_matching("GPT-4-Turbo") == "gpt-4-turbo"

    def test_underscore_to_hyphen(self):
        """Replaces underscores with hyphens."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        assert matcher._normalize_for_matching("qwen3_coder") == "qwen3-coder"
        assert matcher._normalize_for_matching("glm4_moe") == "glm4-moe"


class TestExtractBaseFamily:
    """Tests for _extract_base_family method."""

    def test_extracts_base_family(self):
        """Extracts the base family name."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        assert matcher._extract_base_family("qwen3") == "qwen"
        assert matcher._extract_base_family("qwen3_coder") == "qwen"
        assert matcher._extract_base_family("glm4_moe") == "glm"
        assert matcher._extract_base_family("nemotron3_nano") == "nemotron"
        assert matcher._extract_base_family("minimax_m2") == "minimax"
        assert matcher._extract_base_family("solar_open") == "solar"
        assert matcher._extract_base_family("hermes") == "hermes"


class TestExtractVariant:
    """Tests for _extract_variant method."""

    def test_extracts_variant_suffix(self):
        """Extracts the variant suffix."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        assert matcher._extract_variant("qwen3_coder") == "coder"
        assert matcher._extract_variant("qwen3_moe") == "moe"
        assert matcher._extract_variant("qwen3_vl") == "vl"
        assert matcher._extract_variant("glm4_moe") == "moe"
        assert matcher._extract_variant("minimax_m2") == "m2"

    def test_no_variant_returns_none(self):
        """Returns None when no variant."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        assert matcher._extract_variant("qwen3") is None
        assert matcher._extract_variant("hermes") is None


class TestContainsOptionTokens:
    """Tests for _contains_option_tokens method."""

    def test_base_family_must_be_present(self):
        """Base family must be in model name."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        # qwen family matches qwen models
        assert matcher._contains_option_tokens("mlx-community/qwen3-8b", "qwen3")
        # qwen family doesn't match llama models
        assert not matcher._contains_option_tokens("mlx-community/llama-3-8b", "qwen3")

    def test_variant_must_be_present_for_variant_options(self):
        """Variant must be in model name for variant options."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        # qwen3_coder matches models with "coder" in name
        assert matcher._contains_option_tokens("mlx-community/qwen3-coder-7b", "qwen3_coder")
        # qwen3_coder doesn't match base qwen3 models
        assert not matcher._contains_option_tokens("mlx-community/qwen3-8b", "qwen3_coder")

    def test_m2_variant_special_handling(self):
        """M2 variant has special handling for m2.x patterns."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        # m2 variant matches m2.1 pattern (tests line 110-111 direct match)
        assert matcher._contains_option_tokens("mlx-community/minimax-m2.1-3b", "minimax_m2")
        # Test various m2 patterns
        assert matcher._contains_option_tokens("mlx-community/minimax-m2-3b", "minimax_m2")
        assert matcher._contains_option_tokens("mlx-community/minimax-m2.5-3b", "minimax_m2")
        assert matcher._contains_option_tokens("mlx-community/minimax-m23-3b", "minimax_m2")
        # Test that models without m2 pattern don't match
        assert not matcher._contains_option_tokens("mlx-community/minimax-text-3b", "minimax_m2")

    def test_base_option_doesnt_match_variant_models(self):
        """Base options don't match models with known variants."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        # Base qwen3 option doesn't match coder models
        assert not matcher._contains_option_tokens("mlx-community/qwen3-coder-7b", "qwen3")
        # Base option doesn't match moe models
        assert not matcher._contains_option_tokens("mlx-community/qwen3-moe-8b", "qwen3")


class TestGroupOptionsByFamily:
    """Tests for _group_options_by_family method."""

    def test_groups_options_correctly(self):
        """Groups options by base family."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        options = ["qwen3", "qwen3_coder", "glm4_moe", "hermes"]
        groups = matcher._group_options_by_family(options)

        assert "qwen" in groups
        assert "qwen3" in groups["qwen"]
        assert "qwen3_coder" in groups["qwen"]
        assert "glm" in groups
        assert "glm4_moe" in groups["glm"]
        assert "hermes" in groups


class TestFindBestMatch:
    """Tests for find_best_match method."""

    def test_returns_none_for_empty_options(self):
        """Returns None when no options available."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        # Since get_parser_options returns empty, all find_best_match return None
        result = matcher.find_best_match("mlx-community/Qwen3-8B", "tool_call_parser")
        assert result is None

    def test_returns_none_for_unknown_parser_type(self):
        """Returns None for unknown parser types."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        result = matcher.find_best_match("mlx-community/Qwen3-8B", "unknown_parser")
        assert result is None

    def test_find_best_match_with_mocked_options_tool_call_parser(self):
        """Test find_best_match with mocked options for tool_call_parser (covers lines 166-205)."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()

        # Mock get_parser_options to return test options
        with patch("mlx_manager.utils.fuzzy_matcher.get_parser_options") as mock_get_opts:
            mock_get_opts.return_value = {
                "tool_call_parsers": ["qwen3", "qwen3_coder", "glm4_moe"],
                "reasoning_parsers": [],
                "message_converters": [],
            }

            # Test matching qwen3_coder (more specific than qwen3)
            result = matcher.find_best_match(
                "mlx-community/Qwen3-Coder-7B-4bit", "tool_call_parser"
            )
            assert result == "qwen3_coder"

            # Test matching base qwen3 (no variant)
            result = matcher.find_best_match("mlx-community/Qwen3-8B-4bit", "tool_call_parser")
            assert result == "qwen3"

            # Test matching glm4_moe
            result = matcher.find_best_match("mlx-community/GLM-4-MoE-8B", "tool_call_parser")
            assert result == "glm4_moe"

    def test_find_best_match_with_mocked_options_message_converter(self):
        """Test find_best_match with message_converter type (different threshold path)."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()

        with patch("mlx_manager.utils.fuzzy_matcher.get_parser_options") as mock_get_opts:
            mock_get_opts.return_value = {
                "tool_call_parsers": [],
                "reasoning_parsers": [],
                "message_converters": ["qwen3", "qwen3_coder"],
            }

            # Test message_converter doesn't match base option for variant models
            result = matcher.find_best_match("mlx-community/Qwen3-Coder-7B", "message_converter")
            assert result == "qwen3_coder"

            # Test base option matches when no variant
            result = matcher.find_best_match("mlx-community/Qwen3-8B", "message_converter")
            assert result == "qwen3"

    def test_find_best_match_single_family_fallback(self):
        """Test single-family fallback (lines 189-196) for non-message_converter."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()

        with patch("mlx_manager.utils.fuzzy_matcher.get_parser_options") as mock_get_opts:
            # Only one option in glm family
            mock_get_opts.return_value = {
                "tool_call_parsers": ["glm4_moe"],
                "reasoning_parsers": [],
                "message_converters": [],
            }

            # Should match glm4_moe even though model doesn't have "moe" in name
            # because it's the only option in the glm family
            result = matcher.find_best_match("mlx-community/GLM-4-9B", "tool_call_parser")
            assert result == "glm4_moe"

    def test_find_best_match_no_single_family_fallback_for_message_converter(self):
        """Test that single-family fallback is skipped for message_converter (line 189)."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()

        with patch("mlx_manager.utils.fuzzy_matcher.get_parser_options") as mock_get_opts:
            mock_get_opts.return_value = {
                "tool_call_parsers": [],
                "reasoning_parsers": [],
                "message_converters": ["qwen3_coder"],
            }

            # Should NOT match qwen3_coder for base Qwen3 (no fallback for message_converter)
            result = matcher.find_best_match("mlx-community/Qwen3-8B", "message_converter")
            assert result is None

    def test_find_best_match_prefers_longer_matches(self):
        """Test that longer/more specific options are preferred (lines 201-203)."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()

        with patch("mlx_manager.utils.fuzzy_matcher.get_parser_options") as mock_get_opts:
            mock_get_opts.return_value = {
                "tool_call_parsers": ["qwen3", "qwen3_coder"],
                "reasoning_parsers": [],
                "message_converters": [],
            }

            # For a coder model, qwen3_coder (longer) should be preferred over qwen3
            result = matcher.find_best_match("mlx-community/Qwen3-Coder-7B", "tool_call_parser")
            assert result == "qwen3_coder"

    def test_find_best_match_no_candidates_returns_none(self):
        """Test that no candidates returns None (line 198-199)."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()

        with patch("mlx_manager.utils.fuzzy_matcher.get_parser_options") as mock_get_opts:
            mock_get_opts.return_value = {
                "tool_call_parsers": ["qwen3"],
                "reasoning_parsers": [],
                "message_converters": [],
            }

            # Llama model won't match qwen3 option
            result = matcher.find_best_match("mlx-community/Llama-3-8B", "tool_call_parser")
            assert result is None


class TestRapidfuzzMatcherPartialScore:
    """Tests for RapidfuzzMatcher._calculate_partial_score."""

    def test_calculates_partial_score(self):
        """Calculates partial match score."""
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        matcher = RapidfuzzMatcher()
        # High score for contained substring
        score = matcher._calculate_partial_score("qwen3-coder-7b-4bit", "qwen3")
        assert score >= 50

        # Lower score for non-matching strings
        score = matcher._calculate_partial_score("llama-3-8b", "qwen3")
        assert score < 50


class TestDifflibMatcherPartialScore:
    """Tests for DifflibMatcher._calculate_partial_score."""

    def test_calculates_partial_score(self):
        """Calculates partial match score using difflib."""
        from mlx_manager.utils.fuzzy_matcher import DifflibMatcher

        matcher = DifflibMatcher()
        # High score for contained substring
        score = matcher._calculate_partial_score("qwen3-coder-7b-4bit", "qwen3")
        assert score >= 50

    def test_empty_string_handling(self):
        """Handles empty strings correctly."""
        from mlx_manager.utils.fuzzy_matcher import DifflibMatcher

        matcher = DifflibMatcher()
        # Empty s2 returns 0
        score = matcher._calculate_partial_score("test", "")
        assert score == 0

        # Empty s1 returns 0
        score = matcher._calculate_partial_score("", "test")
        assert score == 0


class TestGetMatcherFallback:
    """Tests for get_matcher fallback behavior."""

    def test_fallback_to_difflib_when_rapidfuzz_unavailable(self):
        """Falls back to DifflibMatcher when rapidfuzz not installed."""
        # Clear the cache first
        from mlx_manager.utils import fuzzy_matcher
        from mlx_manager.utils.fuzzy_matcher import DifflibMatcher

        fuzzy_matcher.get_matcher.cache_clear()

        with patch.dict("sys.modules", {"rapidfuzz": None}):
            # Mock the import to fail
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if name == "rapidfuzz":
                    raise ImportError("No module named 'rapidfuzz'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                # Get fresh matcher
                fuzzy_matcher.get_matcher.cache_clear()
                matcher = fuzzy_matcher.get_matcher()
                assert isinstance(matcher, DifflibMatcher)

        # Reset cache for other tests
        fuzzy_matcher.get_matcher.cache_clear()
