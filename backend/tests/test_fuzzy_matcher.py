"""Tests for the fuzzy matcher module.

NOTE: Parser options are deprecated with the embedded MLX Server.
The fuzzy matcher code is kept for backwards compatibility but the
main entry points (find_parser_options, get_parser_options) return
empty results.
"""

import pytest


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
