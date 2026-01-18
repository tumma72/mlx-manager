"""Tests for the parser options service."""

from unittest.mock import MagicMock, patch

import pytest

from mlx_manager.services.parser_options import (
    FALLBACK_MESSAGE_CONVERTERS,
    FALLBACK_REASONING_PARSERS,
    FALLBACK_TOOL_PARSERS,
    get_parser_options,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the lru_cache before and after each test."""
    get_parser_options.cache_clear()
    yield
    get_parser_options.cache_clear()


class TestGetParserOptionsImportError:
    """Tests for ImportError handling in get_parser_options."""

    def test_import_error_returns_fallback_tool_parsers(self):
        """Test that ImportError returns fallback tool parsers."""
        modules_patch = {"app": None, "app.parsers": None, "app.message_converters": None}
        with patch.dict("sys.modules", modules_patch):
            with patch(
                "mlx_manager.services.parser_options.get_parser_options.__wrapped__",
                side_effect=None,
            ):
                # Force reimport by clearing cache and patching the import
                get_parser_options.cache_clear()

                # Mock the import to raise ImportError
                if hasattr(__builtins__, "__import__"):
                    original_import = __builtins__.__import__
                else:
                    original_import = __import__

                def mock_import(name, *args, **kwargs):
                    if name.startswith("app"):
                        raise ImportError(f"No module named '{name}'")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = get_parser_options()

        assert sorted(FALLBACK_TOOL_PARSERS) == result["tool_call_parsers"]

    def test_import_error_returns_fallback_reasoning_parsers(self):
        """Test that ImportError returns fallback reasoning parsers."""

        def mock_import(name, *args, **kwargs):
            if name.startswith("app"):
                raise ImportError(f"No module named '{name}'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = get_parser_options()

        assert sorted(FALLBACK_REASONING_PARSERS) == result["reasoning_parsers"]

    def test_import_error_returns_fallback_message_converters(self):
        """Test that ImportError returns fallback message converters."""

        def mock_import(name, *args, **kwargs):
            if name.startswith("app"):
                raise ImportError(f"No module named '{name}'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = get_parser_options()

        assert sorted(FALLBACK_MESSAGE_CONVERTERS) == result["message_converters"]

    def test_import_error_logs_warning(self):
        """Test that ImportError logs a warning message."""

        def mock_import(name, *args, **kwargs):
            if name.startswith("app"):
                raise ImportError(f"No module named '{name}'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("mlx_manager.services.parser_options.logger") as mock_logger:
                get_parser_options()

                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "Could not import mlx-openai-server" in warning_msg
                assert "using fallback values" in warning_msg


class TestGetParserOptionsGeneralException:
    """Tests for general Exception handling in get_parser_options."""

    def test_general_exception_returns_fallback_tool_parsers(self):
        """Test that general Exception returns fallback tool parsers."""
        # Create a mock module that raises an exception when accessing keys()
        mock_parser_map = MagicMock()
        mock_parser_map.keys.side_effect = RuntimeError("Unexpected error accessing keys")

        mock_app_parsers = MagicMock()
        mock_app_parsers.TOOL_PARSER_MAP = mock_parser_map
        mock_app_parsers.REASONING_PARSER_MAP = {}
        mock_app_parsers.UNIFIED_PARSER_MAP = {}

        mock_app_message_converters = MagicMock()
        mock_app_message_converters.MESSAGE_CONVERTER_MAP = {}

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(),
                "app.parsers": mock_app_parsers,
                "app.message_converters": mock_app_message_converters,
            },
        ):
            result = get_parser_options()

        assert sorted(FALLBACK_TOOL_PARSERS) == result["tool_call_parsers"]

    def test_general_exception_returns_fallback_reasoning_parsers(self):
        """Test that general Exception returns fallback reasoning parsers."""
        mock_parser_map = MagicMock()
        mock_parser_map.keys.side_effect = RuntimeError("Unexpected error")

        mock_app_parsers = MagicMock()
        mock_app_parsers.TOOL_PARSER_MAP = mock_parser_map
        mock_app_parsers.REASONING_PARSER_MAP = {}
        mock_app_parsers.UNIFIED_PARSER_MAP = {}

        mock_app_message_converters = MagicMock()
        mock_app_message_converters.MESSAGE_CONVERTER_MAP = {}

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(),
                "app.parsers": mock_app_parsers,
                "app.message_converters": mock_app_message_converters,
            },
        ):
            result = get_parser_options()

        assert sorted(FALLBACK_REASONING_PARSERS) == result["reasoning_parsers"]

    def test_general_exception_returns_fallback_message_converters(self):
        """Test that general Exception returns fallback message converters."""
        mock_parser_map = MagicMock()
        mock_parser_map.keys.side_effect = RuntimeError("Unexpected error")

        mock_app_parsers = MagicMock()
        mock_app_parsers.TOOL_PARSER_MAP = mock_parser_map
        mock_app_parsers.REASONING_PARSER_MAP = {}
        mock_app_parsers.UNIFIED_PARSER_MAP = {}

        mock_app_message_converters = MagicMock()
        mock_app_message_converters.MESSAGE_CONVERTER_MAP = {}

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(),
                "app.parsers": mock_app_parsers,
                "app.message_converters": mock_app_message_converters,
            },
        ):
            result = get_parser_options()

        assert sorted(FALLBACK_MESSAGE_CONVERTERS) == result["message_converters"]

    def test_general_exception_logs_error(self):
        """Test that general Exception logs an error message."""
        mock_parser_map = MagicMock()
        mock_parser_map.keys.side_effect = RuntimeError("Test runtime error")

        mock_app_parsers = MagicMock()
        mock_app_parsers.TOOL_PARSER_MAP = mock_parser_map
        mock_app_parsers.REASONING_PARSER_MAP = {}
        mock_app_parsers.UNIFIED_PARSER_MAP = {}

        mock_app_message_converters = MagicMock()
        mock_app_message_converters.MESSAGE_CONVERTER_MAP = {}

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(),
                "app.parsers": mock_app_parsers,
                "app.message_converters": mock_app_message_converters,
            },
        ):
            with patch("mlx_manager.services.parser_options.logger") as mock_logger:
                get_parser_options()

                mock_logger.error.assert_called_once()
                error_msg = mock_logger.error.call_args[0][0]
                assert "Error loading parser options" in error_msg
                assert "using fallback values" in error_msg


class TestGetParserOptionsCache:
    """Tests for lru_cache functionality in get_parser_options."""

    def test_cache_returns_same_result(self):
        """Test that cached function returns the same result on subsequent calls."""
        result1 = get_parser_options()
        result2 = get_parser_options()

        # Should be the exact same object due to caching
        assert result1 is result2

    def test_cache_info_shows_hits(self):
        """Test that cache_info shows cache hits after multiple calls."""
        get_parser_options.cache_clear()

        # First call - should be a miss
        get_parser_options()
        cache_info = get_parser_options.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits == 0

        # Second call - should be a hit
        get_parser_options()
        cache_info = get_parser_options.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits == 1

        # Third call - should be another hit
        get_parser_options()
        cache_info = get_parser_options.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits == 2

    def test_cache_clear_resets_cache(self):
        """Test that cache_clear resets the cache."""
        get_parser_options()
        get_parser_options()

        cache_info_before = get_parser_options.cache_info()
        assert cache_info_before.hits >= 1

        get_parser_options.cache_clear()

        cache_info_after = get_parser_options.cache_info()
        assert cache_info_after.hits == 0
        assert cache_info_after.misses == 0


class TestGetParserOptionsSuccess:
    """Tests for successful parser options loading."""

    def test_success_returns_sorted_tool_parsers(self):
        """Test that successful import returns sorted tool parsers."""
        mock_app_parsers = MagicMock()
        mock_app_parsers.TOOL_PARSER_MAP = {"parser_b": None, "parser_a": None}
        mock_app_parsers.REASONING_PARSER_MAP = {}
        mock_app_parsers.UNIFIED_PARSER_MAP = {"unified_c": None}

        mock_app_message_converters = MagicMock()
        mock_app_message_converters.MESSAGE_CONVERTER_MAP = {}

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(),
                "app.parsers": mock_app_parsers,
                "app.message_converters": mock_app_message_converters,
            },
        ):
            result = get_parser_options()

        # Tool parsers should include both TOOL_PARSER_MAP and UNIFIED_PARSER_MAP keys, sorted
        assert result["tool_call_parsers"] == ["parser_a", "parser_b", "unified_c"]

    def test_success_returns_sorted_reasoning_parsers(self):
        """Test that successful import returns sorted reasoning parsers."""
        mock_app_parsers = MagicMock()
        mock_app_parsers.TOOL_PARSER_MAP = {}
        mock_app_parsers.REASONING_PARSER_MAP = {"reason_z": None, "reason_a": None}
        mock_app_parsers.UNIFIED_PARSER_MAP = {"unified_m": None}

        mock_app_message_converters = MagicMock()
        mock_app_message_converters.MESSAGE_CONVERTER_MAP = {}

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(),
                "app.parsers": mock_app_parsers,
                "app.message_converters": mock_app_message_converters,
            },
        ):
            result = get_parser_options()

        # Reasoning parsers include REASONING_PARSER_MAP + UNIFIED_PARSER_MAP keys, sorted
        assert result["reasoning_parsers"] == ["reason_a", "reason_z", "unified_m"]

    def test_success_returns_sorted_message_converters(self):
        """Test that successful import returns sorted message converters."""
        mock_app_parsers = MagicMock()
        mock_app_parsers.TOOL_PARSER_MAP = {}
        mock_app_parsers.REASONING_PARSER_MAP = {}
        mock_app_parsers.UNIFIED_PARSER_MAP = {}

        mock_app_message_converters = MagicMock()
        mock_app_message_converters.MESSAGE_CONVERTER_MAP = {
            "conv_c": None,
            "conv_a": None,
            "conv_b": None,
        }

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(),
                "app.parsers": mock_app_parsers,
                "app.message_converters": mock_app_message_converters,
            },
        ):
            result = get_parser_options()

        assert result["message_converters"] == ["conv_a", "conv_b", "conv_c"]

    def test_success_logs_info(self):
        """Test that successful import logs an info message."""
        mock_app_parsers = MagicMock()
        mock_app_parsers.TOOL_PARSER_MAP = {"tool1": None, "tool2": None}
        mock_app_parsers.REASONING_PARSER_MAP = {"reason1": None}
        mock_app_parsers.UNIFIED_PARSER_MAP = {"unified1": None}

        mock_app_message_converters = MagicMock()
        mock_app_message_converters.MESSAGE_CONVERTER_MAP = {"conv1": None}

        with patch.dict(
            "sys.modules",
            {
                "app": MagicMock(),
                "app.parsers": mock_app_parsers,
                "app.message_converters": mock_app_message_converters,
            },
        ):
            with patch("mlx_manager.services.parser_options.logger") as mock_logger:
                get_parser_options()

                mock_logger.info.assert_called_once()
                info_msg = mock_logger.info.call_args[0][0]
                assert "Loaded parser options from mlx-openai-server" in info_msg
                assert "3 tool parsers" in info_msg  # 2 from TOOL + 1 from UNIFIED
                assert "2 reasoning parsers" in info_msg  # 1 from REASONING + 1 from UNIFIED
                assert "1 message converters" in info_msg


class TestFallbackValues:
    """Tests to verify fallback values are correctly defined."""

    def test_fallback_tool_parsers_is_set(self):
        """Test that FALLBACK_TOOL_PARSERS is a non-empty set."""
        assert isinstance(FALLBACK_TOOL_PARSERS, set)
        assert len(FALLBACK_TOOL_PARSERS) > 0

    def test_fallback_reasoning_parsers_is_set(self):
        """Test that FALLBACK_REASONING_PARSERS is a non-empty set."""
        assert isinstance(FALLBACK_REASONING_PARSERS, set)
        assert len(FALLBACK_REASONING_PARSERS) > 0

    def test_fallback_message_converters_is_set(self):
        """Test that FALLBACK_MESSAGE_CONVERTERS is a non-empty set."""
        assert isinstance(FALLBACK_MESSAGE_CONVERTERS, set)
        assert len(FALLBACK_MESSAGE_CONVERTERS) > 0

    def test_fallback_values_are_strings(self):
        """Test that all fallback values contain strings."""
        for parser in FALLBACK_TOOL_PARSERS:
            assert isinstance(parser, str)

        for parser in FALLBACK_REASONING_PARSERS:
            assert isinstance(parser, str)

        for converter in FALLBACK_MESSAGE_CONVERTERS:
            assert isinstance(converter, str)
