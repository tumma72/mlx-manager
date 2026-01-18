"""
Dynamic discovery of parser options from mlx-openai-server.

This module imports the actual parser/converter dictionaries from the
mlx-openai-server package to stay in sync with the installed version.
Falls back to known values if the package is not available.
"""

import logging
from functools import lru_cache
from typing import TypedDict

logger = logging.getLogger(__name__)


class ParserOptions(TypedDict):
    """Available parser options from mlx-openai-server."""

    tool_call_parsers: list[str]
    reasoning_parsers: list[str]
    message_converters: list[str]


# Fallback values from mlx-openai-server 1.5.0
# Used when the package is not installed or import fails
FALLBACK_TOOL_PARSERS = {
    "functiongemma",
    "glm4_moe",
    "harmony",
    "hermes",
    "iquest_coder_v1",
    "minimax_m2",
    "nemotron3_nano",
    "qwen3",
    "qwen3_coder",
    "qwen3_moe",
    "qwen3_vl",
    "solar_open",
}

FALLBACK_REASONING_PARSERS = {
    "glm4_moe",
    "harmony",
    "hermes",
    "minimax_m2",
    "nemotron3_nano",
    "qwen3",
    "qwen3_moe",
    "qwen3_vl",
    "solar_open",
}

FALLBACK_MESSAGE_CONVERTERS = {
    "glm4_moe",
    "minimax",
    "minimax_m2",
    "nemotron3_nano",
    "qwen3_coder",
}


@lru_cache(maxsize=1)
def get_parser_options() -> ParserOptions:
    """
    Discover available parser options from mlx-openai-server.

    Results are cached for the lifetime of the process.

    Returns:
        Dictionary with tool_call_parsers, reasoning_parsers, and
        message_converters lists.
    """
    tool_parsers: set[str] = set()
    reasoning_parsers: set[str] = set()
    message_converters: set[str] = set()

    try:
        # Import from mlx-openai-server package (installed as 'app' module)
        from app.message_converters import MESSAGE_CONVERTER_MAP
        from app.parsers import (
            REASONING_PARSER_MAP,
            TOOL_PARSER_MAP,
            UNIFIED_PARSER_MAP,
        )

        # Tool parsers = TOOL_PARSER_MAP + UNIFIED_PARSER_MAP
        tool_parsers = set(TOOL_PARSER_MAP.keys()) | set(UNIFIED_PARSER_MAP.keys())

        # Reasoning parsers = REASONING_PARSER_MAP + UNIFIED_PARSER_MAP
        reasoning_parsers = set(REASONING_PARSER_MAP.keys()) | set(UNIFIED_PARSER_MAP.keys())

        # Message converters
        message_converters = set(MESSAGE_CONVERTER_MAP.keys())

        logger.info(
            f"Loaded parser options from mlx-openai-server: "
            f"{len(tool_parsers)} tool parsers, "
            f"{len(reasoning_parsers)} reasoning parsers, "
            f"{len(message_converters)} message converters"
        )

    except ImportError as e:
        # Fallback if mlx-openai-server not installed
        logger.warning(
            f"Could not import mlx-openai-server parser maps ({e}), using fallback values"
        )
        tool_parsers = FALLBACK_TOOL_PARSERS.copy()
        reasoning_parsers = FALLBACK_REASONING_PARSERS.copy()
        message_converters = FALLBACK_MESSAGE_CONVERTERS.copy()

    except Exception as e:
        # Unexpected error - use fallback
        logger.error(f"Error loading parser options: {e}, using fallback values")
        tool_parsers = FALLBACK_TOOL_PARSERS.copy()
        reasoning_parsers = FALLBACK_REASONING_PARSERS.copy()
        message_converters = FALLBACK_MESSAGE_CONVERTERS.copy()

    return {
        "tool_call_parsers": sorted(tool_parsers),
        "reasoning_parsers": sorted(reasoning_parsers),
        "message_converters": sorted(message_converters),
    }
