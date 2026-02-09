"""Parser registry: string ID -> parser instance resolution."""

from __future__ import annotations

from mlx_manager.mlx_server.parsers.base import ThinkingParser, ToolCallParser
from mlx_manager.mlx_server.parsers.thinking import NullThinkingParser, ThinkTagParser
from mlx_manager.mlx_server.parsers.tool_call import (
    Glm4NativeParser,
    Glm4XmlParser,
    HermesJsonParser,
    LlamaPythonParser,
    LlamaXmlParser,
    NullToolParser,
)

# Registry mapping parser_id strings to parser classes
TOOL_PARSERS: dict[str, type[ToolCallParser]] = {
    "hermes_json": HermesJsonParser,
    "glm4_native": Glm4NativeParser,
    "glm4_xml": Glm4XmlParser,
    "llama_xml": LlamaXmlParser,
    "llama_python": LlamaPythonParser,
    "null": NullToolParser,
}

THINKING_PARSERS: dict[str, type[ThinkingParser]] = {
    "think_tag": ThinkTagParser,
    "null": NullThinkingParser,
}


def resolve_tool_parser(parser_id: str) -> ToolCallParser:
    """Resolve a parser_id string to a ToolCallParser instance.

    Args:
        parser_id: String identifier (e.g., 'hermes_json')

    Returns:
        ToolCallParser instance

    Raises:
        KeyError: If parser_id is not registered
    """
    cls = TOOL_PARSERS.get(parser_id)
    if cls is None:
        available = list(TOOL_PARSERS.keys())
        raise KeyError(f"Unknown tool parser: {parser_id!r}. Available: {available}")
    return cls()


def resolve_thinking_parser(parser_id: str) -> ThinkingParser:
    """Resolve a parser_id string to a ThinkingParser instance.

    Args:
        parser_id: String identifier (e.g., 'think_tag')

    Returns:
        ThinkingParser instance

    Raises:
        KeyError: If parser_id is not registered
    """
    cls = THINKING_PARSERS.get(parser_id)
    if cls is None:
        raise KeyError(
            f"Unknown thinking parser: {parser_id!r}. Available: {list(THINKING_PARSERS.keys())}"
        )
    return cls()
