"""Standalone, reusable extraction strategies for model output."""

from mlx_manager.mlx_server.parsers.base import ThinkingParser, ToolCallParser
from mlx_manager.mlx_server.parsers.registry import (
    THINKING_PARSERS,
    TOOL_PARSERS,
    resolve_thinking_parser,
    resolve_tool_parser,
)
from mlx_manager.mlx_server.parsers.thinking import NullThinkingParser, ThinkTagParser
from mlx_manager.mlx_server.parsers.tool_call import (
    Glm4NativeParser,
    Glm4XmlParser,
    HermesJsonParser,
    LlamaPythonParser,
    LlamaXmlParser,
    NullToolParser,
)

__all__ = [
    "ToolCallParser",
    "ThinkingParser",
    "HermesJsonParser",
    "Glm4NativeParser",
    "Glm4XmlParser",
    "LlamaXmlParser",
    "LlamaPythonParser",
    "NullToolParser",
    "ThinkTagParser",
    "NullThinkingParser",
    "TOOL_PARSERS",
    "THINKING_PARSERS",
    "resolve_tool_parser",
    "resolve_thinking_parser",
]
