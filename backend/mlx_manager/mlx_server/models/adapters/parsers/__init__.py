"""Tool call parsers for model-specific output formats.

Each model family has its own unique format for tool calls:
- Llama: <function=name>{"param": "value"}</function>
- Qwen: <tool_call>{"name": "func", "arguments": {...}}</tool_call> (Hermes style)
- GLM4: <tool_call><name>func</name><arguments>{...}</arguments></tool_call>
"""

from mlx_manager.mlx_server.models.adapters.parsers.base import ToolCallParser
from mlx_manager.mlx_server.models.adapters.parsers.llama import LlamaToolParser

__all__ = [
    "ToolCallParser",
    "LlamaToolParser",
]
