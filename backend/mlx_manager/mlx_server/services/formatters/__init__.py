"""Protocol formatters for the adapter pipeline.

Layer 3 of the 3-layer adapter pipeline: ModelAdapter → StreamProcessor → ProtocolFormatter.

Formatters convert protocol-neutral IR types (StreamEvent, TextResult) into
protocol-specific response formats (OpenAI, Anthropic).
"""

from mlx_manager.mlx_server.services.formatters.anthropic import (
    AnthropicFormatter,
    InternalRequest,
    anthropic_stop_to_openai,
    openai_stop_to_anthropic,
)
from mlx_manager.mlx_server.services.formatters.base import ProtocolFormatter
from mlx_manager.mlx_server.services.formatters.openai import OpenAIFormatter

__all__ = [
    "AnthropicFormatter",
    "InternalRequest",
    "OpenAIFormatter",
    "ProtocolFormatter",
    "anthropic_stop_to_openai",
    "openai_stop_to_anthropic",
]
