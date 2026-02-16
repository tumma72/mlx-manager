"""Data-driven family configurations for model adapters.

Each FamilyConfig describes a model family's behavior entirely through data:
parsers, stop tokens, flags, and optional strategy function references.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from mlx_manager.mlx_server.models.adapters.strategies import (
    MessageConvertStrategy,
    PostLoadHook,
    TemplateStrategy,
    ToolFormatStrategy,
    glm4_template,
    glm4_tool_formatter,
    hermes_message_converter,
    liquid_template,
    llama_message_converter,
    llama_tool_formatter,
    mistral_template,
    qwen_template,
    qwen_tool_formatter,
    whisper_post_load_hook,
)
from mlx_manager.mlx_server.parsers import (
    Glm4NativeParser,
    HermesJsonParser,
    LlamaXmlParser,
    ThinkTagParser,
)


class FamilyConfig(BaseModel):
    """Data-driven configuration for a model family.

    All behavioral differences between families are captured here as data.
    Strategy functions handle the 5 families that need custom code paths.
    """

    family: str
    tool_parser_factory: Any | None = None  # Callable[[], ToolCallParser]
    thinking_parser_factory: Any | None = None  # Callable[[], ThinkingParser]
    extra_stop_tokens: list[str] = []
    tool_call_stop_tokens: list[str] = []
    native_tools: bool = False
    template_strategy: TemplateStrategy | None = None
    tool_format_strategy: ToolFormatStrategy | None = None
    message_convert_strategy: MessageConvertStrategy | None = None
    post_load_hook: PostLoadHook | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


FAMILY_CONFIGS: dict[str, FamilyConfig] = {
    "default": FamilyConfig(family="default"),
    "qwen": FamilyConfig(
        family="qwen",
        tool_parser_factory=lambda: HermesJsonParser(),
        thinking_parser_factory=lambda: ThinkTagParser(),
        extra_stop_tokens=["<|im_end|>"],
        template_strategy=qwen_template,
        tool_format_strategy=qwen_tool_formatter,
        message_convert_strategy=hermes_message_converter,
    ),
    "glm4": FamilyConfig(
        family="glm4",
        tool_parser_factory=lambda: Glm4NativeParser(),
        thinking_parser_factory=lambda: ThinkTagParser(),
        extra_stop_tokens=["<|user|>", "<|observation|>", "<|endoftext|>"],
        native_tools=True,
        template_strategy=glm4_template,
        tool_format_strategy=glm4_tool_formatter,
        message_convert_strategy=hermes_message_converter,
    ),
    "llama": FamilyConfig(
        family="llama",
        tool_parser_factory=lambda: LlamaXmlParser(),
        thinking_parser_factory=lambda: ThinkTagParser(),
        extra_stop_tokens=["<|eot_id|>", "<|end_of_turn|>"],
        tool_call_stop_tokens=["<|eom_id|>"],
        tool_format_strategy=llama_tool_formatter,
        message_convert_strategy=llama_message_converter,
    ),
    "gemma": FamilyConfig(
        family="gemma",
        extra_stop_tokens=["<end_of_turn>"],
    ),
    "mistral": FamilyConfig(
        family="mistral",
        template_strategy=mistral_template,
    ),
    "liquid": FamilyConfig(
        family="liquid",
        tool_parser_factory=lambda: __import__(
            "mlx_manager.mlx_server.parsers.tool_call", fromlist=["LiquidPythonParser"]
        ).LiquidPythonParser(),
        thinking_parser_factory=lambda: ThinkTagParser(),
        extra_stop_tokens=["<|im_end|>"],
        native_tools=True,
        template_strategy=liquid_template,
    ),
    "whisper": FamilyConfig(
        family="whisper",
        post_load_hook=whisper_post_load_hook,
    ),
    "kokoro": FamilyConfig(
        family="kokoro",
    ),
    "audio_default": FamilyConfig(
        family="audio_default",
    ),
    "embeddings": FamilyConfig(
        family="embeddings",
    ),
}
