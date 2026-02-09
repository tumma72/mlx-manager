"""Text generation model probe strategy.

Tests thinking support, native tool support, and estimates
practical context window based on KV cache memory requirements.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel


class TextGenProbe:
    """Probe strategy for text generation models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.TEXT_GEN

    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        result.model_type = ModelType.TEXT_GEN

        # Detect model family
        from mlx_manager.mlx_server.models.adapters.registry import (
            detect_model_family,
        )

        result.model_family = detect_model_family(model_id)

        # Step 1: Estimate practical context window
        yield ProbeStep(step="check_context", status="running")
        try:
            practical_max = _estimate_practical_max_tokens(model_id, loaded)
            result.practical_max_tokens = practical_max
            yield ProbeStep(
                step="check_context",
                status="completed",
                capability="practical_max_tokens",
                value=practical_max,
            )
        except Exception as e:
            logger.warning(f"Context check failed for {model_id}: {e}")
            yield ProbeStep(step="check_context", status="failed", error=str(e))

        tokenizer = loaded.tokenizer

        # Step 2: Test thinking support
        if tokenizer is not None:
            yield ProbeStep(step="test_thinking", status="running")
            try:
                from mlx_manager.mlx_server.utils.template_tools import (
                    has_thinking_support,
                )

                supports_thinking = has_thinking_support(tokenizer)
                result.supports_thinking = supports_thinking
                result.thinking_parser_id = "think_tag" if supports_thinking else "null"
                yield ProbeStep(
                    step="test_thinking",
                    status="completed",
                    capability="supports_thinking",
                    value=supports_thinking,
                )
            except Exception as e:
                logger.warning(f"Thinking test failed for {model_id}: {e}")
                yield ProbeStep(step="test_thinking", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_thinking", status="skipped")

        # Step 3: Generation-based tool verification (3-tier)
        if tokenizer is not None:
            yield ProbeStep(step="test_tools", status="running")
            try:
                tool_format, tool_parser_id = await _verify_tool_support(loaded)
                result.supports_native_tools = tool_format is not None
                result.tool_format = tool_format
                result.tool_parser_id = tool_parser_id
                yield ProbeStep(
                    step="test_tools",
                    status="completed",
                    capability="tool_format",
                    value=tool_format,
                )
            except Exception as e:
                logger.warning(f"Tool verification failed for {model_id}: {e}")
                yield ProbeStep(step="test_tools", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_tools", status="skipped")


# Tool definitions used for generation-based tool probing
_TOOL_PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. Tokyo",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

_TOOL_PROBE_MESSAGES = [
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

_HERMES_SYSTEM_PROMPT = (
    "You are a function calling AI model. You are provided with function signatures "
    "within <tools></tools> XML tags. You may call one or more functions to assist with "
    "the user query. Don't make assumptions about what values to plug into functions.\n\n"
    "<tools>\n"
    '[{"type": "function", "function": {"name": "get_weather", '
    '"description": "Get the current weather for a location", '
    '"parameters": {"type": "object", "properties": {"location": '
    '{"type": "string", "description": "City name"}}, '
    '"required": ["location"]}}}]\n'
    "</tools>\n\n"
    "For each function call return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags as follows:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-dict>}\n'
    "</tool_call>"
)


async def _verify_tool_support(
    loaded: LoadedModel,
) -> tuple[str | None, str | None]:
    """3-tier generation-based tool verification with parser registry.

    Tier 1 — Native tools: Template supports tools= parameter →
             generate with tool definition → validate with ALL parsers
             → return ("native", parser_id)

    Tier 2 — Hermes format: No native tools= → inject Hermes-style prompt
             → generate → validate with ALL parsers
             → return ("hermes", parser_id)

    Tier 3 — No tool support: Neither produces valid tool calls
             → return (None, None)
    """
    from mlx_manager.mlx_server.utils.template_tools import (
        has_native_tool_support,
    )

    tokenizer = loaded.tokenizer

    # Tier 1: Try native tool support
    if has_native_tool_support(tokenizer):
        try:
            output = await _generate_with_native_tools(loaded)
            parser_id = _find_matching_parser(output, "get_weather")
            if parser_id:
                logger.info(
                    "Tool probe: native tool support verified (parser=%s)",
                    parser_id,
                )
                return ("native", parser_id)
            logger.debug(
                "Native tools template accepted but no parser matched: %s",
                output[:200],
            )
        except Exception as e:
            logger.debug("Native tool generation failed: %s", e)

    # Tier 2: Try Hermes-style tool injection
    try:
        output = await _generate_with_hermes_prompt(loaded)
        parser_id = _find_matching_parser(output, "get_weather")
        if parser_id:
            logger.info(
                "Tool probe: Hermes format verified (parser=%s)",
                parser_id,
            )
            return ("hermes", parser_id)
        logger.debug(
            "Hermes prompt did not match any parser: %s",
            output[:200],
        )
    except Exception as e:
        logger.debug("Hermes tool generation failed: %s", e)

    # Tier 3: No tool support
    logger.info("Tool probe: no tool support detected")
    return (None, None)


async def _generate_with_native_tools(loaded: LoadedModel) -> str:
    """Generate a response using the tokenizer's native tools= parameter."""
    import asyncio

    from mlx_lm import generate as mlx_generate

    tokenizer = loaded.tokenizer
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    prompt = actual_tokenizer.apply_chat_template(
        _TOOL_PROBE_MESSAGES,
        tools=_TOOL_PROBE_TOOL,
        add_generation_prompt=True,
        tokenize=False,
    )

    output = await asyncio.to_thread(
        mlx_generate,
        loaded.model,
        tokenizer,
        prompt=prompt,
        max_tokens=200,
        verbose=False,
    )
    return output


async def _generate_with_hermes_prompt(loaded: LoadedModel) -> str:
    """Generate a response using Hermes-style system prompt for tool injection."""
    import asyncio

    from mlx_lm import generate as mlx_generate

    tokenizer = loaded.tokenizer
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    messages = [
        {"role": "system", "content": _HERMES_SYSTEM_PROMPT},
        {"role": "user", "content": "What is the weather in Tokyo?"},
    ]

    prompt = actual_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    output = await asyncio.to_thread(
        mlx_generate,
        loaded.model,
        tokenizer,
        prompt=prompt,
        max_tokens=200,
        verbose=False,
    )
    return output


def _find_matching_parser(output: str, expected_function: str) -> str | None:
    """Try ALL registered parsers to find one that validates the output.

    Returns parser_id of the first matching parser, or None.
    """
    from mlx_manager.mlx_server.parsers import TOOL_PARSERS

    for parser_id, parser_cls in TOOL_PARSERS.items():
        if parser_id == "null":
            continue
        parser = parser_cls()
        if parser.validates(output, expected_function):
            return parser_id
    return None


def _estimate_practical_max_tokens(model_id: str, loaded: Any) -> int | None:
    """Estimate practical max tokens based on model config and available memory.

    Uses the formula:
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  (bytes, fp16)
    available_bytes = (device_memory * 0.75 - model_size_gb - 1.0) * 1e9
    practical_max = min(max_position_embeddings, available_bytes / kv_per_token)
    """
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        max_pos = config.get(
            "max_position_embeddings",
            config.get("max_sequence_length", config.get("seq_length")),
        )
        if max_pos is None:
            return None

        num_layers = config.get(
            "num_hidden_layers", config.get("n_layer", config.get("num_layers"))
        )
        num_kv_heads = config.get(
            "num_key_value_heads",
            config.get("num_attention_heads", config.get("n_head")),
        )
        head_dim = config.get("head_dim")
        if head_dim is None:
            hidden_size = config.get("hidden_size", config.get("d_model"))
            num_heads = config.get("num_attention_heads", config.get("n_head"))
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads

        if not all([num_layers, num_kv_heads, head_dim]):
            return int(max_pos)

        # KV cache cost per token in bytes (fp16)
        kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2

        # Available memory
        from mlx_manager.mlx_server.utils.memory import get_device_memory_gb

        device_memory_gb = get_device_memory_gb()
        model_size_gb = loaded.size_gb

        available_gb = (device_memory_gb * 0.75) - model_size_gb - 1.0
        if available_gb <= 0:
            return int(min(max_pos, 2048))  # Minimum practical context

        available_bytes = available_gb * 1e9
        practical_max = int(min(max_pos, available_bytes / kv_per_token))

        return max(practical_max, 512)  # At least 512 tokens

    except Exception as e:
        logger.debug(f"Could not estimate practical max tokens for {model_id}: {e}")
        return None
