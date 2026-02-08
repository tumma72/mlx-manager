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
                tool_format = await _verify_tool_support(loaded)
                result.supports_native_tools = tool_format is not None
                result.tool_format = tool_format
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


async def _verify_tool_support(loaded: LoadedModel) -> str | None:
    """3-tier generation-based tool verification.

    Tier 1 — Native tools: Template supports tools= parameter →
             generate with tool definition → check output for valid tool call
             → return "native"

    Tier 2 — Hermes format: No native tools= → inject Hermes-style system prompt
             → generate → check for <tool_call> with correct function
             → return "hermes"

    Tier 3 — No tool support: Neither produces valid tool calls → return None

    Returns:
        "native", "hermes", or None
    """
    from mlx_manager.mlx_server.utils.template_tools import has_native_tool_support

    tokenizer = loaded.tokenizer

    # Tier 1: Try native tool support
    if has_native_tool_support(tokenizer):
        try:
            output = await _generate_with_native_tools(loaded)
            if _contains_valid_tool_call(output, "get_weather"):
                logger.info("Tool probe: native tool support verified via generation")
                return "native"
            logger.debug(
                f"Native tools template accepted but output has no tool call: {output[:200]}"
            )
        except Exception as e:
            logger.debug(f"Native tool generation failed: {e}")

    # Tier 2: Try Hermes-style tool injection
    try:
        output = await _generate_with_hermes_prompt(loaded)
        if _contains_valid_tool_call(output, "get_weather"):
            logger.info("Tool probe: Hermes tool format verified via generation")
            return "hermes"
        logger.debug(f"Hermes prompt did not produce tool call: {output[:200]}")
    except Exception as e:
        logger.debug(f"Hermes tool generation failed: {e}")

    # Tier 3: No tool support detected
    logger.info("Tool probe: no tool support detected")
    return None


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


def _contains_valid_tool_call(output: str, expected_function: str) -> bool:
    """Check if generation output contains a valid tool call for the expected function.

    Checks for:
    1. Native JSON tool calls (e.g. {"name": "get_weather", ...})
    2. Hermes-style <tool_call>{"name": "get_weather", ...}</tool_call>
    3. Function call patterns with the expected function name
    """
    import re

    if not output or expected_function not in output:
        return False

    # Check for Hermes-style <tool_call> tags
    hermes_pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)
    for match in hermes_pattern.finditer(output):
        try:
            data = json.loads(match.group(1))
            if data.get("name") == expected_function:
                return True
        except (json.JSONDecodeError, AttributeError):
            continue

    # Check for raw JSON tool call objects in output
    json_pattern = re.compile(
        r'\{\s*"name"\s*:\s*"' + re.escape(expected_function) + r'".*?\}', re.DOTALL
    )
    for match in json_pattern.finditer(output):
        try:
            json.loads(match.group(0))
            return True
        except json.JSONDecodeError:
            continue

    return False


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
