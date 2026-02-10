"""Text generation model probe strategy.

Tests thinking support, native tool support, and estimates
practical context window based on KV cache memory requirements.

Tool verification uses a 2-attempt, adapter-driven approach:
1. Template delivery: adapter passes tools= to tokenizer natively
2. Adapter delivery: adapter injects tool prompt into messages

Thinking verification uses generation-based validation with
adapter's thinking_parser, then sweeps all registered parsers.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter
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
            logger.warning("Context check failed for {}: {}", model_id, e)
            yield ProbeStep(step="check_context", status="failed", error=str(e))

        tokenizer = loaded.tokenizer
        adapter = loaded.adapter

        # Guard: adapter must exist for generation-based probing
        if adapter is None:
            logger.warning(
                "No adapter available for {}; skipping thinking and tool probes",
                model_id,
            )
            yield ProbeStep(step="test_thinking", status="skipped")
            yield ProbeStep(step="test_tools", status="skipped")
            return

        # Step 2: Test thinking support (generation-based)
        if tokenizer is not None:
            yield ProbeStep(step="test_thinking", status="running")
            try:
                supports_thinking, thinking_parser_id = await _verify_thinking_support(
                    loaded, adapter
                )
                result.supports_thinking = supports_thinking
                result.thinking_parser_id = thinking_parser_id
                yield ProbeStep(
                    step="test_thinking",
                    status="completed",
                    capability="supports_thinking",
                    value=supports_thinking,
                )
            except Exception as e:
                logger.warning("Thinking test failed for {}: {}", model_id, e)
                yield ProbeStep(step="test_thinking", status="failed", error=str(e))
        else:
            yield ProbeStep(step="test_thinking", status="skipped")

        # Step 3: Generation-based tool verification (2-attempt)
        if tokenizer is not None:
            yield ProbeStep(step="test_tools", status="running")
            try:
                tool_format, tool_parser_id = await _verify_tool_support(loaded, adapter)
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
                logger.warning("Tool verification failed for {}: {}", model_id, e)
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

# Known XML tags that are NOT indicators of unknown tool calling formats
_KNOWN_XML_TAGS: frozenset[str] = frozenset(
    {
        "think",
        "thinking",
        "reasoning",
        "reflection",
        "tool_call",
        "tool_response",
        "function_call",
        "output",
        "result",
        "code",
        "step",
        "answer",
        "solution",
    }
)


async def _generate_via_adapter(
    loaded: LoadedModel,
    messages: list[dict],
    tools: list[dict] | None = None,
    enable_thinking: bool = False,
) -> str:
    """Generate a response using the model's adapter for template handling."""
    import asyncio

    from mlx_lm import generate as mlx_generate

    adapter = loaded.adapter
    if adapter is None:
        msg = "No adapter available for generation"
        raise RuntimeError(msg)

    prompt = adapter.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tools=tools,
        enable_thinking=enable_thinking,
    )
    output = await asyncio.to_thread(
        mlx_generate,
        loaded.model,
        loaded.tokenizer,
        prompt=prompt,
        max_tokens=200,
        verbose=False,
    )
    return output


async def _verify_thinking_support(loaded: LoadedModel, adapter: ModelAdapter) -> tuple[bool, str]:
    """Verify thinking support via generation and parser validation.

    Returns:
        (supports_thinking, thinking_parser_id)
    """
    from mlx_manager.mlx_server.parsers import THINKING_PARSERS
    from mlx_manager.mlx_server.utils.template_tools import has_thinking_support

    tokenizer = loaded.tokenizer
    template_supports = has_thinking_support(tokenizer)

    if not template_supports:
        return (False, "null")

    # Template supports thinking -- generate to verify output tags
    try:
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        output = await _generate_via_adapter(loaded, messages, enable_thinking=True)

        # Try adapter's thinking_parser first
        thinking_parser = adapter.thinking_parser
        if thinking_parser.parser_id != "null":
            thinking_content = thinking_parser.extract(output)
            if thinking_content is not None:
                logger.info(
                    "Thinking probe: verified via adapter parser (parser={})",
                    thinking_parser.parser_id,
                )
                return (True, thinking_parser.parser_id)

        # Sweep all registered thinking parsers
        for parser_id, parser_cls in THINKING_PARSERS.items():
            if parser_id == "null":
                continue
            if parser_id == thinking_parser.parser_id:
                continue  # already tried
            parser = parser_cls()
            thinking_content = parser.extract(output)
            if thinking_content is not None:
                logger.info(
                    "Thinking probe: verified via sweep parser (parser={})",
                    parser_id,
                )
                return (True, parser_id)

        # Template supports it but no tags found in output -- still report
        # supported since template check is authoritative for toggle capability
        logger.info(
            "Thinking probe: template supports thinking but no tags in output; "
            "reporting supported (template authoritative)"
        )
        fallback_id = (
            thinking_parser.parser_id if thinking_parser.parser_id != "null" else "think_tag"
        )
        return (True, fallback_id)

    except Exception as e:
        logger.debug("Thinking generation failed, falling back to template check: {}", e)
        # Template says it supports thinking; trust that even if generation failed
        return (True, "think_tag")


async def _verify_tool_support(
    loaded: LoadedModel, adapter: ModelAdapter
) -> tuple[str | None, str | None]:
    """2-attempt adapter-driven tool verification.

    Attempt 1 -- Template delivery: If adapter supports native tools or
    tokenizer has native tool support, generate with tools= param.

    Attempt 2 -- Adapter delivery: If adapter can format tools for prompt
    injection, inject as system message and generate.

    Returns:
        (tool_format, tool_parser_id) or (None, None)
    """
    from mlx_manager.mlx_server.utils.template_tools import has_native_tool_support

    tokenizer = loaded.tokenizer
    last_output: str | None = None

    # Attempt 1: Template delivery
    if adapter.supports_native_tools() or has_native_tool_support(tokenizer):
        try:
            last_output = await _generate_via_adapter(
                loaded, _TOOL_PROBE_MESSAGES, tools=_TOOL_PROBE_TOOL
            )
            parser_id = _validate_tool_output(last_output, "get_weather", adapter)
            if parser_id:
                logger.info(
                    "Tool probe: template delivery verified (parser={})",
                    parser_id,
                )
                return ("template", parser_id)
            logger.debug(
                "Template delivery produced output but no parser matched: {}",
                last_output[:200],
            )
        except Exception as e:
            logger.debug("Template delivery generation failed: {}", e)

    # Attempt 2: Adapter delivery
    tool_prompt = adapter.format_tools_for_prompt(_TOOL_PROBE_TOOL)
    if tool_prompt:
        try:
            messages_with_tools = [
                {"role": "system", "content": tool_prompt},
                *_TOOL_PROBE_MESSAGES,
            ]
            last_output = await _generate_via_adapter(loaded, messages_with_tools)
            parser_id = _validate_tool_output(last_output, "get_weather", adapter)
            if parser_id:
                logger.info(
                    "Tool probe: adapter delivery verified (parser={})",
                    parser_id,
                )
                return ("adapter", parser_id)
            logger.debug(
                "Adapter delivery produced output but no parser matched: {}",
                last_output[:200],
            )
        except Exception as e:
            logger.debug("Adapter delivery generation failed: {}", e)

    # No match -- scan for unknown XML tags as diagnostic
    try:
        if last_output is None:
            last_output = await _generate_via_adapter(loaded, _TOOL_PROBE_MESSAGES)
        unknown_tags = _detect_unknown_xml_tags(last_output)
        if unknown_tags:
            logger.warning(
                "Tool probe: no parser matched but found unknown XML tags: {}",
                unknown_tags,
            )
        else:
            logger.info("Tool probe: no tool support detected")
    except Exception:
        logger.info("Tool probe: no tool support detected")

    return (None, None)


def _validate_tool_output(output: str, expected_fn: str, adapter: ModelAdapter) -> str | None:
    """Validate tool output using adapter's parser first, then sweep all parsers.

    Returns parser_id of the matching parser, or None.
    """
    # Try adapter's own tool parser first
    adapter_parser = adapter.tool_parser
    adapter_parser_id: str | None = None
    if adapter_parser.parser_id != "null":
        adapter_parser_id = adapter_parser.parser_id
        if adapter_parser.validates(output, expected_fn):
            return adapter_parser_id

    # Sweep remaining parsers, excluding the one we already tried
    sweep_result = _find_matching_parser(output, expected_fn, exclude_parser_id=adapter_parser_id)
    return sweep_result


def _find_matching_parser(
    output: str,
    expected_function: str,
    exclude_parser_id: str | None = None,
) -> str | None:
    """Try ALL registered parsers to find one that validates the output.

    Args:
        output: Model generation output to validate.
        expected_function: Expected function name in tool call.
        exclude_parser_id: Parser ID to skip (already checked).

    Returns:
        parser_id of the first matching parser, or None.
    """
    from mlx_manager.mlx_server.parsers import TOOL_PARSERS

    for parser_id, parser_cls in TOOL_PARSERS.items():
        if parser_id == "null":
            continue
        if parser_id == exclude_parser_id:
            continue
        parser = parser_cls()
        if parser.validates(output, expected_function):
            return parser_id
    return None


def _detect_unknown_xml_tags(output: str) -> set[str]:
    """Scan output for XML-style tags not in the known set.

    Returns set of unknown tag names found.
    """
    # Match opening tags like <tag_name> (not self-closing, not closing)
    tag_pattern = re.compile(r"<([a-zA-Z_][\w-]*)(?:\s[^>]*)?>")
    found_tags: set[str] = set()
    for match in tag_pattern.finditer(output):
        tag_name = match.group(1).lower()
        # Verify there's a corresponding closing tag
        close_pattern = re.compile(rf"</{re.escape(match.group(1))}>", re.IGNORECASE)
        if close_pattern.search(output):
            found_tags.add(tag_name)

    unknown = found_tags - _KNOWN_XML_TAGS
    return unknown


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
        logger.debug("Could not estimate practical max tokens for {}: {}", model_id, e)
        return None
