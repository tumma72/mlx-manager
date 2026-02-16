"""Behavioral strategy functions for model adapters.

Each function is a standalone strategy extracted from the old adapter subclass
methods. Strategies are referenced by FamilyConfig and called by ModelAdapter.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

if TYPE_CHECKING:
    pass

# ── Type aliases for strategy functions ──────────────────────────────

TemplateStrategy = Callable[..., str]
ToolFormatStrategy = Callable[[list[dict[str, Any]]], str]
MessageConvertStrategy = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
PostLoadHook = Callable[[Any, str], Awaitable[None]]


# ── Template strategies ──────────────────────────────────────────────


def qwen_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    native_tools: list[dict[str, Any]] | None,
    template_options: dict[str, Any] | None,
) -> str:
    """Apply Qwen template with enable_thinking support."""
    opts = template_options or {}
    enable_thinking = opts.get("enable_thinking", True)
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": False,
        "enable_thinking": enable_thinking,
    }
    if native_tools:
        kwargs["tools"] = native_tools
    try:
        return cast(str, tokenizer.apply_chat_template(messages, **kwargs))
    except (TypeError, ValueError, KeyError, AttributeError):
        del kwargs["enable_thinking"]
        return cast(str, tokenizer.apply_chat_template(messages, **kwargs))


def glm4_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    native_tools: list[dict[str, Any]] | None,
    template_options: dict[str, Any] | None,
) -> str:
    """Apply GLM4 template with native tool support and ChatML fallback."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            kwargs: dict[str, Any] = {
                "add_generation_prompt": add_generation_prompt,
                "tokenize": False,
            }
            if native_tools:
                kwargs["tools"] = native_tools
            return cast(str, tokenizer.apply_chat_template(messages, **kwargs))
        except Exception as e:
            logger.warning("GLM4 tokenizer.apply_chat_template failed: {}", e)
    # Manual ChatML fallback
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    if add_generation_prompt:
        parts.append("<|assistant|>")
    return "\n".join(parts)


def mistral_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    native_tools: list[dict[str, Any]] | None,
    template_options: dict[str, Any] | None,
) -> str:
    """Apply Mistral template with system message handling for v1/v2."""
    # Mistral v1/v2: merge system message into first user message
    processed = list(messages)
    if processed and processed[0].get("role") == "system":
        system_content = processed[0].get("content", "")
        processed = processed[1:]
        if processed and processed[0].get("role") == "user":
            user_content = processed[0].get("content", "")
            processed[0] = {
                "role": "user",
                "content": f"{system_content}\n\n{user_content}",
            }
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": False,
    }
    if native_tools:
        kwargs["tools"] = native_tools
    return cast(str, tokenizer.apply_chat_template(processed, **kwargs))


def liquid_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    native_tools: list[dict[str, Any]] | None,
    template_options: dict[str, Any] | None,
) -> str:
    """Apply Liquid template with keep_past_thinking and native tool support."""
    opts = template_options or {}
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": False,
    }
    if native_tools:
        kwargs["tools"] = native_tools
    # Pass known template options as kwargs to tokenizer
    if "keep_past_thinking" in opts:
        kwargs["keep_past_thinking"] = opts["keep_past_thinking"]
    try:
        return cast(str, tokenizer.apply_chat_template(messages, **kwargs))
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        # Fallback: remove extra kwargs
        logger.debug("Liquid template with options failed: {}, retrying without", e)
        fallback_kwargs: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": False,
        }
        if native_tools:
            fallback_kwargs["tools"] = native_tools
        return cast(str, tokenizer.apply_chat_template(messages, **fallback_kwargs))


# ── Tool format strategies ───────────────────────────────────────────


def qwen_tool_formatter(tools: list[dict[str, Any]]) -> str:
    """Format tools as XML for Qwen-style prompt injection."""
    if not tools:
        return ""
    tool_docs: list[str] = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        doc = (
            f'{{\n  "name": "{name}",\n'
            f'  "description": "{description}",\n'
            f'  "parameters": {json.dumps(parameters)}\n}}'
        )
        tool_docs.append(doc)
    nl = "\n"
    return (
        f"<tools>\n{nl.join(tool_docs)}\n</tools>\n\n"
        "When you need to call a tool, respond with:\n"
        '<tool_call>{"name": "function_name", '
        '"arguments": {"param": "value"}}</tool_call>\n\n'
        "Only call tools when necessary. "
        "If no tool call is needed, respond normally."
    )


def glm4_tool_formatter(tools: list[dict[str, Any]]) -> str:
    """Format tools as XML for GLM4-style prompt injection."""
    if not tools:
        return ""
    tool_docs: list[str] = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        doc = (
            f"<tool>\n<name>{name}</name>\n"
            f"<description>{description}</description>\n"
            f"<parameters>{json.dumps(parameters)}</parameters>\n"
            "</tool>"
        )
        tool_docs.append(doc)
    nl = "\n"
    return (
        "You have access to the following tools:\n\n"
        f"{nl.join(tool_docs)}\n\n"
        "When you need to call a tool, use this format:\n"
        "<tool_call>\n"
        "<name>tool_name</name>\n"
        '<arguments>{"param": "value"}</arguments>\n'
        "</tool_call>\n\n"
        "Only call tools when necessary."
    )


def llama_tool_formatter(tools: list[dict[str, Any]]) -> str:
    """Format tools as YAML for Llama-style prompt injection."""
    if not tools:
        return ""
    tool_docs: list[str] = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        doc = (
            f"{name}:\n"
            f"  description: {description}\n"
            f"  parameters: {json.dumps(parameters, indent=2)}"
        )
        tool_docs.append(doc)
    nl = "\n"
    return (
        "You have access to the following functions:\n\n"
        f"{nl.join(tool_docs)}\n\n"
        "To call a function, respond with:\n"
        '<function=function_name>{"param": "value"}</function>\n\n'
        "Only call functions when necessary. "
        "If no function call is needed, respond normally."
    )


# ── Message conversion strategies ────────────────────────────────────


def hermes_message_converter(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages: tool→user, assistant tool_calls→Hermes tags.

    Used by Qwen and GLM4 families.
    """
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            content = msg.get("content", "")
            converted.append(
                {
                    "role": "user",
                    "content": (
                        f"[Tool Result for {tool_call_id}]\n"
                        f"{content}\n[End Tool Result]\n\n"
                        "Please provide your response based on "
                        "this tool result."
                    ),
                }
            )
        elif role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls", [])
            tool_text = ""
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_call_data = {
                    "name": func.get("name"),
                    "arguments": json.loads(func.get("arguments", "{}")),
                }
                tool_text += f"\n<tool_call>{json.dumps(tool_call_data)}</tool_call>"
            content = (msg.get("content", "") or "") + tool_text
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append(msg)
    return converted


def llama_message_converter(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages for Llama: tool→user, assistant→function tags."""
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            content = msg.get("content", "")
            converted.append(
                {
                    "role": "user",
                    "content": (
                        f"[Tool Result for {tool_call_id}]\n"
                        f"{content}\n[End Tool Result]\n\n"
                        "Please provide your response based on "
                        "this tool result."
                    ),
                }
            )
        elif role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls", [])
            tool_text = ""
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                tool_text += f"\n<function={name}>{args}</function>"
            content = (msg.get("content", "") or "") + tool_text
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append(msg)
    return converted


# ── Post-load hooks ──────────────────────────────────────────────────


async def whisper_post_load_hook(model: Any, model_id: str) -> None:
    """Fix missing/broken WhisperProcessor on mlx-community models."""
    proc = getattr(model, "_processor", None)
    tok = getattr(proc, "tokenizer", None) if proc else None
    if proc is not None and tok is not None and getattr(tok, "vocab_size", 0) > 0:
        return  # Processor is fine

    # Derive the canonical repo: mlx-community/whisper-X → openai/whisper-X
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    canonical_repo = f"openai/{name}"

    try:
        from transformers import WhisperProcessor

        processor = await asyncio.to_thread(WhisperProcessor.from_pretrained, canonical_repo)
        model._processor = processor
        logger.info(f"Loaded WhisperProcessor from fallback repo: {canonical_repo}")
    except Exception as e:
        logger.warning(f"Could not load WhisperProcessor from {canonical_repo}: {e}")
