"""Composable adapter architecture with dependency-injected parsers."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, cast

from loguru import logger

from mlx_manager.mlx_server.parsers import (
    Glm4NativeParser,
    HermesJsonParser,
    LlamaXmlParser,
    NullThinkingParser,
    NullToolParser,
    ThinkingParser,
    ThinkTagParser,
    ToolCallParser,
)

# Common special tokens across model families
COMMON_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|end|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "assistant",  # Sometimes appears as raw text
]


class ModelAdapter(ABC):
    """Stateful adapter with injected parsers.

    Created once per loaded model, holds tokenizer reference + parser
    instances. Replaces the old Protocol-based stateless ModelAdapter.
    """

    def __init__(
        self,
        tokenizer: Any,
        tool_parser: ToolCallParser | None = None,
        thinking_parser: ThinkingParser | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        # Get actual tokenizer (Processor wraps tokenizer, regular is itself)
        self._actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self._tool_parser = tool_parser or self._default_tool_parser()
        self._thinking_parser = thinking_parser or self._default_thinking_parser()
        # Pre-compute stop tokens at init
        self._stop_tokens = self._compute_stop_tokens()

    @property
    @abstractmethod
    def family(self) -> str:
        """Model family identifier."""
        ...

    @abstractmethod
    def _default_tool_parser(self) -> ToolCallParser:
        """Default tool parser for this family."""
        ...

    @abstractmethod
    def _default_thinking_parser(self) -> ThinkingParser:
        """Default thinking parser for this family."""
        ...

    @property
    def tool_parser(self) -> ToolCallParser:
        return self._tool_parser

    @property
    def thinking_parser(self) -> ThinkingParser:
        return self._thinking_parser

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def stop_tokens(self) -> list[int]:
        """Pre-computed stop token IDs."""
        return self._stop_tokens

    def _compute_stop_tokens(self) -> list[int]:
        """Compute stop tokens from tokenizer. Override in subclasses."""
        return [self._actual_tokenizer.eos_token_id]

    def supports_native_tools(self) -> bool:
        """Whether adapter passes tools= to apply_chat_template."""
        return False

    def supports_tool_calling(self) -> bool:
        """Whether adapter supports tool calling (native or injected)."""
        return not isinstance(self._tool_parser, NullToolParser)

    def get_stream_markers(self) -> list[tuple[str, str]]:
        """Combined stream markers from tool + thinking parsers."""
        markers = list(self._tool_parser.stream_markers)
        markers.extend(self._thinking_parser.stream_markers)
        return markers

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Apply chat template. No tokenizer param needed - held internally."""
        return cast(
            str,
            self._actual_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            ),
        )

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Default: no tool formatting (override in subclasses)."""
        return ""

    def get_tool_call_stop_tokens(self) -> list[int]:
        """Additional stop tokens when tools are enabled. Override."""
        return []

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Default safe fallback: convert tool messages to user messages."""
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "unknown")
                content = msg.get("content", "")
                converted.append(
                    {
                        "role": "user",
                        "content": (f"[Tool Result for {tool_call_id}]\n{content}"),
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                content = msg.get("content", "") or ""
                tool_calls = msg.get("tool_calls", [])
                tool_text_parts = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    tool_text_parts.append(f"[Tool Call: {name}({args})]")
                tool_text = "\n".join(tool_text_parts)
                if tool_text_parts:
                    content = f"{content}\n{tool_text}" if content else tool_text
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append(msg)
        return converted

    def clean_response(self, text: str) -> str:
        """Clean response text by removing special tokens."""
        cleaned = text
        for token in COMMON_SPECIAL_TOKENS:
            cleaned = cleaned.replace(token, "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()


class DefaultAdapter(ModelAdapter):
    """Default adapter for unknown model families."""

    @property
    def family(self) -> str:
        return "default"

    def _default_tool_parser(self) -> ToolCallParser:
        return NullToolParser()

    def _default_thinking_parser(self) -> ThinkingParser:
        return NullThinkingParser()


class QwenAdapter(ModelAdapter):
    """Qwen family: Hermes JSON tool calls, <think> reasoning."""

    @property
    def family(self) -> str:
        return "qwen"

    def _default_tool_parser(self) -> ToolCallParser:
        return HermesJsonParser()

    def _default_thinking_parser(self) -> ThinkingParser:
        return ThinkTagParser()

    def _compute_stop_tokens(self) -> list[int]:
        stop_tokens = [self._actual_tokenizer.eos_token_id]
        try:
            im_end_id = self._actual_tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None and im_end_id != self._actual_tokenizer.unk_token_id:
                stop_tokens.append(im_end_id)
        except Exception:
            pass
        return stop_tokens

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Apply Qwen template with enable_thinking support."""
        try:
            result: str = cast(
                str,
                self._actual_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                    enable_thinking=True,
                ),
            )
            return result
        except (TypeError, ValueError, KeyError, AttributeError):
            return cast(
                str,
                self._actual_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                ),
            )

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
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

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages: tool→user, assistant tool_calls→Hermes tags."""
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


class GLM4Adapter(ModelAdapter):
    """GLM4 family: native tool support, <think> reasoning.

    FIX: Unlike old GLM4Adapter, this passes tools= to tokenizer natively.
    """

    @property
    def family(self) -> str:
        return "glm4"

    def _default_tool_parser(self) -> ToolCallParser:
        return Glm4NativeParser()

    def _default_thinking_parser(self) -> ThinkingParser:
        return ThinkTagParser()

    def supports_native_tools(self) -> bool:
        """GLM4.7 supports native tools."""
        return True

    def _compute_stop_tokens(self) -> list[int]:
        stop_tokens: list[int] = [self._actual_tokenizer.eos_token_id]
        special_tokens = ["<|user|>", "<|observation|>", "<|endoftext|>"]
        for token_str in special_tokens:
            try:
                token_id = self._actual_tokenizer.convert_tokens_to_ids(token_str)
                if (
                    token_id is not None
                    and token_id != self._actual_tokenizer.unk_token_id
                    and token_id not in stop_tokens
                ):
                    stop_tokens.append(token_id)
            except Exception:
                pass
        return stop_tokens

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Apply GLM4 template. Passes tools= natively (the fix)."""
        if hasattr(self._actual_tokenizer, "apply_chat_template"):
            try:
                kwargs: dict[str, Any] = {
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": False,
                }
                if tools and self.supports_native_tools():
                    kwargs["tools"] = tools
                return cast(
                    str,
                    self._actual_tokenizer.apply_chat_template(messages, **kwargs),
                )
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

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
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

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages for GLM4."""
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
                content = msg.get("content", "") + tool_text
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append(msg)
        return converted


class LlamaAdapter(ModelAdapter):
    """Llama family: XML tool calls, <think> reasoning."""

    @property
    def family(self) -> str:
        return "llama"

    def _default_tool_parser(self) -> ToolCallParser:
        return LlamaXmlParser()

    def _default_thinking_parser(self) -> ThinkingParser:
        return ThinkTagParser()

    def _compute_stop_tokens(self) -> list[int]:
        stop_tokens: list[int] = [self._actual_tokenizer.eos_token_id]
        # Critical: <|eot_id|> for Llama 3.x
        try:
            eot_id = self._actual_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != self._actual_tokenizer.unk_token_id:
                stop_tokens.append(eot_id)
        except Exception:
            pass
        # Some variants have <|end_of_turn|>
        try:
            end_turn_id = self._actual_tokenizer.convert_tokens_to_ids("<|end_of_turn|>")
            if (
                end_turn_id is not None
                and end_turn_id != self._actual_tokenizer.unk_token_id
                and end_turn_id not in stop_tokens
            ):
                stop_tokens.append(end_turn_id)
        except Exception:
            pass
        return stop_tokens

    def get_tool_call_stop_tokens(self) -> list[int]:
        stop_tokens: list[int] = []
        try:
            eom_id = self._actual_tokenizer.convert_tokens_to_ids("<|eom_id|>")
            if eom_id is not None and eom_id != self._actual_tokenizer.unk_token_id:
                stop_tokens.append(eom_id)
        except Exception:
            pass
        return stop_tokens

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
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

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages for Llama."""
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


class GemmaAdapter(ModelAdapter):
    """Gemma family: no tool support, no thinking."""

    @property
    def family(self) -> str:
        return "gemma"

    def _default_tool_parser(self) -> ToolCallParser:
        return NullToolParser()

    def _default_thinking_parser(self) -> ThinkingParser:
        return NullThinkingParser()

    def _compute_stop_tokens(self) -> list[int]:
        stop_tokens = [self._actual_tokenizer.eos_token_id]
        try:
            end_turn_id = self._actual_tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_turn_id is not None and end_turn_id != self._actual_tokenizer.unk_token_id:
                stop_tokens.append(end_turn_id)
        except Exception:
            pass
        return stop_tokens


class MistralAdapter(ModelAdapter):
    """Mistral family: no tool support, no thinking.

    Handles system message prepend for v1/v2 compatibility.
    """

    @property
    def family(self) -> str:
        return "mistral"

    def _default_tool_parser(self) -> ToolCallParser:
        return NullToolParser()

    def _default_thinking_parser(self) -> ThinkingParser:
        return NullThinkingParser()

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Apply Mistral template with system message handling for v1/v2."""
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
        return cast(
            str,
            self._actual_tokenizer.apply_chat_template(
                processed,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            ),
        )


# --- Factory ---

# Family name -> composable adapter class
FAMILY_REGISTRY: dict[str, type[ModelAdapter]] = {
    "qwen": QwenAdapter,
    "glm4": GLM4Adapter,
    "llama": LlamaAdapter,
    "gemma": GemmaAdapter,
    "mistral": MistralAdapter,
    "default": DefaultAdapter,
}


def create_adapter(
    family: str,
    tokenizer: Any,
    tool_parser: ToolCallParser | None = None,
    thinking_parser: ThinkingParser | None = None,
) -> ModelAdapter:
    """Create a composable adapter for a model family.

    Args:
        family: Model family name (e.g., "qwen", "llama")
        tokenizer: HuggingFace tokenizer or processor
        tool_parser: Override default tool parser
        thinking_parser: Override default thinking parser

    Returns:
        ModelAdapter instance
    """
    cls = FAMILY_REGISTRY.get(family, DefaultAdapter)
    return cls(
        tokenizer=tokenizer,
        tool_parser=tool_parser,
        thinking_parser=thinking_parser,
    )
