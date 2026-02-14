"""Composable adapter architecture with dependency-injected parsers."""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.ir import PreparedInput, TextResult
    from mlx_manager.mlx_server.services.response_processor import StreamProcessor

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
        tokenizer: Any | None = None,
        tool_parser: ToolCallParser | None = None,
        thinking_parser: ThinkingParser | None = None,
        model_id: str | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        # Get actual tokenizer (Processor wraps tokenizer, regular is itself)
        # Audio adapters pass None; text/vision always have a tokenizer.
        self._actual_tokenizer: Any = (
            getattr(tokenizer, "tokenizer", tokenizer) if tokenizer is not None else None
        )
        self._tool_parser = tool_parser or self._default_tool_parser()
        self._thinking_parser = thinking_parser or self._default_thinking_parser()
        # Pre-compute stop tokens at init
        self._stop_tokens = self._compute_stop_tokens()
        self._model_id = model_id

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
        if self._actual_tokenizer is None:
            return []
        return [self._actual_tokenizer.eos_token_id]

    async def post_load_configure(self, model: Any, model_id: str) -> None:
        """Post-load configuration hook. Override for model-specific fixups."""

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

    # ── Tool delivery ────────────────────────────────────────────

    @staticmethod
    def _inject_tools(messages: list[dict[str, Any]], tool_prompt: str) -> list[dict[str, Any]]:
        """Inject tool definitions into the system message."""
        result = list(messages)
        system_idx = None
        for i, msg in enumerate(result):
            if msg.get("role") == "system":
                system_idx = i
                break
        if system_idx is not None:
            existing = result[system_idx].get("content", "")
            result[system_idx] = {
                **result[system_idx],
                "content": f"{existing}\n\n{tool_prompt}",
            }
        else:
            result.insert(0, {"role": "system", "content": tool_prompt})
        return result

    def _prepare_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Prepare messages for tool-aware template application.

        Returns:
            (effective_messages, native_tools) where native_tools is
            passed to tokenizer.apply_chat_template(tools=...) for
            models with native template support, or None when tools
            were injected into messages (or absent).
        """
        if not tools:
            return messages, None
        if self.supports_native_tools():
            return messages, tools
        # Adapter delivery: inject formatted tools into system message
        tool_prompt = self.format_tools_for_prompt(tools)
        if tool_prompt:
            return self._inject_tools(messages, tool_prompt), None
        return messages, None

    # ── Template application ─────────────────────────────────────

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Apply chat template with automatic tool delivery."""
        effective, native_tools = self._prepare_tools(messages, tools)
        kwargs: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": False,
        }
        if native_tools:
            kwargs["tools"] = native_tools
        return cast(
            str,
            self._actual_tokenizer.apply_chat_template(effective, **kwargs),
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

    def create_stream_processor(self, prompt: str = "") -> StreamProcessor:
        """Create a StreamProcessor for this adapter.

        Factory method that creates a properly configured processor using
        the adapter's parsers. Detects whether the prompt ends with a
        thinking start tag (from this adapter's thinking parser markers)
        to initialize the processor in thinking mode.

        Args:
            prompt: The prompt string being sent to the model. Used to
                    detect if the model should start in thinking mode.

        Returns:
            StreamProcessor configured with this adapter's parsers
        """
        # Detect if prompt ends with a thinking start tag
        starts_in_thinking = False
        if prompt:
            stripped = prompt.rstrip()
            thinking_starts = [start for start, _ in self._thinking_parser.stream_markers]
            starts_in_thinking = any(stripped.endswith(tag) for tag in thinking_starts)

        from mlx_manager.mlx_server.services.response_processor import StreamProcessor

        return StreamProcessor(adapter=self, starts_in_thinking=starts_in_thinking)

    def prepare_input(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_prompt_injection: bool = False,
        images: list[Any] | None = None,
    ) -> PreparedInput:
        """Prepare model-ready input from messages and optional tools.

        Encapsulates the full input pipeline:
        1. convert_messages() — format conversion
        2. Tool capability check
        3. apply_chat_template() — prompt generation
        4. Stop token aggregation

        When images are provided (vision models), uses mlx-vlm chat template
        instead of the regular tokenizer template.
        """
        from mlx_manager.mlx_server.models.ir import PreparedInput

        # Vision path: use mlx-vlm chat template
        if images:
            from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
            from mlx_vlm.utils import load_config

            # Build flat text prompt from messages
            text_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Multipart content - extract text parts only
                    text = " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                else:
                    text = str(content) if content else ""
                if text:
                    if role == "system":
                        text_parts.append(f"System: {text}")
                    elif role == "user":
                        text_parts.append(text)
                    elif role == "assistant":
                        text_parts.append(f"Assistant: {text}")
            text_prompt = "\n".join(text_parts)

            # Load model config and apply vision chat template
            config = load_config(self._model_id)
            prompt = vlm_apply_chat_template(
                self._tokenizer, config, text_prompt, num_images=len(images)
            )

            stop_ids: set[int] = set(self.stop_tokens)

            return PreparedInput(
                prompt=prompt,
                stop_token_ids=list(stop_ids),
                pixel_values=images,
            )

        # Text path: regular chat template
        converted = self.convert_messages(messages)

        use_tools = tools and (self.supports_tool_calling() or enable_prompt_injection)
        effective_tools = tools if use_tools else None

        prompt = self.apply_chat_template(
            messages=converted,
            add_generation_prompt=True,
            tools=effective_tools,
        )

        stop_ids = set(self.stop_tokens)
        if use_tools:
            stop_ids.update(self.get_tool_call_stop_tokens())

        return PreparedInput(
            prompt=prompt,
            stop_token_ids=list(stop_ids),
        )

    def process_complete(self, raw_text: str, finish_reason: str = "stop") -> TextResult:
        """Post-process raw model output into a TextResult.

        Encapsulates the full output pipeline:
        1. Tool call extraction
        2. Thinking/reasoning extraction
        3. Response cleaning
        """
        from mlx_manager.mlx_server.models.ir import TextResult

        tool_calls_list = self.tool_parser.extract(raw_text)
        reasoning_content = self.thinking_parser.extract(raw_text)

        final_content = raw_text
        if reasoning_content:
            final_content = self.thinking_parser.remove(final_content)
        final_content = self.clean_response(final_content)

        tool_calls = None
        if tool_calls_list:
            tool_calls = [tc.model_dump() for tc in tool_calls_list]
            finish_reason = "tool_calls"

        return TextResult(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )


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
        effective, native_tools = self._prepare_tools(messages, tools)
        kwargs: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": False,
            "enable_thinking": True,
        }
        if native_tools:
            kwargs["tools"] = native_tools
        try:
            return cast(
                str,
                self._actual_tokenizer.apply_chat_template(effective, **kwargs),
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            del kwargs["enable_thinking"]
            return cast(
                str,
                self._actual_tokenizer.apply_chat_template(effective, **kwargs),
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
        """Apply GLM4 template with native tool support."""
        effective, native_tools = self._prepare_tools(messages, tools)
        if hasattr(self._actual_tokenizer, "apply_chat_template"):
            try:
                kwargs: dict[str, Any] = {
                    "add_generation_prompt": add_generation_prompt,
                    "tokenize": False,
                }
                if native_tools:
                    kwargs["tools"] = native_tools
                return cast(
                    str,
                    self._actual_tokenizer.apply_chat_template(effective, **kwargs),
                )
            except Exception as e:
                logger.warning("GLM4 tokenizer.apply_chat_template failed: {}", e)
        # Manual ChatML fallback
        parts: list[str] = []
        for msg in effective:
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
        effective, native_tools = self._prepare_tools(messages, tools)
        # Mistral v1/v2: merge system message into first user message
        processed = list(effective)
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
        return cast(
            str,
            self._actual_tokenizer.apply_chat_template(processed, **kwargs),
        )


class LiquidAdapter(ModelAdapter):
    """Adapter for LiquidAI LFM2/LFM2.5 family models."""

    @property
    def family(self) -> str:
        return "liquid"

    def _default_tool_parser(self) -> ToolCallParser:
        from mlx_manager.mlx_server.parsers.tool_call import LiquidPythonParser

        return LiquidPythonParser()

    def _default_thinking_parser(self) -> ThinkingParser:
        return ThinkTagParser()

    def supports_native_tools(self) -> bool:
        return True

    def _compute_stop_tokens(self) -> list[int]:
        stop_ids = []
        eos = getattr(self._actual_tokenizer, "eos_token_id", None)
        if eos is not None:
            stop_ids.append(eos)
        for token_str in ("<|im_end|>",):
            try:
                tid = self._actual_tokenizer.convert_tokens_to_ids(token_str)
                if (
                    tid is not None
                    and tid != getattr(self._actual_tokenizer, "unk_token_id", None)
                    and tid not in stop_ids
                ):
                    stop_ids.append(tid)
            except Exception:
                pass
        return stop_ids


# --- Audio Adapters ---


class BaseAudioAdapter(ModelAdapter):
    """Base adapter for audio models (TTS/STT).

    Audio models have no tokenizer, so stop_tokens are empty
    and tool/thinking parsers are null.
    """

    @property
    def family(self) -> str:
        return "audio_default"

    def _default_tool_parser(self) -> ToolCallParser:
        return NullToolParser()

    def _default_thinking_parser(self) -> ThinkingParser:
        return NullThinkingParser()


class WhisperAdapter(BaseAudioAdapter):
    """Whisper family: STT models that may need processor fixups.

    MLX-community whisper repos often lack preprocessor_config.json.
    When mlx-audio loads these models, _processor is set to None,
    causing STT to fail. post_load_configure loads the processor
    from the canonical OpenAI repo as a fallback.
    """

    @property
    def family(self) -> str:
        return "whisper"

    async def post_load_configure(self, model: Any, model_id: str) -> None:
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


class KokoroAdapter(BaseAudioAdapter):
    """Kokoro family: TTS models."""

    @property
    def family(self) -> str:
        return "kokoro"


class DefaultAudioAdapter(BaseAudioAdapter):
    """Catchall adapter for unknown audio model families."""


class EmbeddingsAdapter(ModelAdapter):
    """Adapter for embedding models.

    Embedding models have no streaming, tool calling, or thinking support.
    They use a simple batch tokenize → forward pass → normalize pipeline.
    """

    @property
    def family(self) -> str:
        return "embeddings"

    def _default_tool_parser(self) -> ToolCallParser:
        return NullToolParser()

    def _default_thinking_parser(self) -> ThinkingParser:
        return NullThinkingParser()


# --- Factory ---

# Family name -> composable adapter class
FAMILY_REGISTRY: dict[str, type[ModelAdapter]] = {
    "qwen": QwenAdapter,
    "glm4": GLM4Adapter,
    "llama": LlamaAdapter,
    "gemma": GemmaAdapter,
    "mistral": MistralAdapter,
    "liquid": LiquidAdapter,
    "whisper": WhisperAdapter,
    "kokoro": KokoroAdapter,
    "audio_default": DefaultAudioAdapter,
    "embeddings": EmbeddingsAdapter,
    "default": DefaultAdapter,
}


def create_adapter(
    family: str,
    tokenizer: Any | None = None,
    tool_parser: ToolCallParser | None = None,
    thinking_parser: ThinkingParser | None = None,
    model_id: str | None = None,
) -> ModelAdapter:
    """Create a composable adapter for a model family.

    Args:
        family: Model family name (e.g., "qwen", "llama", "whisper")
        tokenizer: HuggingFace tokenizer or processor (None for audio)
        tool_parser: Override default tool parser
        thinking_parser: Override default thinking parser
        model_id: Model identifier (needed for vision models to load config)

    Returns:
        ModelAdapter instance
    """
    cls = FAMILY_REGISTRY.get(family, DefaultAdapter)
    return cls(
        tokenizer=tokenizer,
        tool_parser=tool_parser,
        thinking_parser=thinking_parser,
        model_id=model_id,
    )
