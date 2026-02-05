"""Qwen model family adapter (Qwen, Qwen2, Qwen2.5, Qwen3).

Qwen models use ChatML format with <|im_start|> and <|im_end|> tokens.
Qwen3-thinking variants output chain-of-thought content in <think> tags.

Tool call parsing and reasoning extraction are handled by ResponseProcessor.
This adapter provides chat template formatting, stop token configuration,
and tool prompt formatting for Hermes-style tool calls.
"""

import json
from typing import Any, cast

from loguru import logger

from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter


class QwenAdapter(DefaultAdapter):
    """Adapter for Qwen model family.

    Supports reasoning mode for Qwen3-thinking variants that output
    chain-of-thought content in <think> tags.
    """

    @property
    def family(self) -> str:
        """Return 'qwen' family identifier."""
        return "qwen"

    def apply_chat_template(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Apply Qwen chat template using tokenizer's built-in template.

        For Qwen3 models, enables thinking mode which wraps reasoning
        in <think>...</think> tags.

        Args:
            tools: Ignored - Qwen uses prompt injection for tools, not native support.
        """
        # Note: tools parameter ignored - Qwen uses prompt injection via format_tools_for_prompt
        # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        # Try with enable_thinking for Qwen3 (wraps reasoning in <think> tags)
        try:
            result: str = cast(
                str,
                actual_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                    enable_thinking=True,  # Qwen3 thinking mode
                ),
            )
            logger.info("Applied Qwen3 chat template with enable_thinking=True")
            logger.debug(f"Chat template result (last 200 chars): ...{result[-200:]}")
            return result
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # Older tokenizers don't support enable_thinking parameter
            logger.debug(f"Tokenizer doesn't support enable_thinking, falling back: {e}")
            result = cast(
                str,
                actual_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                ),
            )
            return result

    def get_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Get Qwen stop tokens.

        Qwen uses ChatML format with <|im_end|> as end-of-turn marker.
        Must include both eos_token_id and <|im_end|> to prevent runaway generation.

        Handles both Tokenizer and Processor objects (vision models use Processor).
        """
        # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        stop_tokens = [actual_tokenizer.eos_token_id]

        # Add <|im_end|> token (ChatML end of turn)
        try:
            im_end_id = actual_tokenizer.convert_tokens_to_ids("<|im_end|>")
            # Check it's a valid token (not None or unk)
            if im_end_id is not None and im_end_id != actual_tokenizer.unk_token_id:
                stop_tokens.append(im_end_id)
        except Exception:
            pass

        return stop_tokens

    # --- Tool Calling Support ---

    def supports_tool_calling(self) -> bool:
        """Check if this model family supports tool calling.

        Returns:
            True - Qwen supports tool calling via Hermes format
        """
        return True

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions for inclusion in system prompt.

        Qwen uses Hermes-style tool call format:
        <tool_call>{"name": "func", "arguments": {...}}</tool_call>

        Args:
            tools: List of tool definitions in OpenAI format

        Returns:
            Formatted string to append to system prompt
        """
        if not tools:
            return ""

        tool_docs: list[str] = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            doc = f"""{{
  "name": "{name}",
  "description": "{description}",
  "parameters": {json.dumps(parameters)}
}}"""
            tool_docs.append(doc)

        return f"""<tools>
{chr(10).join(tool_docs)}
</tools>

When you need to call a tool, respond with:
<tool_call>{{"name": "function_name", "arguments": {{"param": "value"}}}}</tool_call>

Only call tools when necessary. If no tool call is needed, respond normally."""

    def get_tool_call_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Get additional stop tokens to use when tools are enabled.

        Qwen stops at </tool_call> for tool completions.

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            List of token IDs that indicate tool call completion
        """
        # </tool_call> is typically tokenized as multiple tokens
        # We rely on the regular stop tokens for now
        # ResponseProcessor will detect tool calls in the output
        return []

    # --- Reasoning Mode Support ---

    def supports_reasoning_mode(self) -> bool:
        """Check if this model supports thinking/reasoning output.

        Returns True because Qwen3-thinking variants use <think> tags
        for chain-of-thought content. Not all Qwen models output reasoning,
        but the adapter reports the capability; extraction only happens
        when the model actually outputs reasoning tags.

        Returns:
            True - Qwen family supports reasoning mode
        """
        return True
