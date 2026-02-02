"""Qwen model family adapter (Qwen, Qwen2, Qwen2.5, Qwen3).

Qwen models use ChatML format with <|im_start|> and <|im_end|> tokens.
Qwen3-thinking variants output chain-of-thought content in <think> tags.

Tool call parsing and reasoning extraction are now handled by ResponseProcessor.
This adapter provides chat template formatting and stop token configuration.
"""

import logging
from typing import Any, cast

from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter

logger = logging.getLogger(__name__)


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
    ) -> str:
        """Apply Qwen chat template using tokenizer's built-in template.

        For Qwen3 models, enables thinking mode which wraps reasoning
        in <think>...</think> tags.
        """
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
        except TypeError as e:
            # Older tokenizers don't support enable_thinking parameter
            logger.warning(f"Tokenizer doesn't support enable_thinking: {e}")
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

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]] | None:
        """Parse tool calls from model output text.

        Qwen uses Hermes-style format: <tool_call>{"name": ..., "arguments": ...}</tool_call>

        Args:
            text: Model output text that may contain tool calls

        Returns:
            List of tool call dicts in OpenAI format, or None if no calls found.
        """
        calls = _tool_parser.parse(text)
        return calls if calls else None

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions for inclusion in system prompt.

        Args:
            tools: List of tool definitions in OpenAI format

        Returns:
            Formatted string to append to system prompt
        """
        return _tool_parser.format_tools(tools)

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
        # The parser will detect tool calls in the output
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

    def extract_reasoning(self, text: str) -> tuple[str | None, str]:
        """Extract reasoning content from response.

        Qwen3-thinking variants output chain-of-thought in <think> tags.
        Delegates to ReasoningExtractor for pattern matching.

        Args:
            text: Model output text that may contain reasoning tags

        Returns:
            Tuple of (reasoning_content, final_content).
            reasoning_content is None if no reasoning tags found.
        """
        return _reasoning_extractor.extract(text)

    def clean_response(self, text: str) -> str:
        """Clean response by removing tool calls, special tokens, and reasoning tags.

        Args:
            text: Raw model output

        Returns:
            Cleaned text suitable for display
        """
        return _tool_parser.clean_response(text)
