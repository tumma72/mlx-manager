"""GLM4 model family adapter (GLM-4, ChatGLM-4).

GLM4 models use ChatML-like format with special token handling.

Tool call parsing and reasoning extraction are now handled by ResponseProcessor.
This adapter provides chat template formatting and stop token configuration.
"""

import logging
from typing import Any, cast

from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter

logger = logging.getLogger(__name__)


class GLM4Adapter(DefaultAdapter):
    """Adapter for GLM4 model family.

    Supports:
    - Chat completion with ChatML-like format
    - Tool calling via XML-style format
    - Reasoning mode via <think> tags
    """

    @property
    def family(self) -> str:
        """Return 'glm4' family identifier."""
        return "glm4"

    def apply_chat_template(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply GLM4 chat template.

        GLM4 uses ChatML-like format. The tokenizer should have a built-in template.
        Falls back to manual formatting if no template is available.
        """
        # Try using tokenizer's built-in template first
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                result: str = cast(
                    str,
                    tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=add_generation_prompt,
                        tokenize=False,
                    ),
                )
                return result
            except Exception as e:
                logger.warning("GLM4 tokenizer.apply_chat_template failed: %s", e)

        # Manual fallback using ChatML-like format
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")

        if add_generation_prompt:
            parts.append("<|assistant|>")

        return "\n".join(parts)

    def get_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Get GLM4 stop tokens.

        GLM4 uses special end tokens similar to ChatML.

        Handles both Tokenizer and Processor objects.
        """
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        stop_tokens: list[int] = [actual_tokenizer.eos_token_id]

        # Try to add GLM4-specific stop tokens
        special_tokens = ["<|user|>", "<|observation|>", "<|endoftext|>"]
        for token_str in special_tokens:
            try:
                token_id = actual_tokenizer.convert_tokens_to_ids(token_str)
                if token_id is not None and token_id != actual_tokenizer.unk_token_id:
                    if token_id not in stop_tokens:
                        stop_tokens.append(token_id)
            except Exception:
                pass

        return stop_tokens

    # --- Tool Calling Support ---

    def supports_tool_calling(self) -> bool:
        """Check if this model family supports tool calling.

        Returns:
            True - GLM4 supports tool calling via XML format
        """
        return True

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]] | None:
        """Parse tool calls from model output text.

        GLM4 uses XML format: <tool_call><name>func</name><arguments>{...}</arguments>

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

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            List of token IDs that indicate tool call completion
        """
        # GLM4 tool call markers are multi-token, rely on regular stop tokens
        return []

    # --- Reasoning Mode Support ---

    def supports_reasoning_mode(self) -> bool:
        """Check if this model supports thinking/reasoning output.

        Returns:
            True - GLM4 may support reasoning mode via <think> tags
        """
        return True

    def extract_reasoning(self, text: str) -> tuple[str | None, str]:
        """Extract reasoning content from response.

        Args:
            text: Model output text that may contain reasoning tags

        Returns:
            Tuple of (reasoning_content, final_content).
            reasoning_content is None if no reasoning tags found.
        """
        return _reasoning_extractor.extract(text)
