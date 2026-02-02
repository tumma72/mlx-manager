"""Model adapter base protocol and default implementation."""

import re
from typing import Any, Protocol, cast, runtime_checkable

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


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for model-specific handling.

    Each model family (Llama, Qwen, Mistral, etc.) has specific:
    - Chat template formatting
    - Stop token configuration
    - Tool call parsing
    - Thinking/reasoning token handling
    - Message conversion

    The protocol defines required methods (family, apply_chat_template, get_stop_tokens)
    and optional methods with sensible defaults for extended functionality.
    """

    @property
    def family(self) -> str:
        """Model family identifier (e.g., 'llama', 'qwen')."""
        ...

    def apply_chat_template(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply model-specific chat template.

        Args:
            tokenizer: HuggingFace tokenizer
            messages: List of {"role": str, "content": str} dicts
            add_generation_prompt: Whether to add assistant role marker

        Returns:
            Formatted prompt string
        """
        ...

    def get_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Get model-specific stop token IDs.

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            List of token IDs that should stop generation
        """
        ...

    # --- Optional Methods: Tool Calling Support ---

    def supports_tool_calling(self) -> bool:
        """Check if this model family supports tool calling.

        Returns:
            True if the model supports tool calling, False otherwise
        """
        ...

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]] | None:
        """Parse tool calls from model output text.

        Args:
            text: Model output text that may contain tool calls

        Returns:
            List of tool call dicts with format:
            [{"id": str, "type": "function", "function": {"name": str, "arguments": str}}]
            Returns None if no tool calls found.
        """
        ...

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions for inclusion in system prompt.

        Args:
            tools: List of tool definitions in OpenAI format

        Returns:
            Formatted string to append to system prompt
        """
        ...

    def get_tool_call_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Get additional stop tokens to use when tools are enabled.

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            List of token IDs that indicate tool call completion
        """
        ...

    # --- Optional Methods: Reasoning Mode Support ---

    def supports_reasoning_mode(self) -> bool:
        """Check if this model supports thinking/reasoning output.

        Returns:
            True if the model supports reasoning mode (e.g., <think> tags)
        """
        ...

    def extract_reasoning(self, text: str) -> tuple[str | None, str]:
        """Extract reasoning content from response.

        Args:
            text: Model output text that may contain reasoning tags

        Returns:
            Tuple of (reasoning_content, final_content).
            reasoning_content is None if no reasoning found.
        """
        ...

    # --- Optional Methods: Message Conversion ---

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to model-specific format.

        Some models require specific message formats, tool result handling,
        or system message placement that differs from OpenAI's format.

        Args:
            messages: Messages in OpenAI format

        Returns:
            Messages in model-specific format
        """
        ...

    def clean_response(self, text: str) -> str:
        """Clean response text by removing tool calls and special tokens.

        Args:
            text: Raw model output

        Returns:
            Cleaned text suitable for display
        """
        ...


class DefaultAdapter:
    """Default adapter using tokenizer's built-in chat template.

    Used for unknown model families as a fallback.
    Provides sensible default implementations for all optional methods.
    """

    @property
    def family(self) -> str:
        """Return 'default' family identifier."""
        return "default"

    def apply_chat_template(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Use tokenizer's built-in chat template."""
        result: str = cast(
            str,
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            ),
        )
        return result

    def get_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Return only the standard EOS token."""
        return [tokenizer.eos_token_id]

    # --- Optional Methods: Tool Calling Support ---

    def supports_tool_calling(self) -> bool:
        """Default: Tool calling not supported."""
        return False

    def parse_tool_calls(self, text: str) -> list[dict[str, Any]] | None:
        """Default: No tool calls parsed."""
        return None

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Default: No tool formatting."""
        return ""

    def get_tool_call_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Default: No additional stop tokens for tools."""
        return []

    # --- Optional Methods: Reasoning Mode Support ---

    def supports_reasoning_mode(self) -> bool:
        """Default: Reasoning mode not supported."""
        return False

    def extract_reasoning(self, text: str) -> tuple[str | None, str]:
        """Default: Return text unchanged with no reasoning."""
        return None, text

    # --- Optional Methods: Message Conversion ---

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Default: Return messages unchanged."""
        return messages

    # --- Optional Methods: Response Cleaning ---

    def clean_response(self, text: str) -> str:
        """Clean response text by removing special tokens.

        Override in subclasses for model-specific cleaning (e.g., tool call removal).
        """
        cleaned = text

        # Remove common special tokens
        for token in COMMON_SPECIAL_TOKENS:
            cleaned = cleaned.replace(token, "")

        # Clean up excessive whitespace from removals
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

        return cleaned
