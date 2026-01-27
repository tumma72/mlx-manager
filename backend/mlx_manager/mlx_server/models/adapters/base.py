"""Model adapter base protocol and default implementation."""

from typing import Any, Protocol, cast, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for model-specific handling.

    Each model family (Llama, Qwen, Mistral, etc.) has specific:
    - Chat template formatting
    - Stop token configuration
    - Tool call parsing (future)
    - Thinking token handling (future)
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


class DefaultAdapter:
    """Default adapter using tokenizer's built-in chat template.

    Used for unknown model families as a fallback.
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
