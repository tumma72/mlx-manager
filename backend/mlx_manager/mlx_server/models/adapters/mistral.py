"""Mistral model family adapter (Mistral, Mixtral).

Mistral v1/v2 uses [INST] ... [/INST] format without native system message support.
Mistral v3+ has native system message support in tokenizer template.

This adapter prepends system message to first user message for v1/v2 compatibility.
"""

from typing import Any, cast

from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter


class MistralAdapter(DefaultAdapter):
    """Adapter for Mistral model family."""

    @property
    def family(self) -> str:
        """Return 'mistral' family identifier."""
        return "mistral"

    def apply_chat_template(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply Mistral chat template with system message handling.

        For Mistral v1/v2 models that don't support system role natively,
        prepend system message content to the first user message.
        Mistral v3+ tokenizers handle system messages correctly.
        """
        processed = list(messages)

        # Handle system message for older Mistral versions
        if processed and processed[0].get("role") == "system":
            system_content = processed[0].get("content", "")
            processed = processed[1:]

            # Prepend to first user message if exists
            if processed and processed[0].get("role") == "user":
                user_content = processed[0].get("content", "")
                processed[0] = {
                    "role": "user",
                    "content": f"{system_content}\n\n{user_content}",
                }

        # Use tokenizer's built-in template (works for v3+, fallback for v1/v2)
        result: str = cast(
            str,
            tokenizer.apply_chat_template(
                processed,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            ),
        )
        return result

    def get_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Get Mistral stop tokens.

        Mistral uses </s> as the end-of-sequence token.

        Handles both Tokenizer and Processor objects (vision models use Processor).
        """
        # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        return [actual_tokenizer.eos_token_id]
