"""Gemma model family adapter (Gemma, Gemma2, Gemma3).

Gemma models use <start_of_turn> and <end_of_turn> tokens for chat formatting.
"""

from typing import Any, cast


class GemmaAdapter:
    """Adapter for Gemma model family."""

    @property
    def family(self) -> str:
        """Return 'gemma' family identifier."""
        return "gemma"

    def apply_chat_template(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply Gemma chat template using tokenizer's built-in template."""
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
        """Get Gemma stop tokens.

        Gemma uses <end_of_turn> as the end-of-turn marker.
        Must include both eos_token_id and <end_of_turn> to prevent runaway generation.
        """
        stop_tokens = [tokenizer.eos_token_id]

        # Add <end_of_turn> token
        try:
            end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_turn_id is not None and end_turn_id != tokenizer.unk_token_id:
                stop_tokens.append(end_turn_id)
        except Exception:
            pass

        return stop_tokens
