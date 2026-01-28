"""Qwen model family adapter (Qwen, Qwen2, Qwen2.5, Qwen3).

Qwen models use ChatML format with <|im_start|> and <|im_end|> tokens.
"""

from typing import Any, cast


class QwenAdapter:
    """Adapter for Qwen model family."""

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
        """Apply Qwen chat template using tokenizer's built-in template."""
        # Qwen tokenizers have proper chat templates - use them
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
        """Get Qwen stop tokens.

        Qwen uses ChatML format with <|im_end|> as end-of-turn marker.
        Must include both eos_token_id and <|im_end|> to prevent runaway generation.
        """
        stop_tokens = [tokenizer.eos_token_id]

        # Add <|im_end|> token (ChatML end of turn)
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            # Check it's a valid token (not None or unk)
            if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
                stop_tokens.append(im_end_id)
        except Exception:
            pass

        return stop_tokens
