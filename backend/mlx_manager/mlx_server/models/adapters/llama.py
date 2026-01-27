"""Llama family model adapter.

Supports:
- Llama 3.x (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, etc.)
- CodeLlama variants
- Meta Llama models

Critical: Llama 3 requires TWO stop tokens for proper chat completion:
1. eos_token_id (<|end_of_text|>)
2. <|eot_id|> (end of turn)

Without both, the model continues generating past the assistant's response.
"""

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


class LlamaAdapter:
    """Adapter for Llama family models."""

    @property
    def family(self) -> str:
        """Return 'llama' family identifier."""
        return "llama"

    def apply_chat_template(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply Llama chat template.

        Uses tokenizer's built-in template which handles:
        - <|begin_of_text|> prefix
        - <|start_header_id|>{role}<|end_header_id|> markers
        - <|eot_id|> after each message
        """
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
        """Get Llama 3 stop tokens.

        CRITICAL: Must include both:
        - eos_token_id: <|end_of_text|> (128009 for Llama 3)
        - <|eot_id|>: end of turn (128001 for Llama 3)

        The model signals end-of-message with <|eot_id|> but continues
        if only eos_token_id is checked.
        """
        stop_tokens: list[int] = [tokenizer.eos_token_id]

        # Add <|eot_id|> for Llama 3.x
        try:
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != tokenizer.unk_token_id:
                stop_tokens.append(eot_id)
                logger.debug("Llama adapter: added <|eot_id|> (%d) to stop tokens", eot_id)
        except Exception as e:
            logger.warning("Could not get <|eot_id|> token: %s", e)

        # Also check for end_of_turn if present (some variants)
        try:
            end_turn_id = tokenizer.convert_tokens_to_ids("<|end_of_turn|>")
            if end_turn_id is not None and end_turn_id != tokenizer.unk_token_id:
                if end_turn_id not in stop_tokens:
                    stop_tokens.append(end_turn_id)
        except Exception:
            pass  # Not all models have this token

        return stop_tokens

    def is_stop_token(self, token_id: int, tokenizer: Any) -> bool:
        """Check if a token ID is a stop token.

        Convenience method for generation loop.
        """
        return token_id in self.get_stop_tokens(tokenizer)
