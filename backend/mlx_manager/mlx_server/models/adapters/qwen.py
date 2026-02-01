"""Qwen model family adapter (Qwen, Qwen2, Qwen2.5, Qwen3).

Qwen models use ChatML format with <|im_start|> and <|im_end|> tokens.
Qwen3-thinking variants output chain-of-thought content in <think> tags.
"""

from typing import Any, cast

from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter
from mlx_manager.mlx_server.services.reasoning import ReasoningExtractor

# Module-level extractor instance for reasoning extraction
_reasoning_extractor = ReasoningExtractor()


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
