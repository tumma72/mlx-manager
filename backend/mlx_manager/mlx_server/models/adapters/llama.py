"""Llama family model adapter.

Supports:
- Llama 3.x (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, etc.)
- CodeLlama variants
- Meta Llama models
- Llama-thinking variants with <think> tag reasoning mode

Critical: Llama 3 requires TWO stop tokens for proper chat completion:
1. eos_token_id (<|end_of_text|>)
2. <|eot_id|> (end of turn)

Without both, the model continues generating past the assistant's response.

Tool call parsing and reasoning extraction are handled by ResponseProcessor.
This adapter provides chat template formatting, stop token configuration,
and tool prompt formatting for Llama XML-style tool calls.
"""

import json
from typing import Any, cast

from loguru import logger

from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter


class LlamaAdapter(DefaultAdapter):
    """Adapter for Llama family models.

    Supports reasoning mode for Llama-thinking variants that output
    chain-of-thought content in <think> tags.
    """

    @property
    def family(self) -> str:
        """Return 'llama' family identifier."""
        return "llama"

    def apply_chat_template(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Apply Llama chat template.

        Uses tokenizer's built-in template which handles:
        - <|begin_of_text|> prefix
        - <|start_header_id|>{role}<|end_header_id|> markers
        - <|eot_id|> after each message

        Handles both Tokenizer and Processor objects (vision models use Processor).
        """
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        result: str = cast(
            str,
            actual_tokenizer.apply_chat_template(
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

        Handles both Tokenizer and Processor objects (vision models use Processor).
        """
        # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        stop_tokens: list[int] = [actual_tokenizer.eos_token_id]

        # Add <|eot_id|> for Llama 3.x
        try:
            eot_id = actual_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != actual_tokenizer.unk_token_id:
                stop_tokens.append(eot_id)
                logger.debug("Llama adapter: added <|eot_id|> (%d) to stop tokens", eot_id)
        except Exception as e:
            logger.warning("Could not get <|eot_id|> token: %s", e)

        # Also check for end_of_turn if present (some variants)
        try:
            end_turn_id = actual_tokenizer.convert_tokens_to_ids("<|end_of_turn|>")
            if end_turn_id is not None and end_turn_id != actual_tokenizer.unk_token_id:
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

    # --- Tool Calling Support ---

    def supports_tool_calling(self) -> bool:
        """Check if this model family supports tool calling.

        Returns:
            True - Llama 3.x supports tool calling
        """
        return True

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions for inclusion in system prompt.

        Llama 3.x uses XML-style tool call format:
        <function=name>{"param": "value"}</function>

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

            doc = f"""{name}:
  description: {description}
  parameters: {json.dumps(parameters, indent=2)}"""
            tool_docs.append(doc)

        return f"""You have access to the following functions:

{chr(10).join(tool_docs)}

To call a function, respond with:
<function=function_name>{{"param": "value"}}</function>

Only call functions when necessary. If no function call is needed, respond normally."""

    def get_tool_call_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Get additional stop tokens to use when tools are enabled.

        Llama 3.x stops at </function> and <|eom_id|> for tool calls.

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            List of token IDs that indicate tool call completion
        """
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        stop_tokens: list[int] = []

        # Try to get </function> token (may not exist in all tokenizers)
        # The string "</function>" may be tokenized as multiple tokens
        # so we look for <|eom_id|> which is a special token for tool completion
        try:
            eom_id = actual_tokenizer.convert_tokens_to_ids("<|eom_id|>")
            if eom_id is not None and eom_id != actual_tokenizer.unk_token_id:
                stop_tokens.append(eom_id)
                logger.debug("Llama tool stop token: <|eom_id|> (%d)", eom_id)
        except Exception:
            pass

        return stop_tokens

    # --- Message Conversion ---

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Llama-compatible format.

        Llama tokenizers cannot handle role="tool" messages or assistant messages
        with tool_calls in their Jinja templates. This method converts:
        - role="tool" -> role="user" with structured tool result text
        - assistant with tool_calls -> assistant with <function=name> tags

        Args:
            messages: Messages in OpenAI format (may include tool role)

        Returns:
            Messages with tool messages converted to Llama-friendly format
        """
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "tool":
                # Convert tool result to user message
                tool_call_id = msg.get("tool_call_id", "unknown")
                content = msg.get("content", "")
                tool_result_content = (
                    f"[Tool Result for {tool_call_id}]\n{content}\n[End Tool Result]\n\n"
                    f"Please provide your response based on this tool result."
                )
                converted.append({"role": "user", "content": tool_result_content})
            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant tool_calls to Llama XML-style text
                tool_calls = msg.get("tool_calls", [])
                tool_text = ""
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    tool_text += f"\n<function={name}>{args}</function>"
                content = (msg.get("content", "") or "") + tool_text
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append(msg)

        return converted

    # --- Reasoning Mode Support ---

    def supports_reasoning_mode(self) -> bool:
        """Check if this model supports thinking/reasoning output.

        Returns True because Llama-thinking variants use <think> tags
        for chain-of-thought content. Not all Llama models output reasoning,
        but the adapter reports the capability; extraction only happens
        when the model actually outputs reasoning tags.

        Returns:
            True - Llama family supports reasoning mode
        """
        return True
