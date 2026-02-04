"""GLM4 model family adapter (GLM-4, ChatGLM-4, GLM-4.7).

GLM4 models use ChatML-like format with special token handling.

Tool call parsing and reasoning extraction are handled by ResponseProcessor.
This adapter provides chat template formatting, stop token configuration,
and tool prompt formatting for GLM4 models.

GLM-4.7 models may support native tool calling via tokenizer's apply_chat_template.
For older GLM4 models, falls back to XML-style prompt injection.
"""

import json
import logging
from typing import Any, cast

from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter

logger = logging.getLogger(__name__)

# Store native tool support status per tokenizer to avoid repeated attempts
_native_tools_cache: dict[int, bool] = {}


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

        Handles both Tokenizer and Processor objects (vision models use Processor).
        """
        # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        # Try using tokenizer's built-in template first
        if hasattr(actual_tokenizer, "apply_chat_template"):
            try:
                result: str = cast(
                    str,
                    actual_tokenizer.apply_chat_template(
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

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions for inclusion in system prompt.

        GLM-4.7 works best with JSON-style tool call format similar to Qwen/Hermes:
        <tool_call>{"name": "func", "arguments": {...}}</tool_call>

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

            # Use JSON format like Qwen/Hermes - more compatible with modern models
            doc = f"""{{
  "name": "{name}",
  "description": "{description}",
  "parameters": {json.dumps(parameters)}
}}"""
            tool_docs.append(doc)

        return f"""You have access to the following tools:

<tools>
{chr(10).join(tool_docs)}
</tools>

When you decide to call a tool, you MUST respond with ONLY the tool call in this exact format:
<tool_call>{{"name": "function_name", "arguments": {{"param": "value"}}}}</tool_call>

IMPORTANT: When calling a tool, output ONLY the <tool_call>...</tool_call> block, nothing else.
If you don't need to call a tool, respond normally with text."""

    def get_tool_call_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Get additional stop tokens to use when tools are enabled.

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            List of token IDs that indicate tool call completion
        """
        # GLM4 tool call markers are multi-token, rely on regular stop tokens
        # ResponseProcessor will detect tool calls in the output
        return []

    # --- Message Conversion ---

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to GLM4-compatible format.

        GLM4 tokenizers may not handle 'tool' role messages directly.
        Convert tool results to a format the model understands.

        Args:
            messages: Messages in OpenAI format (may include tool role)

        Returns:
            Messages with tool results converted to user messages
        """
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "tool":
                # Convert tool result to a user message with clear formatting
                tool_call_id = msg.get("tool_call_id", "unknown")
                content = msg.get("content", "")
                converted.append({
                    "role": "user",
                    "content": f"[Tool Result for {tool_call_id}]\n{content}\n[End Tool Result]\n\nPlease provide your response based on this tool result."
                })
            elif role == "assistant" and msg.get("tool_calls"):
                # Include assistant message but convert tool_calls to text format
                tool_calls = msg.get("tool_calls", [])
                tool_text = ""
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_text += f"\n<tool_call>{json.dumps({'name': func.get('name'), 'arguments': json.loads(func.get('arguments', '{}'))})}</tool_call>"
                content = msg.get("content", "") + tool_text
                converted.append({
                    "role": "assistant",
                    "content": content
                })
            else:
                # Keep other messages as-is
                converted.append(msg)

        return converted

    # --- Reasoning Mode Support ---

    def supports_reasoning_mode(self) -> bool:
        """Check if this model supports thinking/reasoning output.

        Returns:
            True - GLM4 may support reasoning mode via <think> tags
        """
        return True
