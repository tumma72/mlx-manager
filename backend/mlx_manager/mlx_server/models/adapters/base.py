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
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Apply model-specific chat template.

        Args:
            tokenizer: HuggingFace tokenizer
            messages: List of {"role": str, "content": str} dicts
            add_generation_prompt: Whether to add assistant role marker
            tools: Optional list of tool definitions for models with native tool support

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

    def has_native_tool_support(self, tokenizer: Any) -> bool:
        """Check if this tokenizer supports native tool calling.

        Native tool support means the tokenizer's apply_chat_template accepts
        a tools parameter and handles tool formatting internally.

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            True if tokenizer supports native tools via apply_chat_template.
        """
        ...

    # --- Optional Methods: Reasoning Mode Support ---

    def supports_reasoning_mode(self) -> bool:
        """Check if this model supports thinking/reasoning output.

        Returns:
            True if the model supports reasoning mode (e.g., <think> tags)
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
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Use tokenizer's built-in chat template.

        Handles both Tokenizer and Processor objects (vision models use Processor).
        The tools parameter is accepted but ignored by default - models with native
        tool support can override this to pass tools to the tokenizer.
        """
        # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        # Note: tools parameter ignored by default adapter - handled via prompt injection
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
        """Return only the standard EOS token.

        Handles both Tokenizer and Processor objects (vision models use Processor).
        """
        # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        return [actual_tokenizer.eos_token_id]

    # --- Optional Methods: Tool Calling Support ---

    def supports_tool_calling(self) -> bool:
        """Default: Tool calling not supported."""
        return False

    def format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Default: No tool formatting."""
        return ""

    def get_tool_call_stop_tokens(self, tokenizer: Any) -> list[int]:
        """Default: No additional stop tokens for tools."""
        return []

    def has_native_tool_support(self, tokenizer: Any) -> bool:
        """Default: No native tool support, use prompt injection."""
        return False

    # --- Optional Methods: Reasoning Mode Support ---

    def supports_reasoning_mode(self) -> bool:
        """Default: Reasoning mode not supported."""
        return False

    # --- Optional Methods: Message Conversion ---

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Default: Safe fallback that converts tool messages to user messages.

        Tokenizers generally cannot handle role="tool" or assistant messages
        with tool_calls in their Jinja templates. This default implementation
        converts them to a format any tokenizer can handle:
        - role="tool" -> role="user" with structured text
        - assistant with tool_calls -> plain assistant with text representation

        Adapters for tool-capable families (Qwen, Llama, GLM4) should override
        this with family-specific formatting.
        """
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "tool":
                # Convert tool result to user message (safe for any tokenizer)
                tool_call_id = msg.get("tool_call_id", "unknown")
                content = msg.get("content", "")
                converted.append(
                    {
                        "role": "user",
                        "content": f"[Tool Result for {tool_call_id}]\n{content}",
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant tool_calls to text representation
                content = msg.get("content", "") or ""
                tool_calls = msg.get("tool_calls", [])
                tool_text_parts = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    tool_text_parts.append(f"[Tool Call: {name}({args})]")
                tool_text = "\n".join(tool_text_parts)
                if tool_text_parts:
                    content = f"{content}\n{tool_text}" if content else tool_text
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append(msg)

        return converted

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
