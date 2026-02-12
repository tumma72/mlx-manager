"""Protocol translation between OpenAI and Anthropic formats."""

from typing import Any

from pydantic import BaseModel

from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    TextBlock,
)
from mlx_manager.mlx_server.schemas.anthropic import Usage as AnthropicUsage
from mlx_manager.models.value_objects import InferenceParams


class InternalRequest(BaseModel):
    """Internal request format used by inference service."""

    model: str
    messages: list[dict[str, str]]
    params: InferenceParams
    stream: bool = False
    stop: list[str] | None = None


class ProtocolTranslator:
    """Bidirectional translation between OpenAI and Anthropic formats."""

    # Stop reason mapping: Anthropic -> OpenAI
    STOP_REASON_TO_OPENAI: dict[str, str] = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }

    # Stop reason mapping: OpenAI -> Anthropic
    STOP_REASON_TO_ANTHROPIC: dict[str, str] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
        "tool_calls": "tool_use",
    }

    def anthropic_to_internal(self, request: AnthropicMessagesRequest) -> InternalRequest:
        """Convert Anthropic Messages request to internal format."""
        messages: list[dict[str, str]] = []

        # Handle system prompt (Anthropic has separate field)
        if request.system:
            if isinstance(request.system, str):
                system_text = request.system
            else:
                # List of TextBlockParam
                system_text = " ".join(b.text for b in request.system)
            messages.append({"role": "system", "content": system_text})

        # Convert content blocks to simple content
        for msg in request.messages:
            content = self._extract_text_content(msg.content)
            messages.append({"role": msg.role, "content": content})

        return InternalRequest(
            model=request.model,
            messages=messages,
            params=InferenceParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            ),
            stream=request.stream,
            stop=request.stop_sequences,
        )

    def _extract_text_content(self, content: str | list[Any]) -> str:
        """Extract text from content (string or list of blocks).

        Args:
            content: Either a string or list of content blocks (TextBlockParam,
                    ImageBlockParam, or dict representations)

        Returns:
            Concatenated text from all text blocks
        """
        if isinstance(content, str):
            return content

        text_parts: list[str] = []
        for block in content:
            if hasattr(block, "type") and hasattr(block, "text"):
                if block.type == "text":
                    text_parts.append(block.text)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
        return " ".join(text_parts)

    def internal_to_anthropic_response(
        self,
        response_text: str,
        request_id: str,
        model: str,
        stop_reason: str,
        input_tokens: int,
        output_tokens: int,
    ) -> AnthropicMessagesResponse:
        """Convert internal response to Anthropic format."""
        anthropic_stop_reason = self.STOP_REASON_TO_ANTHROPIC.get(stop_reason, "end_turn")

        return AnthropicMessagesResponse(
            id=request_id,
            model=model,
            content=[TextBlock(text=response_text)],
            stop_reason=anthropic_stop_reason,  # type: ignore[arg-type]
            usage=AnthropicUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    def openai_stop_to_anthropic(self, stop_reason: str | None) -> str:
        """Convert OpenAI stop reason to Anthropic format."""
        if stop_reason is None:
            return "end_turn"
        return self.STOP_REASON_TO_ANTHROPIC.get(stop_reason, "end_turn")

    def anthropic_stop_to_openai(self, stop_reason: str | None) -> str:
        """Convert Anthropic stop reason to OpenAI format."""
        if stop_reason is None:
            return "stop"
        return self.STOP_REASON_TO_OPENAI.get(stop_reason, "stop")


# Module-level singleton
_translator: ProtocolTranslator | None = None


def get_translator() -> ProtocolTranslator:
    """Get the protocol translator singleton."""
    global _translator
    if _translator is None:
        _translator = ProtocolTranslator()
    return _translator


def reset_translator() -> None:
    """Reset the translator singleton (for testing)."""
    global _translator
    _translator = None
