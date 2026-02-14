"""Anthropic Messages protocol formatter.

Converts IR types (StreamEvent, TextResult) into Anthropic Messages API
format, including the SSE event sequence for streaming.

Absorbs the output formatting logic from ProtocolTranslator and the
streaming event generation from messages.py.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult
from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    TextBlock,
    Usage,
)
from mlx_manager.mlx_server.services.formatters.base import ProtocolFormatter
from mlx_manager.models.value_objects import InferenceParams

# Stop reason mapping: OpenAI → Anthropic
_STOP_REASON_TO_ANTHROPIC: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "end_turn",
    "tool_calls": "tool_use",
}

# Stop reason mapping: Anthropic → OpenAI
_STOP_REASON_TO_OPENAI: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}

AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


class InternalRequest(BaseModel):
    """Internal request format used by inference service."""

    model: str
    messages: list[dict[str, str]]
    params: InferenceParams
    stream: bool = False
    stop: list[str] | None = None


def openai_stop_to_anthropic(stop_reason: str | None) -> str:
    """Convert OpenAI stop reason to Anthropic format."""
    if stop_reason is None:
        return "end_turn"
    return _STOP_REASON_TO_ANTHROPIC.get(stop_reason, "end_turn")


def anthropic_stop_to_openai(stop_reason: str | None) -> str:
    """Convert Anthropic stop reason to OpenAI format."""
    if stop_reason is None:
        return "stop"
    return _STOP_REASON_TO_OPENAI.get(stop_reason, "stop")


def _extract_text_content(content: str | list[Any]) -> str:
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


class AnthropicFormatter(ProtocolFormatter):
    """Formats IR types into Anthropic Messages API protocol.

    Streaming produces named SSE events following the Anthropic spec:
        message_start → content_block_start → content_block_delta(s)
        → content_block_stop → message_delta → message_stop

    Non-streaming produces an AnthropicMessagesResponse Pydantic model.
    """

    # ── input parsing ────────────────────────────────────────────────

    @staticmethod
    def parse_request(request: AnthropicMessagesRequest) -> InternalRequest:
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
            content = _extract_text_content(msg.content)
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

    # ── streaming ────────────────────────────────────────────────────

    def stream_start(self) -> list[dict[str, Any]]:
        return [
            {
                "event": "message_start",
                "data": json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": self.request_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model_id,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    }
                ),
            },
            {
                "event": "content_block_start",
                "data": json.dumps(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                ),
            },
        ]

    def stream_event(self, event: StreamEvent) -> list[dict[str, Any]]:
        # Anthropic streams text content only (reasoning is not in Anthropic spec)
        text = event.content or ""
        if not text:
            return []
        return [
            {
                "event": "content_block_delta",
                "data": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    }
                ),
            }
        ]

    def stream_end(
        self,
        finish_reason: str,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        output_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        anthropic_stop = openai_stop_to_anthropic(finish_reason)
        return [
            {
                "event": "content_block_stop",
                "data": json.dumps(
                    {
                        "type": "content_block_stop",
                        "index": 0,
                    }
                ),
            },
            {
                "event": "message_delta",
                "data": json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": anthropic_stop,
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": output_tokens},
                    }
                ),
            },
            {
                "event": "message_stop",
                "data": json.dumps({"type": "message_stop"}),
            },
        ]

    # ── non-streaming ────────────────────────────────────────────────

    def format_complete(
        self,
        result: TextResult,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> AnthropicMessagesResponse:
        anthropic_stop: AnthropicStopReason = openai_stop_to_anthropic(  # type: ignore[assignment]
            result.finish_reason
        )
        return AnthropicMessagesResponse(
            id=self.request_id,
            model=self.model_id,
            content=[TextBlock(text=result.content)],
            stop_reason=anthropic_stop,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            ),
        )
