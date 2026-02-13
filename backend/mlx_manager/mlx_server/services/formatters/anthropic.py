"""Anthropic Messages protocol formatter.

Converts IR types (StreamEvent, TextResult) into Anthropic Messages API
format, including the SSE event sequence for streaming.

Absorbs the output formatting logic from ProtocolTranslator and the
streaming event generation from messages.py.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult
from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesResponse,
    TextBlock,
    Usage,
)
from mlx_manager.mlx_server.services.formatters.base import ProtocolFormatter

# Stop reason mapping: OpenAI → Anthropic
_STOP_REASON_TO_ANTHROPIC: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "end_turn",
    "tool_calls": "tool_use",
}

AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


def openai_stop_to_anthropic(stop_reason: str | None) -> str:
    """Convert OpenAI stop reason to Anthropic format."""
    if stop_reason is None:
        return "end_turn"
    return _STOP_REASON_TO_ANTHROPIC.get(stop_reason, "end_turn")


class AnthropicFormatter(ProtocolFormatter):
    """Formats IR types into Anthropic Messages API protocol.

    Streaming produces named SSE events following the Anthropic spec:
        message_start → content_block_start → content_block_delta(s)
        → content_block_stop → message_delta → message_stop

    Non-streaming produces an AnthropicMessagesResponse Pydantic model.
    """

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
