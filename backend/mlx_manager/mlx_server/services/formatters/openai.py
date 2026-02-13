"""OpenAI protocol formatter.

Converts IR types (StreamEvent, TextResult) into OpenAI Chat Completion
format dicts, ready for EventSourceResponse (streaming) or JSON response.
"""

from __future__ import annotations

import json
from typing import Any

from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult
from mlx_manager.mlx_server.services.formatters.base import ProtocolFormatter


class OpenAIFormatter(ProtocolFormatter):
    """Formats IR types into OpenAI Chat Completion protocol.

    Streaming produces ``data:`` SSE events with chat.completion.chunk objects.
    Non-streaming produces a chat.completion response dict.
    """

    # ── streaming ────────────────────────────────────────────────────

    def stream_start(self) -> list[dict[str, Any]]:
        chunk = self._chunk(delta={"role": "assistant", "content": ""})
        return [{"data": json.dumps(chunk)}]

    def stream_event(self, event: StreamEvent) -> list[dict[str, Any]]:
        delta: dict[str, Any] = {}
        if event.reasoning_content:
            delta["reasoning_content"] = event.reasoning_content
        if event.content:
            delta["content"] = event.content
        if not delta:
            return []
        chunk = self._chunk(delta=delta)
        return [{"data": json.dumps(chunk)}]

    def stream_end(
        self,
        finish_reason: str,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        output_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        delta: dict[str, Any] = {}
        if tool_calls:
            delta["tool_calls"] = [
                {
                    "index": i,
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": tc["function"],
                }
                for i, tc in enumerate(tool_calls)
            ]
        chunk = self._chunk(delta=delta, finish_reason=finish_reason)
        return [
            {"data": json.dumps(chunk)},
            {"data": "[DONE]"},
        ]

    # ── non-streaming ────────────────────────────────────────────────

    def format_complete(
        self,
        result: TextResult,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": result.content,
        }
        if result.tool_calls:
            message["tool_calls"] = result.tool_calls
        if result.reasoning_content:
            message["reasoning_content"] = result.reasoning_content

        return {
            "id": self.request_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # ── helpers ───────────────────────────────────────────────────────

    def _chunk(
        self,
        delta: dict[str, Any],
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
