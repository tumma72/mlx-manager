"""Base ProtocolFormatter ABC for the adapter pipeline.

ProtocolFormatter is Layer 3 of the 3-layer adapter pipeline:
  ModelAdapter → StreamProcessor → ProtocolFormatter

Each formatter instance is scoped to a single request and holds
request-level metadata (model_id, request_id, timestamps).

Formatters convert IR types into protocol-specific dicts ready for
EventSourceResponse (streaming) or direct JSON response (non-streaming).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult


class ProtocolFormatter(ABC):
    """Converts IR types into protocol-specific response formats.

    Instantiated per-request with request metadata. Methods produce
    dicts ready for EventSourceResponse (streaming) or FastAPI response.

    Streaming lifecycle:
        1. stream_start()  → initial envelope events
        2. stream_event()  → per-token content events (called N times)
        3. stream_end()    → closing events with finish_reason

    Non-streaming:
        format_complete()  → full response dict/Pydantic model
    """

    def __init__(self, model_id: str, request_id: str) -> None:
        self.model_id = model_id
        self.request_id = request_id
        self.created = int(time.time())

    @abstractmethod
    def stream_start(self) -> list[dict[str, Any]]:
        """Emit initial events before content streaming begins.

        Returns:
            List of SSE-ready dicts. For OpenAI: role chunk.
            For Anthropic: message_start + content_block_start.
        """
        ...

    @abstractmethod
    def stream_event(self, event: StreamEvent) -> list[dict[str, Any]]:
        """Format a single streaming event into protocol-specific chunks.

        Args:
            event: IR StreamEvent with content and/or reasoning_content.

        Returns:
            List of SSE-ready dicts (may be empty if event has no content).
        """
        ...

    @abstractmethod
    def stream_end(
        self,
        finish_reason: str,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        output_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        """Emit closing events after streaming ends.

        Args:
            finish_reason: Why generation stopped (stop, length, tool_calls).
            tool_calls: Optional tool calls detected in the stream.
            output_tokens: Number of tokens generated.

        Returns:
            List of SSE-ready dicts including [DONE] sentinel where applicable.
        """
        ...

    @abstractmethod
    def format_complete(
        self,
        result: TextResult,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> Any:
        """Format a complete non-streaming response.

        Args:
            result: IR TextResult with content, tool_calls, reasoning_content.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Protocol-specific response (dict or Pydantic model).
        """
        ...
