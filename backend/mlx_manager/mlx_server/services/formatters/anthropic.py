"""Anthropic Messages protocol formatter.

Converts IR types (StreamEvent, TextResult) into Anthropic Messages API
format, including the SSE event sequence for streaming.

Absorbs the output formatting logic from ProtocolTranslator and the
streaming event generation from messages.py.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from mlx_manager.mlx_server.models.ir import InternalRequest, StreamEvent, TextResult
from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from mlx_manager.mlx_server.services.formatters.base import ProtocolFormatter
from mlx_manager.models.enums import ApiType
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
        messages: list[dict[str, Any]] = []
        images: list[str] = []

        # Handle system prompt (Anthropic has separate field)
        if request.system:
            if isinstance(request.system, str):
                system_text = request.system
            else:
                # List of TextBlockParam
                system_text = " ".join(b.text for b in request.system)
            messages.append({"role": "system", "content": system_text})

        # Convert tools from Anthropic to OpenAI format
        tools = None
        if request.tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]

        # Convert content blocks to internal format
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                text_parts = []
                for block in msg.content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "image":
                            # Convert Anthropic image to data URL
                            data_url = f"data:{block.source.media_type};base64,{block.source.data}"
                            images.append(data_url)
                        elif block.type == "tool_use":
                            # Tool use from assistant - convert to OpenAI format
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": block.id,
                                            "type": "function",
                                            "function": {
                                                "name": block.name,
                                                "arguments": json.dumps(block.input),
                                            },
                                        }
                                    ],
                                }
                            )
                            continue
                        elif block.type == "tool_result":
                            # Tool result from user - convert to OpenAI format
                            if isinstance(block.content, str):
                                result_content = block.content
                            else:
                                result_content = " ".join(
                                    b.text for b in block.content if hasattr(b, "text")
                                )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.tool_use_id,
                                    "content": result_content,
                                }
                            )
                            continue
                    elif isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "image":
                            source = block.get("source", {})
                            media_type = source.get("media_type", "image/png")
                            data = source.get("data", "")
                            data_url = f"data:{media_type};base64,{data}"
                            images.append(data_url)
                        elif block.get("type") == "tool_use":
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": block["id"],
                                            "type": "function",
                                            "function": {
                                                "name": block["name"],
                                                "arguments": json.dumps(block.get("input", {})),
                                            },
                                        }
                                    ],
                                }
                            )
                            continue
                        elif block.get("type") == "tool_result":
                            rc = block.get("content", "")
                            if isinstance(rc, list):
                                rc = " ".join(
                                    b.get("text", "") for b in rc if b.get("type") == "text"
                                )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block["tool_use_id"],
                                    "content": rc,
                                }
                            )
                            continue
                if text_parts:
                    messages.append({"role": msg.role, "content": " ".join(text_parts)})

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
            tools=tools,
            images=images if images else None,
            original_request=request,
            original_protocol=ApiType.ANTHROPIC,
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
        events: list[dict[str, Any]] = []

        # Close the text content block
        events.append(
            {
                "event": "content_block_stop",
                "data": json.dumps(
                    {
                        "type": "content_block_stop",
                        "index": 0,
                    }
                ),
            }
        )

        # Emit tool_use content blocks if present
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                block_index = i + 1  # text block is index 0
                func = tc.get("function", {})
                tool_id = tc.get("id", f"toolu_{i}")
                tool_name = func.get("name", "")
                try:
                    tool_input = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}

                # content_block_start for tool_use (empty input per Anthropic spec)
                events.append(
                    {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": tool_name,
                                    "input": {},
                                },
                            }
                        ),
                    }
                )
                # content_block_delta with input_json_delta
                events.append(
                    {
                        "event": "content_block_delta",
                        "data": json.dumps(
                            {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(tool_input),
                                },
                            }
                        ),
                    }
                )
                # content_block_stop for tool_use
                events.append(
                    {
                        "event": "content_block_stop",
                        "data": json.dumps(
                            {
                                "type": "content_block_stop",
                                "index": block_index,
                            }
                        ),
                    }
                )

        events.append(
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
            }
        )
        events.append(
            {
                "event": "message_stop",
                "data": json.dumps({"type": "message_stop"}),
            }
        )

        return events

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

        content: list[TextBlock | ToolUseBlock] = []
        if result.content:
            content.append(TextBlock(text=result.content))

        if result.tool_calls:
            anthropic_stop = "tool_use"  # type: ignore[assignment]
            for tc in result.tool_calls:
                func = tc.get("function", {})
                tool_id = tc.get("id", "toolu_0")
                tool_name = func.get("name", "")
                try:
                    tool_input = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}
                content.append(ToolUseBlock(id=tool_id, name=tool_name, input=tool_input))

        if not content:
            content.append(TextBlock(text=""))

        return AnthropicMessagesResponse(
            id=self.request_id,
            model=self.model_id,
            content=content,
            stop_reason=anthropic_stop,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            ),
        )
