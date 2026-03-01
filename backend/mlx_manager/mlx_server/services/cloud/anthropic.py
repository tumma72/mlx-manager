"""Anthropic cloud backend client with format translation."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from loguru import logger

from mlx_manager.mlx_server.models.ir import (
    InferenceResult,
    InternalRequest,
    RoutingOutcome,
    StreamEvent,
    TextResult,
)
from mlx_manager.mlx_server.services.cloud.client import CloudBackendClient
from mlx_manager.mlx_server.services.formatters import anthropic_stop_to_openai
from mlx_manager.models.enums import ApiType

# Default Anthropic API URL
ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicCloudBackend(CloudBackendClient):
    """Anthropic cloud backend with protocol-aware routing.

    Same-protocol (Anthropic->Anthropic): passthrough original request directly.
    Cross-protocol (OpenAI->Anthropic): translate IR to Anthropic format,
    parse response back to IR for the calling endpoint to format.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = ANTHROPIC_API_URL,
        anthropic_version: str = ANTHROPIC_VERSION,
        **kwargs: Any,
    ):
        """Initialize Anthropic cloud backend.

        Args:
            api_key: Anthropic API key
            base_url: API base URL
            anthropic_version: Anthropic API version header
            **kwargs: Additional args for CloudBackendClient
        """
        self._anthropic_version = anthropic_version
        super().__init__(base_url=base_url, api_key=api_key, **kwargs)

    @property
    def protocol(self) -> ApiType:
        return ApiType.ANTHROPIC

    def _build_headers(self) -> dict[str, str]:
        """Build Anthropic-specific headers."""
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self._anthropic_version,
            "Content-Type": "application/json",
        }

    async def forward_request(self, ir: InternalRequest) -> RoutingOutcome:
        """Forward IR request to Anthropic cloud.

        Same-protocol (Anthropic->Anthropic): passthrough original request directly.
        Cross-protocol (OpenAI->Anthropic): translate IR, parse response to IR.
        """
        # Build Anthropic-format request
        if ir.original_protocol == ApiType.ANTHROPIC and ir.original_request is not None:
            # Same-protocol passthrough: use original request with model override
            request_data = ir.original_request.model_dump(exclude_none=True)
            request_data["model"] = ir.model  # Router may have overridden model
        else:
            # Cross-protocol: translate from IR to Anthropic format
            request_data = self._translate_request(
                ir.messages,
                ir.model,
                ir.params.max_tokens or 4096,
                ir.params.temperature or 1.0,
                ir.stream,
            )

        request_data["stream"] = ir.stream
        logger.info(f"Anthropic forward: model={ir.model}, stream={ir.stream}")

        if ir.stream:
            if ir.original_protocol == ApiType.ANTHROPIC:
                # Same-protocol: return raw Anthropic SSE stream
                stream = self._stream_anthropic_native(request_data)
                return RoutingOutcome(raw_stream=stream)
            # Cross-protocol: convert Anthropic stream to IR events
            stream_ir = self._stream_to_ir(request_data)
            return RoutingOutcome(ir_stream=stream_ir)
        else:
            response = await self._post_with_circuit_breaker("/v1/messages", request_data)
            anthropic_response = response.json()
            if ir.original_protocol == ApiType.ANTHROPIC:
                # Same-protocol: return raw Anthropic response
                return RoutingOutcome(raw_response=anthropic_response)
            # Cross-protocol: parse to IR
            return RoutingOutcome(ir_result=self._parse_response_to_ir(anthropic_response))

    def _translate_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Translate OpenAI-format request to Anthropic format."""
        # Extract system message (Anthropic has separate field)
        system_content: str | None = None
        anthropic_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                anthropic_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": temperature,
            "stream": stream,
        }

        if system_content:
            request["system"] = system_content

        # Map OpenAI stop to Anthropic stop_sequences
        if "stop" in kwargs and kwargs["stop"]:
            stop = kwargs["stop"]
            if isinstance(stop, str):
                request["stop_sequences"] = [stop]
            else:
                request["stop_sequences"] = stop

        return request

    def _parse_response_to_ir(self, anthropic_response: dict[str, Any]) -> InferenceResult:
        """Parse Anthropic response to IR InferenceResult."""
        content_blocks = anthropic_response.get("content", [])
        content_text = " ".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        stop_reason = anthropic_response.get("stop_reason", "end_turn")
        openai_stop = anthropic_stop_to_openai(stop_reason)
        usage = anthropic_response.get("usage", {})
        return InferenceResult(
            result=TextResult(
                content=content_text,
                finish_reason=openai_stop,
            ),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        )

    async def _stream_anthropic_native(
        self,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream Anthropic SSE events as dicts for EventSourceResponse."""
        current_event_type: str | None = None
        async for line in self._stream_with_circuit_breaker("/v1/messages", request_data):
            if not line or line.strip() == "":
                continue
            if line.startswith("event:"):
                current_event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str:
                    event: dict[str, Any] = {"data": data_str}
                    if current_event_type:
                        event["event"] = current_event_type
                        current_event_type = None
                    yield event

    async def _stream_to_ir(
        self,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[StreamEvent | TextResult, None]:
        """Convert Anthropic stream to IR events for cross-protocol routing."""
        async for line in self._stream_with_circuit_breaker("/v1/messages", request_data):
            if not line or line.strip() == "":
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")
            if event_type == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield StreamEvent(content=text)
            elif event_type == "message_delta":
                stop_reason = data.get("delta", {}).get("stop_reason")
                if stop_reason:
                    openai_stop = anthropic_stop_to_openai(stop_reason)
                    yield TextResult(content="", finish_reason=openai_stop)
            elif event_type == "message_stop":
                break


def create_anthropic_backend(
    api_key: str,
    base_url: str | None = None,
    **kwargs: Any,
) -> AnthropicCloudBackend:
    """Create Anthropic cloud backend instance.

    Args:
        api_key: Anthropic API key
        base_url: Optional custom API base URL
        **kwargs: Additional arguments for CloudBackendClient

    Returns:
        Configured AnthropicCloudBackend instance
    """
    return AnthropicCloudBackend(
        api_key=api_key,
        base_url=base_url or ANTHROPIC_API_URL,
        **kwargs,
    )
