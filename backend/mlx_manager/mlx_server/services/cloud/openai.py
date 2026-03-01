"""OpenAI cloud backend client."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any, cast

from loguru import logger

from mlx_manager.mlx_server.models.ir import (
    InferenceResult,
    InternalRequest,
    RoutingOutcome,
    StreamEvent,
    TextResult,
)
from mlx_manager.mlx_server.services.cloud.client import CloudBackendClient
from mlx_manager.models.enums import ApiType

# Default OpenAI API URL
OPENAI_API_URL = "https://api.openai.com"


class OpenAICloudBackend(CloudBackendClient):
    """OpenAI cloud backend for fallback routing.

    Supports same-protocol passthrough (OpenAI endpoint -> OpenAI cloud)
    and cross-protocol IR conversion (Anthropic endpoint -> OpenAI cloud).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENAI_API_URL,
        **kwargs: Any,
    ):
        """Initialize OpenAI cloud backend.

        Args:
            api_key: OpenAI API key
            base_url: API base URL (default: https://api.openai.com)
            **kwargs: Additional args passed to CloudBackendClient
        """
        super().__init__(base_url=base_url, api_key=api_key, **kwargs)

    @property
    def protocol(self) -> ApiType:
        return ApiType.OPENAI

    def _build_headers(self) -> dict[str, str]:
        """Build OpenAI-specific headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def forward_request(self, ir: InternalRequest) -> RoutingOutcome:
        """Forward IR request to OpenAI cloud.

        Same-protocol (OpenAI->OpenAI): passthrough original request directly.
        Cross-protocol (Anthropic->OpenAI): build request from IR, parse response to IR.
        """
        # Build OpenAI-format request
        if ir.original_protocol == ApiType.OPENAI and ir.original_request is not None:
            # Same-protocol passthrough: use original request with model override
            request_data = ir.original_request.model_dump(exclude_none=True)
            request_data["model"] = ir.model  # Router may have overridden model
        else:
            # Cross-protocol: build from IR fields
            request_data: dict[str, Any] = {
                "model": ir.model,
                "messages": ir.messages,
                "max_tokens": ir.params.max_tokens or 4096,
                "temperature": ir.params.temperature or 1.0,
            }
            if ir.stop:
                request_data["stop"] = ir.stop
            if ir.tools:
                request_data["tools"] = ir.tools

        request_data["stream"] = ir.stream
        logger.info(f"OpenAI forward: model={ir.model}, stream={ir.stream}")

        if ir.stream:
            stream = self._stream_chat_completion(request_data)
            if ir.original_protocol == ApiType.OPENAI:
                return RoutingOutcome(raw_stream=stream)
            # Cross-protocol: convert OpenAI stream to IR events
            return RoutingOutcome(ir_stream=self._stream_to_ir(stream))
        else:
            result = await self._complete_chat_completion(request_data)
            if ir.original_protocol == ApiType.OPENAI:
                return RoutingOutcome(raw_response=result)
            # Cross-protocol: parse OpenAI response to IR
            return RoutingOutcome(ir_result=self._parse_response_to_ir(result))

    async def _complete_chat_completion(
        self,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Non-streaming chat completion."""
        response = await self._post_with_circuit_breaker(
            "/v1/chat/completions",
            request_data,
        )
        return cast(dict[str, Any], response.json())

    async def _stream_chat_completion(
        self,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[dict, None]:
        """Streaming chat completion."""
        async for line in self._stream_with_circuit_breaker(
            "/v1/chat/completions",
            request_data,
        ):
            # Skip empty lines and [DONE] marker
            if not line or line.strip() == "":
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse SSE data: {data}")
                    continue

    @staticmethod
    def _parse_response_to_ir(response: dict[str, Any]) -> InferenceResult:
        """Parse OpenAI response dict to IR InferenceResult."""
        choice = response["choices"][0]
        msg = choice["message"]
        usage = response.get("usage", {})
        return InferenceResult(
            result=TextResult(
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
                reasoning_content=msg.get("reasoning_content"),
                finish_reason=choice.get("finish_reason", "stop"),
            ),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    @staticmethod
    async def _stream_to_ir(
        stream: AsyncGenerator[dict, None],
    ) -> AsyncGenerator[StreamEvent | TextResult, None]:
        """Convert OpenAI streaming chunks to IR events."""
        async for chunk in stream:
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            if finish_reason:
                yield TextResult(
                    content="",
                    finish_reason=finish_reason,
                )
            elif delta.get("content"):
                yield StreamEvent(content=delta["content"])
            elif delta.get("reasoning_content"):
                yield StreamEvent(reasoning_content=delta["reasoning_content"])


# Factory function for easy creation
def create_openai_backend(
    api_key: str,
    base_url: str | None = None,
    **kwargs: Any,
) -> OpenAICloudBackend:
    """Create OpenAI cloud backend instance.

    Args:
        api_key: OpenAI API key
        base_url: Optional override for API URL (e.g., Azure OpenAI)
        **kwargs: Additional configuration

    Returns:
        Configured OpenAICloudBackend instance
    """
    return OpenAICloudBackend(
        api_key=api_key,
        base_url=base_url or OPENAI_API_URL,
        **kwargs,
    )
