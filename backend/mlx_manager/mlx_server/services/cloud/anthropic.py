"""Anthropic cloud backend client with format translation."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from loguru import logger

from mlx_manager.mlx_server.services.cloud.client import CloudBackendClient
from mlx_manager.mlx_server.services.formatters import anthropic_stop_to_openai

# Default Anthropic API URL
ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicCloudBackend(CloudBackendClient):
    """Anthropic cloud backend with automatic format translation.

    Accepts OpenAI-format requests, translates to Anthropic format,
    sends to Anthropic API, and translates responses back to OpenAI format.
    This enables transparent fallback from local to Anthropic cloud.
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

    def _build_headers(self) -> dict[str, str]:
        """Build Anthropic-specific headers."""
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self._anthropic_version,
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[dict, None] | dict:
        """Send chat completion with format translation.

        Accepts OpenAI-format input, translates to Anthropic Messages format,
        and translates response back to OpenAI format.

        Args:
            messages: OpenAI-format messages [{"role": str, "content": str}]
            model: Model ID (e.g., "claude-3-opus-20240229")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, return async generator
            **kwargs: Additional parameters

        Returns:
            OpenAI-format response (dict or async generator)
        """
        # Translate OpenAI messages to Anthropic format
        anthropic_request = self._translate_request(
            messages, model, max_tokens, temperature, stream, **kwargs
        )

        logger.info(f"Anthropic request: model={model}, stream={stream}")

        if stream:
            return self._stream_with_translation(anthropic_request)
        else:
            return await self._complete_with_translation(anthropic_request)

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

    async def _complete_with_translation(
        self,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Non-streaming with response translation."""
        response = await self._post_with_circuit_breaker(
            "/v1/messages",
            request_data,
        )
        anthropic_response = response.json()
        return self._translate_response(anthropic_response)

    def _translate_response(self, anthropic_response: dict[str, Any]) -> dict[str, Any]:
        """Translate Anthropic response to OpenAI format."""
        # Extract content text
        content_blocks = anthropic_response.get("content", [])
        content_text = " ".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )

        # Translate stop reason
        anthropic_stop = anthropic_response.get("stop_reason")
        openai_stop = anthropic_stop_to_openai(anthropic_stop)

        # Build OpenAI-format response
        usage = anthropic_response.get("usage", {})
        return {
            "id": anthropic_response.get("id", "").replace("msg_", "chatcmpl-"),
            "object": "chat.completion",
            "created": 0,  # Anthropic doesn't include timestamp
            "model": anthropic_response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content_text,
                    },
                    "finish_reason": openai_stop,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        }

    async def _stream_with_translation(
        self,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming with response translation to OpenAI format."""
        async for line in self._stream_with_circuit_breaker(
            "/v1/messages",
            request_data,
        ):
            if not line or line.strip() == "":
                continue

            # Parse Anthropic SSE format: "event: type\ndata: {...}"
            if line.startswith("event:"):
                # Event type line, skip (we get type from data.type)
                continue

            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str:
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type", "")

                # Handle content_block_delta events
                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        token_text = delta.get("text", "")
                        if token_text:
                            yield {
                                "id": "chatcmpl-streaming",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": request_data.get("model", ""),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": token_text},
                                        "finish_reason": None,
                                    }
                                ],
                            }

                # Handle message_delta for finish reason
                elif event_type == "message_delta":
                    stop_reason = data.get("delta", {}).get("stop_reason")
                    if stop_reason:
                        openai_stop = anthropic_stop_to_openai(stop_reason)
                        yield {
                            "id": "chatcmpl-streaming",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": request_data.get("model", ""),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": openai_stop,
                                }
                            ],
                        }

                # Handle message_stop
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
