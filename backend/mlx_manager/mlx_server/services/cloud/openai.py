"""OpenAI cloud backend client."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from mlx_manager.mlx_server.services.cloud.client import CloudBackendClient

logger = logging.getLogger(__name__)

# Default OpenAI API URL
OPENAI_API_URL = "https://api.openai.com"


class OpenAICloudBackend(CloudBackendClient):
    """OpenAI cloud backend for fallback routing.

    Sends requests to OpenAI API in OpenAI format (no translation needed).
    Supports both streaming and non-streaming chat completions.
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

    def _build_headers(self) -> dict[str, str]:
        """Build OpenAI-specific headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
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
        """Send chat completion request to OpenAI API.

        Args:
            messages: List of message dicts with role and content
            model: Model ID (e.g., "gpt-4", "gpt-3.5-turbo")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, return async generator of chunks
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Non-streaming: Complete response dict
            Streaming: Async generator yielding chunk dicts
        """
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        logger.info(f"OpenAI request: model={model}, stream={stream}")

        if stream:
            return self._stream_chat_completion(request_data)
        else:
            return await self._complete_chat_completion(request_data)

    async def _complete_chat_completion(
        self,
        request_data: dict[str, Any],
    ) -> dict:
        """Non-streaming chat completion."""
        response = await self._post_with_circuit_breaker(
            "/v1/chat/completions",
            request_data,
        )
        return response.json()

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
