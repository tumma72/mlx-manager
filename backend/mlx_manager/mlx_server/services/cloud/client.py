"""Base cloud backend client with retry and circuit breaker support."""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from httpx_retries import Retry, RetryTransport
from pybreaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)

# Re-export for consumers
__all__ = ["CloudBackendClient", "CircuitBreakerError"]


class CloudBackendClient(ABC):
    """Base class for cloud backend clients with resilience patterns.

    Features:
    - Automatic retry with exponential backoff on transient failures
    - Circuit breaker to prevent cascade failures
    - Configurable timeouts
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout: float = 60.0,
        circuit_breaker_fail_max: int = 5,
        circuit_breaker_reset_timeout: int = 30,
    ):
        """Initialize cloud backend client.

        Args:
            base_url: API base URL
            api_key: API key for authentication
            max_retries: Maximum retry attempts (default: 3)
            backoff_factor: Exponential backoff factor (default: 0.5)
            timeout: Request timeout in seconds (default: 60)
            circuit_breaker_fail_max: Failures before circuit opens (default: 5)
            circuit_breaker_reset_timeout: Seconds before circuit resets (default: 30)
        """
        self.base_url = base_url
        self._api_key = api_key  # Keep private, don't log

        # Configure retry transport
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            # Retry on these status codes
            status_forcelist=[429, 500, 502, 503, 504],
        )

        # Create async client with retry transport
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=RetryTransport(retry=retry),
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers=self._build_headers(),
        )

        # Circuit breaker per client instance
        self._circuit_breaker = CircuitBreaker(
            fail_max=circuit_breaker_fail_max,
            reset_timeout=circuit_breaker_reset_timeout,
        )

        logger.info(f"Initialized cloud client for {base_url}")

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """Build request headers. Subclasses implement auth-specific headers."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[dict, None] | dict:
        """Send chat completion request. Subclasses implement API-specific format."""
        pass

    async def _post_with_circuit_breaker(
        self,
        endpoint: str,
        json_data: dict[str, Any],
    ) -> httpx.Response:
        """POST request with circuit breaker protection."""
        try:
            # Check circuit breaker state
            if self._circuit_breaker.current_state == "open":
                raise CircuitBreakerError("Circuit breaker is open")

            response = await self._client.post(endpoint, json=json_data)
            response.raise_for_status()

            # Success - record for circuit breaker
            self._circuit_breaker.success()
            return response

        except httpx.HTTPStatusError as e:
            self._circuit_breaker.failure()
            logger.warning(f"HTTP error from {self.base_url}: {e.response.status_code}")
            raise
        except CircuitBreakerError:
            # Don't record failure again for circuit breaker errors
            raise
        except Exception as e:
            self._circuit_breaker.failure()
            logger.warning(f"Request failed to {self.base_url}: {e}")
            raise

    async def _stream_with_circuit_breaker(
        self,
        endpoint: str,
        json_data: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Streaming POST with circuit breaker protection."""
        if self._circuit_breaker.current_state == "open":
            raise CircuitBreakerError("Circuit breaker is open")

        try:
            async with self._client.stream("POST", endpoint, json=json_data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line

            # Success after completing stream
            self._circuit_breaker.success()

        except Exception as e:
            self._circuit_breaker.failure()
            logger.warning(f"Stream failed to {self.base_url}: {e}")
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self._circuit_breaker.current_state == "open"
