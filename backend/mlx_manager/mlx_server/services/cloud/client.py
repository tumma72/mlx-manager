"""Base cloud backend client with retry and circuit breaker support."""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from httpx_retries import Retry, RetryTransport
from loguru import logger


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


class AsyncCircuitBreaker:
    """Simple async-compatible circuit breaker.

    States:
    - closed: Normal operation, requests pass through
    - open: Requests fail immediately, circuit is tripped
    - half-open: Allow one request to test if backend recovered

    The circuit opens after `fail_max` consecutive failures and
    automatically attempts to close after `reset_timeout` seconds.
    """

    def __init__(self, fail_max: int = 5, reset_timeout: float = 30.0):
        """Initialize circuit breaker.

        Args:
            fail_max: Number of failures before opening circuit
            reset_timeout: Seconds to wait before allowing a test request
        """
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout

        self._fail_counter = 0
        self._state = "closed"
        self._opened_at: float | None = None

    @property
    def current_state(self) -> str:
        """Get current circuit state, checking for half-open transition."""
        if self._state == "open" and self._opened_at is not None:
            if time.time() - self._opened_at >= self.reset_timeout:
                self._state = "half-open"
        return self._state

    @property
    def fail_counter(self) -> int:
        """Get current failure count."""
        return self._fail_counter

    def success(self) -> None:
        """Record a successful call - resets failure counter and closes circuit."""
        self._fail_counter = 0
        self._state = "closed"
        self._opened_at = None

    def failure(self) -> None:
        """Record a failed call - may open circuit if threshold reached."""
        self._fail_counter += 1
        if self._fail_counter >= self.fail_max:
            self._state = "open"
            self._opened_at = time.time()

    def check(self) -> None:
        """Check if requests should be allowed.

        Raises:
            CircuitBreakerError: If circuit is open
        """
        state = self.current_state  # Triggers half-open check

        if state == "open":
            raise CircuitBreakerError("Circuit breaker is open")

        # half-open allows one request through (for testing)
        # closed allows all requests


# Re-export for consumers
__all__ = ["CloudBackendClient", "CircuitBreakerError", "AsyncCircuitBreaker"]


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
        self._circuit_breaker = AsyncCircuitBreaker(
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
        # Check circuit breaker before request
        self._circuit_breaker.check()

        try:
            response = await self._client.post(endpoint, json=json_data)
            response.raise_for_status()

            # Success - record for circuit breaker
            self._circuit_breaker.success()
            return response

        except httpx.HTTPStatusError as e:
            self._circuit_breaker.failure()
            logger.warning(f"HTTP error from {self.base_url}: {e.response.status_code}")
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
        # Check circuit breaker before request
        self._circuit_breaker.check()

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
