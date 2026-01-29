"""Tests for CloudBackendClient base class."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mlx_manager.mlx_server.services.cloud.client import (
    AsyncCircuitBreaker,
    CircuitBreakerError,
    CloudBackendClient,
)


class ConcreteCloudClient(CloudBackendClient):
    """Concrete implementation for testing the abstract base class."""

    def _build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[dict, None] | dict:
        # Mock implementation for testing
        return {"choices": [{"message": {"content": "test"}}]}


class TestAsyncCircuitBreaker:
    """Tests for the AsyncCircuitBreaker class."""

    def test_initial_state_closed(self) -> None:
        """Circuit breaker starts in closed state."""
        cb = AsyncCircuitBreaker(fail_max=3)
        assert cb.current_state == "closed"
        assert cb.fail_counter == 0

    def test_opens_after_fail_max_failures(self) -> None:
        """Circuit opens after configured number of failures."""
        cb = AsyncCircuitBreaker(fail_max=3)

        cb.failure()
        assert cb.current_state == "closed"

        cb.failure()
        assert cb.current_state == "closed"

        cb.failure()
        assert cb.current_state == "open"

    def test_success_resets_counter(self) -> None:
        """Success call resets failure counter."""
        cb = AsyncCircuitBreaker(fail_max=3)

        cb.failure()
        cb.failure()
        assert cb.fail_counter == 2

        cb.success()
        assert cb.fail_counter == 0
        assert cb.current_state == "closed"

    def test_success_closes_open_circuit(self) -> None:
        """Success call closes an open circuit."""
        cb = AsyncCircuitBreaker(fail_max=2)

        cb.failure()
        cb.failure()
        assert cb.current_state == "open"

        cb.success()
        assert cb.current_state == "closed"

    def test_check_raises_when_open(self) -> None:
        """check() raises CircuitBreakerError when circuit is open."""
        cb = AsyncCircuitBreaker(fail_max=2)

        cb.failure()
        cb.failure()

        with pytest.raises(CircuitBreakerError, match="Circuit breaker is open"):
            cb.check()

    def test_check_passes_when_closed(self) -> None:
        """check() passes when circuit is closed."""
        cb = AsyncCircuitBreaker(fail_max=3)

        # Should not raise
        cb.check()

    def test_transitions_to_half_open_after_timeout(self) -> None:
        """Circuit transitions to half-open after reset_timeout."""
        cb = AsyncCircuitBreaker(fail_max=2, reset_timeout=0.01)

        cb.failure()
        cb.failure()
        assert cb.current_state == "open"

        # Wait for reset timeout
        import time

        time.sleep(0.02)

        assert cb.current_state == "half-open"

    def test_half_open_allows_check(self) -> None:
        """check() passes when circuit is half-open."""
        cb = AsyncCircuitBreaker(fail_max=2, reset_timeout=0.01)

        cb.failure()
        cb.failure()

        import time

        time.sleep(0.02)

        # Should not raise in half-open state
        cb.check()


class TestCloudBackendClientInitialization:
    """Tests for client initialization."""

    def test_client_created_with_base_url_and_api_key(self) -> None:
        """Client stores base_url and api_key."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key-123",
        )

        assert client.base_url == "https://api.example.com"
        assert client._api_key == "test-key-123"

    def test_circuit_breaker_initialized_closed(self) -> None:
        """Circuit breaker starts in closed state."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
        )

        assert client._circuit_breaker.current_state == "closed"
        assert client.is_circuit_open is False

    def test_headers_built_correctly(self) -> None:
        """Headers include authorization from subclass."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key-123",
        )

        # Access headers through the underlying httpx client
        assert client._client.headers["Authorization"] == "Bearer test-key-123"

    def test_custom_retry_parameters(self) -> None:
        """Client accepts custom retry configuration."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
            max_retries=5,
            backoff_factor=1.0,
            timeout=120.0,
        )

        # Verify timeout is set (we can check the client's timeout)
        assert client._client.timeout.read == 120.0
        assert client._client.timeout.connect == 10.0

    def test_custom_circuit_breaker_parameters(self) -> None:
        """Client accepts custom circuit breaker configuration."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
            circuit_breaker_fail_max=3,
            circuit_breaker_reset_timeout=60,
        )

        assert client._circuit_breaker.fail_max == 3
        assert client._circuit_breaker.reset_timeout == 60


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker behavior in CloudBackendClient."""

    def test_circuit_initially_closed(self) -> None:
        """Circuit breaker starts closed."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
        )

        assert client.is_circuit_open is False
        assert client._circuit_breaker.current_state == "closed"

    def test_circuit_opens_after_fail_max_failures(self) -> None:
        """Circuit opens after configured number of failures."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
            circuit_breaker_fail_max=3,
        )

        # Simulate failures
        for _ in range(3):
            client._circuit_breaker.failure()

        assert client.is_circuit_open is True
        assert client._circuit_breaker.current_state == "open"

    def test_is_circuit_open_reflects_state(self) -> None:
        """is_circuit_open property reflects circuit breaker state."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
            circuit_breaker_fail_max=2,
        )

        assert client.is_circuit_open is False

        client._circuit_breaker.failure()
        assert client.is_circuit_open is False

        client._circuit_breaker.failure()
        assert client.is_circuit_open is True

    def test_success_resets_failure_count(self) -> None:
        """Success call resets failure count."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
            circuit_breaker_fail_max=3,
        )

        # Add some failures but not enough to open
        client._circuit_breaker.failure()
        client._circuit_breaker.failure()

        # Success resets the count
        client._circuit_breaker.success()

        # Now need 3 more failures to open
        client._circuit_breaker.failure()
        client._circuit_breaker.failure()
        assert client.is_circuit_open is False

        client._circuit_breaker.failure()
        assert client.is_circuit_open is True


class TestPostWithCircuitBreaker:
    """Tests for _post_with_circuit_breaker method."""

    @pytest.fixture
    def client(self) -> ConcreteCloudClient:
        """Create a test client."""
        return ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
            circuit_breaker_fail_max=3,
        )

    async def test_raises_circuit_breaker_error_when_open(
        self, client: ConcreteCloudClient
    ) -> None:
        """Raises CircuitBreakerError when circuit is open."""
        # Force circuit open
        for _ in range(3):
            client._circuit_breaker.failure()

        with pytest.raises(CircuitBreakerError, match="Circuit breaker is open"):
            await client._post_with_circuit_breaker("/test", {"data": "value"})

    async def test_records_success_on_successful_response(
        self, client: ConcreteCloudClient
    ) -> None:
        """Records success for circuit breaker on 2xx response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            # Add a failure first
            client._circuit_breaker.failure()
            assert client._circuit_breaker.fail_counter == 1

            # Successful call should reset
            await client._post_with_circuit_breaker("/test", {"data": "value"})

            # Counter resets after success
            assert client._circuit_breaker.fail_counter == 0

    async def test_records_failure_on_http_error(
        self, client: ConcreteCloudClient
    ) -> None:
        """Records failure for circuit breaker on HTTP error."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500

        http_error = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )
        mock_response.raise_for_status.side_effect = http_error

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(httpx.HTTPStatusError):
                await client._post_with_circuit_breaker("/test", {"data": "value"})

            assert client._circuit_breaker.fail_counter == 1

    async def test_records_failure_on_connection_error(
        self, client: ConcreteCloudClient
    ) -> None:
        """Records failure for circuit breaker on connection error."""
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(httpx.ConnectError):
                await client._post_with_circuit_breaker("/test", {"data": "value"})

            assert client._circuit_breaker.fail_counter == 1


class TestStreamWithCircuitBreaker:
    """Tests for _stream_with_circuit_breaker method."""

    @pytest.fixture
    def client(self) -> ConcreteCloudClient:
        """Create a test client."""
        return ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
            circuit_breaker_fail_max=3,
        )

    async def test_raises_circuit_breaker_error_when_open(
        self, client: ConcreteCloudClient
    ) -> None:
        """Raises CircuitBreakerError when circuit is open."""
        # Force circuit open
        for _ in range(3):
            client._circuit_breaker.failure()

        with pytest.raises(CircuitBreakerError, match="Circuit breaker is open"):
            async for _ in client._stream_with_circuit_breaker("/test", {"data": "value"}):
                pass


class TestClose:
    """Tests for close method."""

    async def test_client_can_be_closed(self) -> None:
        """Client close method calls underlying client aclose."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
        )

        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_called_once()

    async def test_close_without_error(self) -> None:
        """Client can be closed without raising errors."""
        client = ConcreteCloudClient(
            base_url="https://api.example.com",
            api_key="test-key",
        )

        # Should not raise
        await client.close()


class TestAbstractMethods:
    """Tests that abstract methods must be implemented."""

    def test_cannot_instantiate_base_class(self) -> None:
        """Cannot instantiate CloudBackendClient directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            CloudBackendClient(  # type: ignore[abstract]
                base_url="https://api.example.com",
                api_key="test-key",
            )

    def test_abstract_methods_defined(self) -> None:
        """Abstract methods are properly defined."""
        assert "_build_headers" in CloudBackendClient.__abstractmethods__
        assert "chat_completion" in CloudBackendClient.__abstractmethods__
