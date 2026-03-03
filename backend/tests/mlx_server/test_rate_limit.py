"""Tests for per-IP rate limiting middleware."""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_manager.mlx_server.middleware.rate_limit import RateLimitMiddleware, TokenBucket

# ---------------------------------------------------------------------------
# TokenBucket unit tests
# ---------------------------------------------------------------------------


class TestTokenBucketAllowsWithinLimit:
    """TokenBucket allows requests within the configured capacity."""

    def test_consume_within_capacity(self) -> None:
        """Consuming tokens within capacity succeeds."""
        bucket = TokenBucket(rate=1.0, capacity=5)
        for _ in range(5):
            assert bucket.consume() is True

    def test_full_bucket_has_correct_token_count(self) -> None:
        """A new bucket starts at full capacity."""
        bucket = TokenBucket(rate=10.0, capacity=10)
        assert bucket.tokens == 10.0


class TestTokenBucketBlocksExceedingLimit:
    """TokenBucket rejects requests once tokens are exhausted."""

    def test_consume_beyond_capacity_fails(self) -> None:
        """Consuming more tokens than capacity returns False."""
        bucket = TokenBucket(rate=1.0, capacity=3)
        for _ in range(3):
            bucket.consume()
        assert bucket.consume() is False

    def test_empty_bucket_stays_empty_without_time(self) -> None:
        """Without elapsed time, tokens do not refill."""
        bucket = TokenBucket(rate=1.0, capacity=1)
        bucket.consume()  # drain
        # No time passes, last_refill is just set
        assert bucket.consume() is False


class TestTokenBucketRefill:
    """TokenBucket refills tokens over time at the configured rate."""

    def test_refill_after_elapsed_time(self) -> None:
        """Tokens refill based on elapsed time and rate."""
        bucket = TokenBucket(rate=10.0, capacity=10)
        # Drain all tokens
        for _ in range(10):
            bucket.consume()
        assert bucket.consume() is False

        # Simulate 1 second passing (should add 10 tokens at rate=10/s)
        bucket.last_refill = time.monotonic() - 1.0
        assert bucket.consume() is True

    def test_refill_does_not_exceed_capacity(self) -> None:
        """Tokens never exceed the bucket capacity after refill."""
        bucket = TokenBucket(rate=100.0, capacity=5)
        # Simulate a very long time passing
        bucket.last_refill = time.monotonic() - 1000.0
        bucket.consume()
        # Even after massive refill, tokens should be capped at capacity - 1
        assert bucket.tokens <= bucket.capacity


class TestTokenBucketRetryAfter:
    """TokenBucket.retry_after returns correct seconds until next token."""

    def test_retry_after_with_tokens_available(self) -> None:
        """retry_after is 0 when tokens are available."""
        bucket = TokenBucket(rate=1.0, capacity=5)
        assert bucket.retry_after == 0

    def test_retry_after_when_empty(self) -> None:
        """retry_after is positive when bucket is empty."""
        bucket = TokenBucket(rate=1.0, capacity=2)
        bucket.consume()
        bucket.consume()
        assert bucket.retry_after >= 1

    def test_retry_after_reflects_refill_rate(self) -> None:
        """Slower rates produce longer retry_after values."""
        # Very slow rate: 1 token per 60 seconds
        bucket = TokenBucket(rate=1.0 / 60.0, capacity=1)
        bucket.consume()
        # Should need ~60 seconds for next token
        assert bucket.retry_after >= 30


# ---------------------------------------------------------------------------
# Middleware integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _rate_limited_app() -> FastAPI:
    """Create test app with rate limiting enabled (5 RPM)."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, rpm=5)

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    return app


@pytest.fixture
def rate_limited_client(_rate_limited_app: FastAPI) -> TestClient:
    return TestClient(_rate_limited_app, raise_server_exceptions=False)


@pytest.fixture
def _disabled_app() -> FastAPI:
    """Create test app with rate limiting disabled (rpm=0)."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, rpm=0)

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    return app


@pytest.fixture
def disabled_client(_disabled_app: FastAPI) -> TestClient:
    return TestClient(_disabled_app, raise_server_exceptions=False)


class TestMiddlewareReturns429:
    """Middleware returns 429 with correct headers when rate is exceeded."""

    def test_returns_429_after_exceeding_limit(self, rate_limited_client: TestClient) -> None:
        """Requests beyond the limit receive 429 status."""
        # 5 RPM limit - first 5 should pass
        for _ in range(5):
            response = rate_limited_client.get("/ok")
            assert response.status_code == 200

        # 6th request should be rate limited
        response = rate_limited_client.get("/ok")
        assert response.status_code == 429

    def test_429_response_has_retry_after_header(self, rate_limited_client: TestClient) -> None:
        """Rate-limited responses include Retry-After header."""
        for _ in range(5):
            rate_limited_client.get("/ok")

        response = rate_limited_client.get("/ok")
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert int(response.headers["Retry-After"]) >= 1

    def test_429_response_has_rate_limit_headers(self, rate_limited_client: TestClient) -> None:
        """Rate-limited responses include X-RateLimit-* headers."""
        for _ in range(5):
            rate_limited_client.get("/ok")

        response = rate_limited_client.get("/ok")
        assert response.headers["X-RateLimit-Limit"] == "5"
        assert response.headers["X-RateLimit-Remaining"] == "0"

    def test_429_response_body_is_rfc7807(self, rate_limited_client: TestClient) -> None:
        """Rate-limited response body follows RFC 7807 Problem Details."""
        for _ in range(5):
            rate_limited_client.get("/ok")

        response = rate_limited_client.get("/ok")
        body = response.json()
        assert body["type"] == "https://mlx-manager.dev/errors/rate-limited"
        assert body["title"] == "Too Many Requests"
        assert body["status"] == 429
        assert "5 requests per minute" in body["detail"]


class TestMiddlewarePassthroughWhenDisabled:
    """Middleware passes through all requests when rpm=0 (disabled)."""

    def test_all_requests_pass_when_disabled(self, disabled_client: TestClient) -> None:
        """With rpm=0, no requests are rate limited."""
        for _ in range(100):
            response = disabled_client.get("/ok")
            assert response.status_code == 200

    def test_no_rate_limit_headers_when_disabled(self, disabled_client: TestClient) -> None:
        """With rpm=0, rate limit headers are not added."""
        response = disabled_client.get("/ok")
        assert "X-RateLimit-Limit" not in response.headers
        assert "X-RateLimit-Remaining" not in response.headers


class TestMiddlewareAddsRateLimitHeaders:
    """Middleware adds X-RateLimit-Limit and X-RateLimit-Remaining to responses."""

    def test_successful_response_has_limit_header(self, rate_limited_client: TestClient) -> None:
        """Successful responses include X-RateLimit-Limit header."""
        response = rate_limited_client.get("/ok")
        assert response.status_code == 200
        assert response.headers["X-RateLimit-Limit"] == "5"

    def test_successful_response_has_remaining_header(
        self, rate_limited_client: TestClient
    ) -> None:
        """Successful responses include X-RateLimit-Remaining header."""
        response = rate_limited_client.get("/ok")
        assert response.status_code == 200
        remaining = int(response.headers["X-RateLimit-Remaining"])
        assert remaining >= 0
        assert remaining <= 5

    def test_remaining_decreases_with_requests(self, rate_limited_client: TestClient) -> None:
        """X-RateLimit-Remaining decreases as requests are made."""
        first = rate_limited_client.get("/ok")
        second = rate_limited_client.get("/ok")
        first_remaining = int(first.headers["X-RateLimit-Remaining"])
        second_remaining = int(second.headers["X-RateLimit-Remaining"])
        assert second_remaining < first_remaining


class TestClientIPExtraction:
    """Middleware correctly extracts client IP from various sources."""

    def test_ip_from_x_forwarded_for(self, rate_limited_client: TestClient) -> None:
        """Client IP extracted from X-Forwarded-For header (first value)."""
        # Use a unique IP via X-Forwarded-For
        for _ in range(5):
            rate_limited_client.get("/ok", headers={"X-Forwarded-For": "10.0.0.1"})

        # 6th from same IP should be rate limited
        response = rate_limited_client.get("/ok", headers={"X-Forwarded-For": "10.0.0.1"})
        assert response.status_code == 429

        # Different IP should still be allowed
        response = rate_limited_client.get("/ok", headers={"X-Forwarded-For": "10.0.0.2"})
        assert response.status_code == 200

    def test_ip_from_x_forwarded_for_multiple_proxies(
        self, rate_limited_client: TestClient
    ) -> None:
        """X-Forwarded-For with multiple IPs uses the first (client) IP."""
        for _ in range(5):
            rate_limited_client.get(
                "/ok", headers={"X-Forwarded-For": "192.168.1.1, 10.0.0.1, 172.16.0.1"}
            )

        response = rate_limited_client.get(
            "/ok", headers={"X-Forwarded-For": "192.168.1.1, 10.0.0.1, 172.16.0.1"}
        )
        assert response.status_code == 429

    def test_ip_from_direct_connection(self) -> None:
        """Client IP extracted from request.client when no X-Forwarded-For."""
        middleware = RateLimitMiddleware(app=MagicMock(), rpm=60)
        request = MagicMock()
        request.headers = {}
        request.client.host = "127.0.0.1"
        assert middleware._get_client_ip(request) == "127.0.0.1"

    def test_ip_unknown_when_no_client(self) -> None:
        """Returns 'unknown' when request.client is None."""
        middleware = RateLimitMiddleware(app=MagicMock(), rpm=60)
        request = MagicMock()
        request.headers = {}
        request.client = None
        assert middleware._get_client_ip(request) == "unknown"


class TestStaleBucketCleanup:
    """Middleware cleans up stale buckets to prevent unbounded memory growth."""

    def test_stale_buckets_removed_after_cleanup(self) -> None:
        """Buckets inactive for >2 minutes are removed during cleanup."""
        middleware = RateLimitMiddleware(app=MagicMock(), rpm=60)

        # Create a bucket and make it stale
        bucket = middleware._get_bucket("10.0.0.1")
        bucket.last_refill = time.monotonic() - 300  # 5 minutes ago

        # Create a fresh bucket
        middleware._get_bucket("10.0.0.2")

        assert "10.0.0.1" in middleware._buckets
        assert "10.0.0.2" in middleware._buckets

        # Force cleanup by setting last_cleanup far in the past
        middleware._last_cleanup = time.monotonic() - 600
        middleware._maybe_cleanup()

        # Stale bucket should be removed, fresh one kept
        assert "10.0.0.1" not in middleware._buckets
        assert "10.0.0.2" in middleware._buckets

    def test_cleanup_skipped_when_interval_not_elapsed(self) -> None:
        """Cleanup does not run when less than cleanup_interval has passed."""
        middleware = RateLimitMiddleware(app=MagicMock(), rpm=60)

        # Create a stale bucket
        bucket = middleware._get_bucket("10.0.0.1")
        bucket.last_refill = time.monotonic() - 300

        # Don't force cleanup (interval not elapsed)
        middleware._maybe_cleanup()

        # Bucket should still be there
        assert "10.0.0.1" in middleware._buckets

    def test_cleanup_preserves_active_buckets(self) -> None:
        """Active buckets are not removed during cleanup."""
        middleware = RateLimitMiddleware(app=MagicMock(), rpm=60)

        # Create an active bucket (recent last_refill)
        middleware._get_bucket("10.0.0.1")

        # Force cleanup
        middleware._last_cleanup = time.monotonic() - 600
        middleware._maybe_cleanup()

        # Active bucket should be preserved
        assert "10.0.0.1" in middleware._buckets


class TestMiddlewarePerIPIsolation:
    """Different IPs have independent rate limits."""

    def test_separate_limits_per_ip(self, rate_limited_client: TestClient) -> None:
        """Each IP gets its own independent rate limit bucket."""
        # Exhaust limit for IP 10.0.0.1
        for _ in range(5):
            rate_limited_client.get("/ok", headers={"X-Forwarded-For": "10.0.0.1"})

        # 10.0.0.1 should be rate limited
        response = rate_limited_client.get("/ok", headers={"X-Forwarded-For": "10.0.0.1"})
        assert response.status_code == 429

        # 10.0.0.2 should still have full capacity
        for _ in range(5):
            response = rate_limited_client.get("/ok", headers={"X-Forwarded-For": "10.0.0.2"})
            assert response.status_code == 200


class TestConfigSetting:
    """rate_limit_rpm config setting is correctly defined."""

    def test_default_is_zero(self) -> None:
        """Default rate_limit_rpm is 0 (disabled)."""
        from mlx_manager.mlx_server.config import MLXServerSettings

        with patch.dict("os.environ", {}, clear=False):
            settings = MLXServerSettings()
            assert settings.rate_limit_rpm == 0

    def test_accepts_positive_value(self) -> None:
        """rate_limit_rpm accepts positive integer values."""
        from mlx_manager.mlx_server.config import MLXServerSettings

        with patch.dict("os.environ", {"MLX_SERVER_RATE_LIMIT_RPM": "120"}, clear=False):
            settings = MLXServerSettings()
            assert settings.rate_limit_rpm == 120
