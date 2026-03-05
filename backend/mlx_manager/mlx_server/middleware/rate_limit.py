"""Per-IP rate limiting middleware using token bucket algorithm.

Provides configurable requests-per-minute (RPM) rate limiting on a per-IP
basis. When rate_limit_rpm is 0 (the default), the middleware passes through
all requests with no overhead.

Uses a simple token bucket algorithm:
- Each IP gets a bucket with capacity = RPM
- Tokens refill at RPM/60 per second
- Each request consumes one token
- When empty, returns 429 with Retry-After header
"""

import time

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class TokenBucket:
    """Simple token bucket rate limiter.

    Attributes:
        rate: Tokens added per second.
        capacity: Maximum tokens the bucket can hold.
        tokens: Current token count (float for fractional refills).
        last_refill: Monotonic timestamp of last refill calculation.
    """

    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        """Try to consume a token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def retry_after(self) -> int:
        """Seconds until a token is available."""
        if self.tokens >= 1.0:
            return 0
        return max(1, int((1.0 - self.tokens) / self.rate) + 1)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting middleware using token bucket algorithm.

    When rpm <= 0, all requests pass through with zero overhead.
    Otherwise, each client IP is tracked independently and limited
    to rpm requests per minute using a token bucket.

    Stale buckets (inactive for 2+ minutes) are periodically cleaned
    up to prevent unbounded memory growth.

    Response headers on allowed requests:
        X-RateLimit-Limit: Configured RPM
        X-RateLimit-Remaining: Approximate tokens remaining

    Response on rate-limited requests (HTTP 429):
        RFC 7807 Problem Details JSON body
        Retry-After header with seconds until next token
    """

    def __init__(self, app: ASGIApp, rpm: int = 0) -> None:
        super().__init__(app)
        self.rpm = rpm
        self.rate = rpm / 60.0  # tokens per second
        self._buckets: dict[str, TokenBucket] = {}
        self._cleanup_interval = 300  # Clean stale buckets every 5 min
        self._last_cleanup = time.monotonic()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _get_bucket(self, ip: str) -> TokenBucket:
        """Get or create a token bucket for the given IP."""
        if ip not in self._buckets:
            self._buckets[ip] = TokenBucket(rate=self.rate, capacity=self.rpm)
        return self._buckets[ip]

    def _maybe_cleanup(self) -> None:
        """Periodically remove stale buckets to prevent memory growth."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_threshold = now - 120  # Remove buckets inactive for 2 min
        stale = [ip for ip, b in self._buckets.items() if b.last_refill < stale_threshold]
        for ip in stale:
            del self._buckets[ip]

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        if self.rpm <= 0:
            passthrough: Response = await call_next(request)
            return passthrough

        self._maybe_cleanup()

        ip = self._get_client_ip(request)
        bucket = self._get_bucket(ip)

        if not bucket.consume():
            retry_after = bucket.retry_after
            logger.warning(f"Rate limit exceeded for {ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "type": "https://mlx-manager.dev/errors/rate-limited",
                    "title": "Too Many Requests",
                    "status": 429,
                    "detail": f"Rate limit of {self.rpm} requests per minute exceeded",
                },
                headers={
                    "X-RateLimit-Limit": str(self.rpm),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(retry_after),
                },
            )

        response: Response = await call_next(request)

        # Add rate limit headers to successful responses
        remaining = max(0, int(bucket.tokens))
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
