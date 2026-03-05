"""Graceful shutdown middleware for MLX Server.

Provides connection draining during server shutdown:
- Tracks active request count via middleware
- Returns 503 for new requests when draining
- Allows health checks through during drain (for monitoring)
- Supports waiting for all in-flight requests to complete with timeout
"""

import asyncio

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class ShutdownState:
    """Tracks server shutdown state and active request count."""

    def __init__(self) -> None:
        self._shutting_down = False
        self._active_requests = 0
        self._drain_complete = asyncio.Event()
        self._drain_complete.set()  # No drain in progress initially

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    @property
    def active_requests(self) -> int:
        return self._active_requests

    def start_drain(self) -> None:
        """Signal that the server is draining connections."""
        self._shutting_down = True
        if self._active_requests == 0:
            self._drain_complete.set()
        else:
            self._drain_complete.clear()
        logger.info(f"Drain started, {self._active_requests} active requests")

    def request_started(self) -> None:
        """Record that a new request has started processing."""
        self._active_requests += 1
        if self._shutting_down:
            self._drain_complete.clear()

    def request_finished(self) -> None:
        """Record that a request has finished processing."""
        self._active_requests -= 1
        if self._shutting_down and self._active_requests <= 0:
            self._active_requests = 0
            self._drain_complete.set()

    async def wait_for_drain(self, timeout: float) -> bool:
        """Wait for all active requests to complete.

        Args:
            timeout: Maximum seconds to wait for drain to complete.

        Returns:
            True if drain completed (all requests finished), False if timed out.
        """
        try:
            await asyncio.wait_for(self._drain_complete.wait(), timeout=timeout)
            return True
        except TimeoutError:
            logger.warning(
                f"Drain timed out after {timeout}s with {self._active_requests} "
                f"requests still active"
            )
            return False


# Module-level singleton
_shutdown_state: ShutdownState | None = None


def get_shutdown_state() -> ShutdownState:
    """Get or create the singleton ShutdownState instance."""
    global _shutdown_state
    if _shutdown_state is None:
        _shutdown_state = ShutdownState()
    return _shutdown_state


def reset_shutdown_state() -> None:
    """Reset the shutdown state singleton. Used for testing."""
    global _shutdown_state
    _shutdown_state = None


class GracefulShutdownMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks active requests and returns 503 during drain.

    During normal operation, increments/decrements the active request counter.
    When the server is draining (shutting down), rejects new requests with
    503 Service Unavailable, except for health check endpoints which are
    allowed through for monitoring purposes.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with shutdown awareness."""
        state = get_shutdown_state()

        # Allow health checks through even during drain (for monitoring)
        is_health = request.url.path in ("/health", "/v1/health")

        if state.is_shutting_down and not is_health:
            return JSONResponse(
                status_code=503,
                content={
                    "type": "https://mlx-manager.dev/errors/service-unavailable",
                    "title": "Service Unavailable",
                    "status": 503,
                    "detail": "Server is shutting down. Please retry with another instance.",
                },
                headers={"Retry-After": "5"},
            )

        state.request_started()
        try:
            response: Response = await call_next(request)
            return response
        finally:
            state.request_finished()
