"""Request ID middleware for MLX Server.

Propagates or generates a unique request ID for every request,
making it available via request.state.request_id and returned
in the X-Request-ID response header.
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that ensures every request has a unique request ID.

    - Reads incoming X-Request-ID header if present (client-provided)
    - If absent, generates a UUID v4 request ID in format req_{uuid_hex[:12]}
    - Stores it in request.state.request_id for downstream access
    - Adds X-Request-ID header to every response
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with request ID propagation."""
        # Accept client-provided ID or generate new one
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
