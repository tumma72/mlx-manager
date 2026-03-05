"""Metrics middleware for automatic request tracking.

Records request latency and active request count for all HTTP requests
when Prometheus metrics are enabled.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that records request latency and active request count."""

    async def dispatch(self, request: Request, call_next) -> Response:
        from mlx_manager.mlx_server.services.metrics import get_metrics

        metrics = get_metrics()
        endpoint = request.url.path
        method = request.method

        metrics.active_requests.inc(endpoint=endpoint)
        start = time.perf_counter()

        try:
            response: Response = await call_next(request)
            return response
        finally:
            duration = time.perf_counter() - start
            metrics.active_requests.dec(endpoint=endpoint)
            metrics.request_latency.observe(duration, endpoint=endpoint, method=method)
