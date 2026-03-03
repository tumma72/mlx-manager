"""Tests for Prometheus-compatible metrics collection."""

from unittest.mock import patch

import pytest

from mlx_manager.mlx_server.services.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    get_metrics,
    reset_metrics,
)


@pytest.fixture(autouse=True)
def _reset():
    """Reset metrics singleton between tests."""
    reset_metrics()
    yield
    reset_metrics()


# ============================================================================
# Counter tests
# ============================================================================


class TestCounter:
    def test_increment_default(self):
        counter = Counter("test_total", "A test counter")
        counter.inc()
        counter.inc()
        lines = counter.collect()
        assert "test_total 2.0" in lines

    def test_increment_custom_value(self):
        counter = Counter("test_total", "A test counter")
        counter.inc(5.0)
        lines = counter.collect()
        assert "test_total 5.0" in lines

    def test_increment_with_labels(self):
        counter = Counter("http_requests_total", "Total HTTP requests")
        counter.inc(method="GET", endpoint="/api")
        counter.inc(method="POST", endpoint="/api")
        counter.inc(method="GET", endpoint="/api")
        lines = counter.collect()
        assert 'http_requests_total{endpoint="/api",method="GET"} 2.0' in lines
        assert 'http_requests_total{endpoint="/api",method="POST"} 1.0' in lines

    def test_collect_format_has_help_and_type(self):
        counter = Counter("my_counter", "Help text here")
        lines = counter.collect()
        assert lines[0] == "# HELP my_counter Help text here"
        assert lines[1] == "# TYPE my_counter counter"

    def test_collect_empty_counter(self):
        counter = Counter("empty_total", "Empty counter")
        lines = counter.collect()
        assert len(lines) == 2  # Only HELP and TYPE lines


# ============================================================================
# Gauge tests
# ============================================================================


class TestGauge:
    def test_set(self):
        gauge = Gauge("temperature", "Current temperature")
        gauge.set(42.5)
        lines = gauge.collect()
        assert "temperature 42.5" in lines

    def test_inc(self):
        gauge = Gauge("active", "Active items")
        gauge.inc()
        gauge.inc()
        lines = gauge.collect()
        assert "active 2.0" in lines

    def test_dec(self):
        gauge = Gauge("active", "Active items")
        gauge.inc()
        gauge.inc()
        gauge.dec()
        lines = gauge.collect()
        assert "active 1.0" in lines

    def test_set_with_labels(self):
        gauge = Gauge("memory_bytes", "Memory usage")
        gauge.set(1024.0, model="llama")
        gauge.set(2048.0, model="qwen")
        lines = gauge.collect()
        assert 'memory_bytes{model="llama"} 1024.0' in lines
        assert 'memory_bytes{model="qwen"} 2048.0' in lines

    def test_inc_dec_with_labels(self):
        gauge = Gauge("active_requests", "Active requests")
        gauge.inc(endpoint="/chat")
        gauge.inc(endpoint="/chat")
        gauge.dec(endpoint="/chat")
        lines = gauge.collect()
        assert 'active_requests{endpoint="/chat"} 1.0' in lines

    def test_collect_format_has_help_and_type(self):
        gauge = Gauge("my_gauge", "Help text")
        lines = gauge.collect()
        assert lines[0] == "# HELP my_gauge Help text"
        assert lines[1] == "# TYPE my_gauge gauge"


# ============================================================================
# Histogram tests
# ============================================================================


class TestHistogram:
    def test_observe_single_value(self):
        hist = Histogram(
            "request_duration_seconds",
            "Request duration",
            buckets=(0.1, 0.5, 1.0, float("inf")),
        )
        hist.observe(0.25)
        lines = hist.collect()
        # 0.25 is <= 0.5, 1.0, +Inf but not <= 0.1
        assert 'request_duration_seconds_bucket{le="0.1"} 0' in lines
        assert 'request_duration_seconds_bucket{le="0.5"} 1' in lines
        assert 'request_duration_seconds_bucket{le="1.0"} 1' in lines
        assert 'request_duration_seconds_bucket{le="+Inf"} 1' in lines
        assert "request_duration_seconds_sum 0.25" in lines
        assert "request_duration_seconds_count 1" in lines

    def test_observe_multiple_values(self):
        hist = Histogram(
            "latency",
            "Latency",
            buckets=(0.1, 0.5, 1.0, float("inf")),
        )
        hist.observe(0.05)
        hist.observe(0.3)
        hist.observe(0.8)
        lines = hist.collect()
        assert 'latency_bucket{le="0.1"} 1' in lines
        assert 'latency_bucket{le="0.5"} 2' in lines
        assert 'latency_bucket{le="1.0"} 3' in lines
        assert 'latency_bucket{le="+Inf"} 3' in lines
        assert f"latency_sum {0.05 + 0.3 + 0.8}" in lines
        assert "latency_count 3" in lines

    def test_observe_with_labels(self):
        hist = Histogram(
            "request_duration",
            "Duration",
            buckets=(0.5, 1.0, float("inf")),
        )
        hist.observe(0.3, method="GET")
        hist.observe(0.8, method="POST")
        lines = hist.collect()
        assert 'request_duration_bucket{method="GET",le="0.5"} 1' in lines
        assert 'request_duration_bucket{method="POST",le="0.5"} 0' in lines
        assert 'request_duration_bucket{method="POST",le="1.0"} 1' in lines

    def test_collect_format_has_help_and_type(self):
        hist = Histogram("my_hist", "Help text", buckets=(1.0, float("inf")))
        lines = hist.collect()
        assert lines[0] == "# HELP my_hist Help text"
        assert lines[1] == "# TYPE my_hist histogram"

    def test_default_buckets(self):
        hist = Histogram("default_hist", "Default buckets")
        assert hist.buckets == Histogram.DEFAULT_BUCKETS
        assert float("inf") in hist.buckets


# ============================================================================
# MetricsRegistry tests
# ============================================================================


class TestMetricsRegistry:
    def test_collect_all_returns_all_metrics(self):
        registry = MetricsRegistry()
        output = registry.collect_all()
        # Should contain all metric names
        assert "mlx_request_latency_seconds" in output
        assert "mlx_active_requests" in output
        assert "mlx_token_throughput_total" in output
        assert "mlx_model_load_duration_seconds" in output
        assert "mlx_model_memory_bytes" in output
        assert "mlx_pool_cache_hits_total" in output
        assert "mlx_pool_cache_misses_total" in output

    def test_collect_all_ends_with_newline(self):
        registry = MetricsRegistry()
        output = registry.collect_all()
        assert output.endswith("\n")

    def test_collect_all_has_help_and_type_for_each(self):
        registry = MetricsRegistry()
        output = registry.collect_all()
        # Each metric should have HELP and TYPE lines
        for metric in registry._all_metrics:
            assert f"# HELP {metric.name}" in output
            assert f"# TYPE {metric.name}" in output

    def test_collect_all_with_data(self):
        registry = MetricsRegistry()
        registry.token_throughput.inc(100, model="llama")
        registry.active_requests.set(5)
        registry.pool_cache_hits.inc(10)
        output = registry.collect_all()
        assert 'mlx_token_throughput_total{model="llama"} 100' in output
        assert "mlx_active_requests 5" in output
        assert "mlx_pool_cache_hits_total 10" in output


# ============================================================================
# Singleton tests
# ============================================================================


class TestSingleton:
    def test_get_metrics_returns_same_instance(self):
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_reset_metrics_creates_new_instance(self):
        m1 = get_metrics()
        reset_metrics()
        m2 = get_metrics()
        assert m1 is not m2


# ============================================================================
# Metrics endpoint tests
# ============================================================================


class TestMetricsEndpoint:
    @pytest.mark.anyio
    async def test_metrics_disabled_returns_404(self):
        """Metrics endpoint returns 404 when metrics_enabled=False."""
        from mlx_manager.mlx_server.api.v1.admin import prometheus_metrics

        with patch("mlx_manager.mlx_server.api.v1.admin.get_settings") as mock_settings:
            mock_settings.return_value.metrics_enabled = False
            with pytest.raises(Exception) as exc_info:
                await prometheus_metrics()
            # HTTPException with 404
            assert exc_info.value.status_code == 404  # type: ignore[union-attr]
            assert "Metrics not enabled" in str(exc_info.value.detail)  # type: ignore[union-attr]

    @pytest.mark.anyio
    async def test_metrics_enabled_returns_200(self):
        """Metrics endpoint returns 200 with Prometheus text when enabled."""
        from mlx_manager.mlx_server.api.v1.admin import prometheus_metrics

        with patch("mlx_manager.mlx_server.api.v1.admin.get_settings") as mock_settings:
            mock_settings.return_value.metrics_enabled = True
            response = await prometheus_metrics()
            assert response.status_code == 200
            assert "text/plain" in response.media_type
            body = response.body.decode()
            assert "mlx_request_latency_seconds" in body
            assert "mlx_active_requests" in body


# ============================================================================
# MetricsMiddleware tests
# ============================================================================


class TestMetricsMiddleware:
    @pytest.mark.anyio
    async def test_middleware_tracks_request_latency(self):
        """MetricsMiddleware records latency and active request count."""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import Response
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from mlx_manager.mlx_server.middleware.metrics import MetricsMiddleware

        async def homepage(request: Request) -> Response:
            return Response("OK", media_type="text/plain")

        app = Starlette(routes=[Route("/test", homepage)])
        app.add_middleware(MetricsMiddleware)

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200

        # Check metrics were recorded
        metrics = get_metrics()
        latency_lines = metrics.request_latency.collect()
        # Should have bucket entries for endpoint=/test, method=GET
        found_bucket = False
        for line in latency_lines:
            if 'endpoint="/test"' in line and 'method="GET"' in line:
                found_bucket = True
                break
        assert found_bucket, f"No latency bucket found for /test GET in: {latency_lines}"

        # Active requests should be back to 0 (request completed)
        active_lines = metrics.active_requests.collect()
        for line in active_lines:
            if 'endpoint="/test"' in line:
                assert "0.0" in line, f"Active requests not back to 0: {line}"

    @pytest.mark.anyio
    async def test_middleware_tracks_on_error(self):
        """MetricsMiddleware still records metrics even when handler raises."""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from mlx_manager.mlx_server.middleware.metrics import MetricsMiddleware

        async def error_handler(request: Request):
            raise ValueError("boom")

        app = Starlette(routes=[Route("/error", error_handler)])
        app.add_middleware(MetricsMiddleware)

        client = TestClient(app, raise_server_exceptions=False)
        client.get("/error")

        # Metrics should still be recorded
        metrics = get_metrics()
        latency_lines = metrics.request_latency.collect()
        found = any('endpoint="/error"' in line for line in latency_lines)
        assert found, "Latency should be recorded even on error"
