"""Lightweight Prometheus-compatible metrics collection.

This module provides a custom implementation of Counter, Gauge, and Histogram
metric types with thread-safe operations and Prometheus text format output.
No external dependencies (prometheus_client) are required.
"""

import threading
from collections import defaultdict


class Counter:
    """Thread-safe counter metric."""

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help_text = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment the counter by the given value."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] += value

    def collect(self) -> list[str]:
        """Collect metric lines in Prometheus text format."""
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} counter"]
        with self._lock:
            for labels, value in self._values.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in labels) if labels else ""
                metric_name = f"{self.name}{{{label_str}}}" if label_str else self.name
                lines.append(f"{metric_name} {value}")
        return lines


class Gauge:
    """Thread-safe gauge metric."""

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help_text = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        """Set the gauge to the given value."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment the gauge by the given value."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] += value

    def dec(self, value: float = 1.0, **labels: str) -> None:
        """Decrement the gauge by the given value."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[key] -= value

    def collect(self) -> list[str]:
        """Collect metric lines in Prometheus text format."""
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} gauge"]
        with self._lock:
            for labels, value in self._values.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in labels) if labels else ""
                metric_name = f"{self.name}{{{label_str}}}" if label_str else self.name
                lines.append(f"{metric_name} {value}")
        return lines


class Histogram:
    """Thread-safe histogram metric with predefined buckets."""

    DEFAULT_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        float("inf"),
    )

    def __init__(self, name: str, help_text: str, buckets: tuple[float, ...] | None = None) -> None:
        self.name = name
        self.help_text = help_text
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: dict[tuple[tuple[str, str], ...], dict[float, int]] = {}
        self._sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._totals: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        """Record an observation in the histogram."""
        key = tuple(sorted(labels.items()))
        with self._lock:
            if key not in self._counts:
                self._counts[key] = {b: 0 for b in self.buckets}
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1
            self._sums[key] += value
            self._totals[key] += 1

    def collect(self) -> list[str]:
        """Collect metric lines in Prometheus text format."""
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for labels, buckets in self._counts.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in labels)
                for bucket_bound, count in sorted(buckets.items()):
                    le = "+Inf" if bucket_bound == float("inf") else str(bucket_bound)
                    if label_str:
                        lines.append(f'{self.name}_bucket{{{label_str},le="{le}"}} {count}')
                    else:
                        lines.append(f'{self.name}_bucket{{le="{le}"}} {count}')
                if label_str:
                    lines.append(f"{self.name}_sum{{{label_str}}} {self._sums[labels]}")
                    lines.append(f"{self.name}_count{{{label_str}}} {self._totals[labels]}")
                else:
                    lines.append(f"{self.name}_sum {self._sums[labels]}")
                    lines.append(f"{self.name}_count {self._totals[labels]}")
        return lines


class MetricsRegistry:
    """Central metrics registry for MLX Server."""

    def __init__(self) -> None:
        # Request metrics
        self.request_latency = Histogram(
            "mlx_request_latency_seconds",
            "Request latency in seconds",
        )
        self.active_requests = Gauge(
            "mlx_active_requests",
            "Number of currently active requests",
        )

        # Token metrics
        self.token_throughput = Counter(
            "mlx_token_throughput_total",
            "Total tokens generated",
        )

        # Model metrics
        self.model_load_duration = Histogram(
            "mlx_model_load_duration_seconds",
            "Model loading duration in seconds",
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf")),
        )
        self.model_memory = Gauge(
            "mlx_model_memory_bytes",
            "Memory used by loaded models",
        )

        # Pool metrics
        self.pool_cache_hits = Counter(
            "mlx_pool_cache_hits_total",
            "Number of model pool cache hits",
        )
        self.pool_cache_misses = Counter(
            "mlx_pool_cache_misses_total",
            "Number of model pool cache misses",
        )

        self._all_metrics: list[Counter | Gauge | Histogram] = [
            self.request_latency,
            self.active_requests,
            self.token_throughput,
            self.model_load_duration,
            self.model_memory,
            self.pool_cache_hits,
            self.pool_cache_misses,
        ]

    def collect_all(self) -> str:
        """Collect all metrics in Prometheus text format."""
        lines: list[str] = []
        for metric in self._all_metrics:
            lines.extend(metric.collect())
            lines.append("")  # Blank line between metrics
        return "\n".join(lines) + "\n"


# Module-level singleton
_registry: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    """Get or create the metrics registry singleton."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


def reset_metrics() -> None:
    """Reset the metrics registry singleton (for testing)."""
    global _registry
    _registry = None
