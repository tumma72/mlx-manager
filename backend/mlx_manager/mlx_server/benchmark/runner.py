"""Benchmark runner for MLX Server inference performance.

Measures throughput (tokens/second) for:
- Local inference (with/without batching)
- Cloud backends (OpenAI, Anthropic)
- Failover scenarios
"""

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from mlx_manager.config import DEFAULT_PORT


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    model: str
    backend: str  # local, openai, anthropic
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float
    success: bool
    error: str | None = None

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second (completion only)."""
        if self.duration_seconds == 0:
            return 0.0
        return self.completion_tokens / self.duration_seconds

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class BenchmarkSummary:
    """Aggregate statistics from multiple benchmark runs."""

    model: str
    backend: str
    runs: int
    successful_runs: int
    avg_tokens_per_second: float
    min_tokens_per_second: float
    max_tokens_per_second: float
    p50_tokens_per_second: float
    p95_tokens_per_second: float
    total_tokens_generated: int
    total_duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "runs": self.runs,
            "successful_runs": self.successful_runs,
            "avg_tok_s": round(self.avg_tokens_per_second, 1),
            "min_tok_s": round(self.min_tokens_per_second, 1),
            "max_tok_s": round(self.max_tokens_per_second, 1),
            "p50_tok_s": round(self.p50_tokens_per_second, 1),
            "p95_tok_s": round(self.p95_tokens_per_second, 1),
            "total_tokens": self.total_tokens_generated,
            "total_duration_s": round(self.total_duration_seconds, 2),
        }


class BenchmarkRunner:
    """Run benchmarks against MLX Server endpoints."""

    def __init__(
        self,
        base_url: str = f"http://localhost:{DEFAULT_PORT}",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "BenchmarkRunner":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("BenchmarkRunner not entered as context manager")
        return self._client

    async def run_single(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        stream: bool = False,
    ) -> BenchmarkResult:
        """Run a single benchmark request.

        Args:
            model: Model ID to benchmark
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            stream: Whether to use streaming mode

        Returns:
            BenchmarkResult with timing and token counts
        """
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        start = time.perf_counter()
        prompt_tokens = 0
        completion_tokens = 0
        error = None
        success = True

        try:
            if stream:
                # Streaming request
                async with self.client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                if delta.get("content"):
                                    completion_tokens += 1  # Approximate
                            except json.JSONDecodeError:
                                pass
            else:
                # Non-streaming request
                response = await self.client.post(
                    "/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

        except httpx.HTTPStatusError as e:
            success = False
            error = f"HTTP {e.response.status_code}: {e.response.text[:100]}"
        except Exception as e:
            success = False
            error = str(e)

        duration = time.perf_counter() - start

        # Detect backend from model routing
        backend = self._detect_backend(model)

        return BenchmarkResult(
            model=model,
            backend=backend,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_seconds=duration,
            success=success,
            error=error,
        )

    def _detect_backend(self, model: str) -> str:
        """Detect backend type from model name."""
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        else:
            return "local"

    async def run_benchmark(
        self,
        model: str,
        prompt: str,
        runs: int = 5,
        max_tokens: int = 256,
        warmup_runs: int = 1,
        stream: bool = False,
    ) -> BenchmarkSummary:
        """Run multiple benchmark iterations and calculate statistics.

        Args:
            model: Model ID to benchmark
            prompt: Input prompt text
            runs: Number of benchmark runs
            max_tokens: Maximum tokens per run
            warmup_runs: Number of warmup runs (not counted)
            stream: Whether to use streaming mode

        Returns:
            BenchmarkSummary with aggregate statistics
        """
        # Warmup runs
        for _ in range(warmup_runs):
            await self.run_single(model, prompt, max_tokens, stream)

        # Benchmark runs
        results: list[BenchmarkResult] = []
        for _ in range(runs):
            result = await self.run_single(model, prompt, max_tokens, stream)
            results.append(result)

        # Calculate statistics
        successful = [r for r in results if r.success]
        if not successful:
            return BenchmarkSummary(
                model=model,
                backend=self._detect_backend(model),
                runs=runs,
                successful_runs=0,
                avg_tokens_per_second=0,
                min_tokens_per_second=0,
                max_tokens_per_second=0,
                p50_tokens_per_second=0,
                p95_tokens_per_second=0,
                total_tokens_generated=0,
                total_duration_seconds=sum(r.duration_seconds for r in results),
            )

        tps_values = sorted(r.tokens_per_second for r in successful)
        total_tokens = sum(r.completion_tokens for r in successful)
        total_duration = sum(r.duration_seconds for r in successful)

        return BenchmarkSummary(
            model=model,
            backend=self._detect_backend(model),
            runs=runs,
            successful_runs=len(successful),
            avg_tokens_per_second=total_tokens / total_duration if total_duration > 0 else 0,
            min_tokens_per_second=tps_values[0],
            max_tokens_per_second=tps_values[-1],
            p50_tokens_per_second=self._percentile(tps_values, 50),
            p95_tokens_per_second=self._percentile(tps_values, 95),
            total_tokens_generated=total_tokens,
            total_duration_seconds=total_duration,
        )

    def _percentile(self, sorted_values: list[float], pct: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        idx = int(len(sorted_values) * pct / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]
