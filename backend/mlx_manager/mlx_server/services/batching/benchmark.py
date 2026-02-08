"""Benchmarking utilities for continuous batching throughput measurement.

This module provides tools to measure and compare throughput between
single-request sequential processing and batched concurrent processing.

The vLLM-MLX project demonstrated 3.4x improvement (328->1112 tok/s) on M4 Max,
which serves as our target benchmark.

Usage:
    # Programmatic
    from mlx_manager.mlx_server.services.batching.benchmark import run_benchmark
    result = await run_benchmark("mlx-community/Llama-3.2-3B-Instruct-4bit", BENCHMARK_PROMPTS)

    # CLI
    python -m mlx_manager.mlx_server.services.batching.benchmark --model MODEL_ID
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


@dataclass
class BenchmarkResult:
    """Result from a throughput benchmark run.

    Attributes:
        mode: Benchmark mode ("single" for sequential, "batched" for concurrent)
        num_requests: Number of requests processed
        total_tokens: Total tokens generated across all requests
        total_time_seconds: Wall clock time for all requests
        tokens_per_second: Throughput (total_tokens / total_time_seconds)
        avg_latency_ms: Average latency per request in milliseconds
        p50_latency_ms: 50th percentile (median) latency
        p99_latency_ms: 99th percentile latency
    """

    mode: str
    num_requests: int
    total_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    latencies_ms: list[float] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        """Human-readable summary of benchmark results."""
        return (
            f"BenchmarkResult({self.mode}):\n"
            f"  Requests: {self.num_requests}\n"
            f"  Total tokens: {self.total_tokens}\n"
            f"  Time: {self.total_time_seconds:.2f}s\n"
            f"  Throughput: {self.tokens_per_second:.1f} tok/s\n"
            f"  Latency (avg): {self.avg_latency_ms:.1f}ms\n"
            f"  Latency (p50): {self.p50_latency_ms:.1f}ms\n"
            f"  Latency (p99): {self.p99_latency_ms:.1f}ms"
        )


# Benchmark prompts with varying lengths (short, medium, long)
BENCHMARK_PROMPTS: list[str] = [
    # Short prompts (5-15 tokens)
    "What is 2+2?",
    "Define recursion.",
    "What is machine learning?",
    "Name three prime numbers.",
    "What color is the sky?",
    # Medium prompts (20-50 tokens)
    "Explain the difference between a list and a tuple in Python. Be concise.",
    "Write a haiku about programming and debugging code.",
    "What are the main benefits of using version control systems like Git?",
    "Describe how a neural network learns from data in simple terms.",
    "What is the time complexity of binary search and why?",
    # Longer prompts (50-100 tokens)
    "You are a helpful assistant. A user asks: I'm trying to understand continuous batching "
    "for LLM inference. Can you explain the key concepts and why it improves throughput "
    "compared to static batching? Keep your explanation focused and practical.",
    "Consider the following scenario: A web application needs to handle multiple concurrent "
    "LLM inference requests. What architectural patterns would you recommend to maximize "
    "throughput while maintaining reasonable latency? List the key considerations.",
    "Explain the PagedAttention mechanism used in vLLM. How does it improve memory efficiency "
    "for KV cache management during autoregressive generation? What are the tradeoffs?",
    "Compare and contrast REST APIs and GraphQL for building web services. What scenarios "
    "favor each approach? Include considerations for caching, versioning, and client needs.",
    "A developer is choosing between SQLite and PostgreSQL for a new application. What factors "
    "should they consider? When would each database be the better choice? Be specific.",
]


def calculate_percentile(sorted_values: list[float], percentile: float) -> float:
    """Calculate a percentile from sorted values.

    Args:
        sorted_values: List of values sorted in ascending order
        percentile: Percentile to calculate (0-100)

    Returns:
        The percentile value
    """
    if not sorted_values:
        return 0.0

    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    # Linear interpolation method
    k = (n - 1) * (percentile / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < n else f

    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def create_benchmark_result(
    mode: str,
    latencies_ms: list[float],
    tokens_per_request: list[int],
    total_time_seconds: float,
) -> BenchmarkResult:
    """Create a BenchmarkResult from raw measurements.

    Args:
        mode: "single" or "batched"
        latencies_ms: Per-request latencies in milliseconds
        tokens_per_request: Number of tokens generated per request
        total_time_seconds: Total wall clock time

    Returns:
        BenchmarkResult with computed statistics
    """
    num_requests = len(latencies_ms)
    total_tokens = sum(tokens_per_request)

    # Handle edge case of no requests
    if num_requests == 0:
        return BenchmarkResult(
            mode=mode,
            num_requests=0,
            total_tokens=0,
            total_time_seconds=total_time_seconds,
            tokens_per_second=0.0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p99_latency_ms=0.0,
            latencies_ms=[],
        )

    # Calculate statistics
    sorted_latencies = sorted(latencies_ms)
    avg_latency = statistics.mean(latencies_ms)
    p50_latency = calculate_percentile(sorted_latencies, 50)
    p99_latency = calculate_percentile(sorted_latencies, 99)

    # Throughput = total tokens / total time
    tokens_per_second = total_tokens / total_time_seconds if total_time_seconds > 0 else 0.0

    return BenchmarkResult(
        mode=mode,
        num_requests=num_requests,
        total_tokens=total_tokens,
        total_time_seconds=total_time_seconds,
        tokens_per_second=tokens_per_second,
        avg_latency_ms=avg_latency,
        p50_latency_ms=p50_latency,
        p99_latency_ms=p99_latency,
        latencies_ms=latencies_ms,
    )


async def run_single_request_benchmark(
    generate_fn: Callable[[str, int], Any],
    prompts: list[str],
    max_tokens: int = 100,
) -> BenchmarkResult:
    """Run sequential single-request benchmark.

    Processes each prompt one at a time, measuring individual latencies
    and total throughput.

    Args:
        generate_fn: Async function(prompt, max_tokens) -> list of tokens
        prompts: List of prompts to process
        max_tokens: Maximum tokens to generate per request

    Returns:
        BenchmarkResult with sequential processing metrics
    """
    latencies_ms: list[float] = []
    tokens_per_request: list[int] = []

    overall_start = time.perf_counter()

    for prompt in prompts:
        request_start = time.perf_counter()

        # Generate tokens for this prompt
        if asyncio.iscoroutinefunction(generate_fn):
            tokens = await generate_fn(prompt, max_tokens)
        else:
            tokens = generate_fn(prompt, max_tokens)

        request_end = time.perf_counter()

        # Record metrics
        latencies_ms.append((request_end - request_start) * 1000)
        tokens_per_request.append(len(tokens) if tokens else 0)

    overall_end = time.perf_counter()
    total_time = overall_end - overall_start

    return create_benchmark_result("single", latencies_ms, tokens_per_request, total_time)


async def run_batched_benchmark(
    submit_fn: Callable[[str, int], Any],
    prompts: list[str],
    max_tokens: int = 100,
) -> BenchmarkResult:
    """Run concurrent batched benchmark.

    Submits all prompts concurrently to the scheduler, measuring time
    for all to complete.

    Args:
        submit_fn: Async function(prompt, max_tokens) -> async generator of tokens
        prompts: List of prompts to process
        max_tokens: Maximum tokens to generate per request

    Returns:
        BenchmarkResult with batched processing metrics
    """

    async def process_request(prompt: str) -> tuple[float, int]:
        """Process a single request and return (latency_ms, token_count)."""
        request_start = time.perf_counter()
        token_count = 0

        # Consume the async generator
        async for _token in submit_fn(prompt, max_tokens):
            token_count += 1

        request_end = time.perf_counter()
        latency_ms = (request_end - request_start) * 1000
        return latency_ms, token_count

    overall_start = time.perf_counter()

    # Submit all requests concurrently
    tasks = [process_request(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    overall_end = time.perf_counter()
    total_time = overall_end - overall_start

    # Unpack results
    latencies_ms = [r[0] for r in results]
    tokens_per_request = [r[1] for r in results]

    return create_benchmark_result("batched", latencies_ms, tokens_per_request, total_time)


async def run_comparison_benchmark(
    single_generate_fn: Callable[[str, int], Any],
    batched_submit_fn: Callable[[str, int], Any],
    prompts: list[str] | None = None,
    max_tokens: int = 100,
) -> dict[str, Any]:
    """Run comparison benchmark between single and batched modes.

    Args:
        single_generate_fn: Function for single-request generation
        batched_submit_fn: Function for batched request submission
        prompts: Optional custom prompts (defaults to BENCHMARK_PROMPTS)
        max_tokens: Maximum tokens per request

    Returns:
        Dictionary with:
            - single: BenchmarkResult for sequential processing
            - batched: BenchmarkResult for concurrent processing
            - speedup: Throughput improvement ratio (batched/single)
            - prompts_count: Number of prompts used
    """
    prompts = prompts or BENCHMARK_PROMPTS

    # Run single-request benchmark first
    single_result = await run_single_request_benchmark(single_generate_fn, prompts, max_tokens)

    # Run batched benchmark
    batched_result = await run_batched_benchmark(batched_submit_fn, prompts, max_tokens)

    # Calculate speedup
    speedup = (
        batched_result.tokens_per_second / single_result.tokens_per_second
        if single_result.tokens_per_second > 0
        else 0.0
    )

    return {
        "single": single_result,
        "batched": batched_result,
        "speedup": speedup,
        "prompts_count": len(prompts),
    }


# Alias for external use
run_benchmark = run_comparison_benchmark


def print_comparison_results(results: dict[str, Any]) -> None:
    """Print formatted comparison benchmark results.

    Args:
        results: Dictionary from run_comparison_benchmark
    """
    single = results["single"]
    batched = results["batched"]
    speedup = results["speedup"]

    print("\n" + "=" * 60)
    print("CONTINUOUS BATCHING BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nPrompts: {results['prompts_count']}")
    print("Target improvement: 2-4x (vLLM-MLX achieved 3.4x)")

    print("\n--- Single Request (Sequential) ---")
    print(single)

    print("\n--- Batched (Concurrent) ---")
    print(batched)

    print("\n--- Comparison ---")
    print(f"Speedup: {speedup:.2f}x")

    if speedup >= 2.0:
        print("Result: MEETS TARGET (>= 2x improvement)")
    elif speedup >= 1.5:
        print("Result: PARTIAL (1.5-2x improvement)")
    else:
        print("Result: BELOW TARGET (< 1.5x improvement)")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark continuous batching throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m mlx_manager.mlx_server.services.batching.benchmark \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit \\
        --max-tokens 100

Note: This requires a running MLX server with batching enabled.
Set MLX_SERVER_ENABLE_BATCHING=true before starting the server.
        """,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID (e.g., mlx-community/Llama-3.2-3B-Instruct-4bit)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens per request (default: 100)",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=None,
        help="Number of prompts to use (default: all BENCHMARK_PROMPTS)",
    )

    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Prompts: {args.prompts or len(BENCHMARK_PROMPTS)}")
    print("\nTo run actual benchmark, integrate with MLX server:")
    print("1. Start server with MLX_SERVER_ENABLE_BATCHING=true")
    print("2. Use run_comparison_benchmark() with actual generate functions")
    print("\nBenchmark prompt samples:")
    for i, prompt in enumerate(BENCHMARK_PROMPTS[:3], 1):
        print(f"  {i}. {prompt[:60]}...")
