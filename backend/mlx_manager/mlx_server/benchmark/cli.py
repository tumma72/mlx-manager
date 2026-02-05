"""CLI for running MLX Server benchmarks.

Usage:
    mlx-benchmark run --model mlx-community/Llama-3.2-3B-Instruct-4bit --runs 5
    mlx-benchmark run --model gpt-4o-mini --backend openai
    mlx-benchmark suite  # Run full benchmark suite
"""

import asyncio

import typer

from mlx_manager.config import DEFAULT_PORT
from mlx_manager.mlx_server.benchmark.runner import BenchmarkRunner, BenchmarkSummary

app = typer.Typer(
    name="mlx-benchmark",
    help="Benchmark MLX Server inference performance",
)

# Default test prompts by size
PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain the concept of machine learning in simple terms.",
    "long": (
        "You are a helpful AI assistant. Please provide a detailed explanation "
        "of the following topic:\n\n"
        "What are the key differences between transformer models and recurrent "
        "neural networks (RNNs)? Include discussion of:\n"
        "1. Architecture differences\n"
        "2. Training characteristics\n"
        "3. Inference speed\n"
        "4. Memory usage\n"
        "5. Common use cases\n\n"
        "Be thorough but concise."
    ),
}

# Default models by tier (if available)
MODEL_TIERS = {
    "small": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "medium": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "large": "mlx-community/Llama-3.3-70B-Instruct-4bit",
}


def print_result(summary: BenchmarkSummary) -> None:
    """Pretty print benchmark result."""
    print(f"\n{'=' * 60}")
    print(f"Model: {summary.model}")
    print(f"Backend: {summary.backend}")
    print(f"Runs: {summary.successful_runs}/{summary.runs} successful")
    print(f"{'=' * 60}")
    print(f"  Avg throughput: {summary.avg_tokens_per_second:.1f} tok/s")
    print(f"  Min throughput: {summary.min_tokens_per_second:.1f} tok/s")
    print(f"  Max throughput: {summary.max_tokens_per_second:.1f} tok/s")
    print(f"  P50 throughput: {summary.p50_tokens_per_second:.1f} tok/s")
    print(f"  P95 throughput: {summary.p95_tokens_per_second:.1f} tok/s")
    print(f"  Total tokens: {summary.total_tokens_generated}")
    print(f"  Total time: {summary.total_duration_seconds:.2f}s")


@app.command()
def run(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model ID to benchmark",
    ),
    runs: int = typer.Option(
        5,
        "--runs",
        "-n",
        help="Number of benchmark runs",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    prompt_size: str = typer.Option(
        "medium",
        "--prompt",
        help="Prompt size: short, medium, long",
    ),
    warmup: int = typer.Option(
        1,
        "--warmup",
        help="Number of warmup runs",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Use streaming mode",
    ),
    base_url: str = typer.Option(
        f"http://localhost:{DEFAULT_PORT}",
        "--url",
        help="MLX Server base URL",
    ),
) -> None:
    """Run benchmark against a specific model."""
    prompt = PROMPTS.get(prompt_size, PROMPTS["medium"])

    async def _run() -> None:
        async with BenchmarkRunner(base_url=base_url) as runner:
            print(f"Running {runs} benchmarks for {model}...")
            print(f"Prompt: {prompt[:50]}...")
            print(f"Max tokens: {max_tokens}")
            print(f"Stream: {stream}")

            summary = await runner.run_benchmark(
                model=model,
                prompt=prompt,
                runs=runs,
                max_tokens=max_tokens,
                warmup_runs=warmup,
                stream=stream,
            )
            print_result(summary)

    asyncio.run(_run())


@app.command()
def suite(
    base_url: str = typer.Option(
        f"http://localhost:{DEFAULT_PORT}",
        "--url",
        help="MLX Server base URL",
    ),
    runs: int = typer.Option(
        3,
        "--runs",
        "-n",
        help="Runs per model",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON)",
    ),
) -> None:
    """Run full benchmark suite across model tiers."""
    import json

    async def _run() -> list[dict]:
        results = []
        async with BenchmarkRunner(base_url=base_url) as runner:
            for tier, model in MODEL_TIERS.items():
                print(f"\nBenchmarking {tier} tier: {model}")
                try:
                    summary = await runner.run_benchmark(
                        model=model,
                        prompt=PROMPTS["medium"],
                        runs=runs,
                        max_tokens=256,
                        warmup_runs=1,
                    )
                    print_result(summary)
                    results.append({"tier": tier, **summary.to_dict()})
                except Exception as e:
                    print(f"  Error: {e}")
                    results.append({"tier": tier, "model": model, "error": str(e)})
        return results

    results = asyncio.run(_run())

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output}")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
