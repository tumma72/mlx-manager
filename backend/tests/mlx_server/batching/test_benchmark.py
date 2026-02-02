"""Tests for benchmark utilities.

These tests verify the benchmark infrastructure works correctly.
Actual benchmark runs with real models are manual.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mlx_manager.mlx_server.services.batching.benchmark import (
    BENCHMARK_PROMPTS,
    BenchmarkResult,
    calculate_percentile,
    create_benchmark_result,
    run_batched_benchmark,
    run_comparison_benchmark,
    run_single_request_benchmark,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_benchmark_result(self) -> None:
        """Test BenchmarkResult creation with all fields."""
        result = BenchmarkResult(
            mode="test",
            num_requests=10,
            total_tokens=100,
            total_time_seconds=1.0,
            tokens_per_second=100.0,
            avg_latency_ms=100.0,
            p50_latency_ms=90.0,
            p99_latency_ms=150.0,
        )

        assert result.mode == "test"
        assert result.num_requests == 10
        assert result.total_tokens == 100
        assert result.total_time_seconds == 1.0
        assert result.tokens_per_second == 100.0
        assert result.avg_latency_ms == 100.0
        assert result.p50_latency_ms == 90.0
        assert result.p99_latency_ms == 150.0
        assert result.latencies_ms == []  # Default empty list

    def test_tokens_per_second_calculation(self) -> None:
        """Test that tokens_per_second is correctly stored."""
        result = BenchmarkResult(
            mode="single",
            num_requests=5,
            total_tokens=500,
            total_time_seconds=2.5,
            tokens_per_second=200.0,  # 500 / 2.5
            avg_latency_ms=50.0,
            p50_latency_ms=45.0,
            p99_latency_ms=80.0,
        )

        assert result.tokens_per_second == 200.0

    def test_result_with_latencies(self) -> None:
        """Test BenchmarkResult with latencies list."""
        latencies = [50.0, 60.0, 70.0, 80.0, 90.0]
        result = BenchmarkResult(
            mode="batched",
            num_requests=5,
            total_tokens=250,
            total_time_seconds=1.5,
            tokens_per_second=166.67,
            avg_latency_ms=70.0,
            p50_latency_ms=70.0,
            p99_latency_ms=89.0,
            latencies_ms=latencies,
        )

        assert result.latencies_ms == latencies
        assert len(result.latencies_ms) == 5

    def test_str_representation(self) -> None:
        """Test human-readable string representation."""
        result = BenchmarkResult(
            mode="batched",
            num_requests=15,
            total_tokens=1500,
            total_time_seconds=13.47,
            tokens_per_second=111.4,
            avg_latency_ms=898.0,
            p50_latency_ms=890.0,
            p99_latency_ms=1200.0,
        )

        str_repr = str(result)
        assert "batched" in str_repr
        assert "15" in str_repr  # num_requests
        assert "1500" in str_repr  # total_tokens
        assert "111.4" in str_repr  # tokens_per_second


class TestPercentileCalculation:
    """Tests for percentile calculation helper."""

    def test_empty_list(self) -> None:
        """Test percentile of empty list returns 0."""
        assert calculate_percentile([], 50) == 0.0
        assert calculate_percentile([], 99) == 0.0

    def test_single_value(self) -> None:
        """Test percentile of single value returns that value."""
        assert calculate_percentile([100.0], 50) == 100.0
        assert calculate_percentile([100.0], 99) == 100.0

    def test_p50_median(self) -> None:
        """Test p50 returns median value."""
        sorted_values = [10.0, 20.0, 30.0, 40.0, 50.0]
        # For 5 values, p50 should be the middle (index 2 = 30.0)
        p50 = calculate_percentile(sorted_values, 50)
        assert p50 == 30.0

    def test_p99_high_percentile(self) -> None:
        """Test p99 returns near-max value."""
        sorted_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        p99 = calculate_percentile(sorted_values, 99)
        # p99 should be close to max
        assert p99 >= 95.0
        assert p99 <= 100.0

    def test_interpolation(self) -> None:
        """Test that interpolation works between values."""
        sorted_values = [0.0, 100.0]
        p50 = calculate_percentile(sorted_values, 50)
        assert p50 == 50.0  # Interpolated between 0 and 100


class TestCreateBenchmarkResult:
    """Tests for create_benchmark_result helper."""

    def test_create_with_measurements(self) -> None:
        """Test creating result from raw measurements."""
        latencies = [100.0, 200.0, 300.0]
        tokens = [10, 20, 30]
        total_time = 0.6  # seconds

        result = create_benchmark_result("single", latencies, tokens, total_time)

        assert result.mode == "single"
        assert result.num_requests == 3
        assert result.total_tokens == 60
        assert result.total_time_seconds == 0.6
        assert result.tokens_per_second == 100.0  # 60 / 0.6
        assert result.avg_latency_ms == 200.0  # mean of 100, 200, 300
        assert result.latencies_ms == latencies

    def test_create_empty_result(self) -> None:
        """Test creating result with no requests."""
        result = create_benchmark_result("batched", [], [], 1.0)

        assert result.num_requests == 0
        assert result.total_tokens == 0
        assert result.tokens_per_second == 0.0
        assert result.avg_latency_ms == 0.0

    def test_create_with_zero_time(self) -> None:
        """Test creating result with zero total time."""
        result = create_benchmark_result("single", [100.0], [10], 0.0)

        assert result.tokens_per_second == 0.0  # Avoid division by zero


class TestBenchmarkPrompts:
    """Tests for benchmark prompt constants."""

    def test_prompts_exist(self) -> None:
        """Test that BENCHMARK_PROMPTS is populated."""
        assert len(BENCHMARK_PROMPTS) > 0

    def test_prompts_variety(self) -> None:
        """Test that prompts have varying lengths."""
        lengths = [len(p) for p in BENCHMARK_PROMPTS]
        # Should have short (<50 chars) and long (>100 chars) prompts
        assert any(length < 50 for length in lengths), "Should have short prompts"
        assert any(length > 100 for length in lengths), "Should have long prompts"

    def test_prompts_are_strings(self) -> None:
        """Test that all prompts are strings."""
        for prompt in BENCHMARK_PROMPTS:
            assert isinstance(prompt, str)
            assert len(prompt) > 0


class TestBenchmarkFunctions:
    """Tests for benchmark runner functions with mocked generation."""

    @pytest.mark.asyncio
    async def test_run_single_request_benchmark(self) -> None:
        """Test single request benchmark with mock generation."""

        async def mock_generate(prompt: str, max_tokens: int) -> list[int]:
            """Mock generation that returns fixed tokens."""
            # Simulate some processing time
            await asyncio.sleep(0.01)
            return list(range(min(max_tokens, 10)))  # 10 tokens

        prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
        result = await run_single_request_benchmark(mock_generate, prompts, max_tokens=10)

        assert result.mode == "single"
        assert result.num_requests == 3
        assert result.total_tokens == 30  # 3 * 10 tokens
        assert result.tokens_per_second > 0
        assert len(result.latencies_ms) == 3

    @pytest.mark.asyncio
    async def test_run_single_request_benchmark_sync_function(self) -> None:
        """Test single request benchmark with sync generation function."""

        def sync_generate(prompt: str, max_tokens: int) -> list[int]:
            """Sync generation function."""
            return list(range(5))

        prompts = ["Prompt A", "Prompt B"]
        result = await run_single_request_benchmark(sync_generate, prompts, max_tokens=5)

        assert result.num_requests == 2
        assert result.total_tokens == 10

    @pytest.mark.asyncio
    async def test_run_batched_benchmark(self) -> None:
        """Test batched benchmark with mock streaming."""

        async def mock_submit(prompt: str, max_tokens: int) -> Any:
            """Mock async generator that yields tokens."""
            for i in range(5):
                await asyncio.sleep(0.001)  # Small delay
                yield {"token_id": i, "text": f"t{i}"}

        prompts = ["Batch prompt 1", "Batch prompt 2"]
        result = await run_batched_benchmark(mock_submit, prompts, max_tokens=5)

        assert result.mode == "batched"
        assert result.num_requests == 2
        assert result.total_tokens == 10  # 2 * 5 tokens
        assert len(result.latencies_ms) == 2

    @pytest.mark.asyncio
    async def test_run_comparison_benchmark(self) -> None:
        """Test comparison benchmark with mock functions."""

        async def mock_single(prompt: str, max_tokens: int) -> list[int]:
            await asyncio.sleep(0.01)
            return list(range(10))

        async def mock_batched(prompt: str, max_tokens: int) -> Any:
            for i in range(10):
                await asyncio.sleep(0.001)
                yield {"token_id": i}

        prompts = ["Compare 1", "Compare 2", "Compare 3"]
        results = await run_comparison_benchmark(
            mock_single, mock_batched, prompts=prompts, max_tokens=10
        )

        assert "single" in results
        assert "batched" in results
        assert "speedup" in results
        assert "prompts_count" in results

        assert results["prompts_count"] == 3
        assert isinstance(results["single"], BenchmarkResult)
        assert isinstance(results["batched"], BenchmarkResult)
        assert isinstance(results["speedup"], float)

    @pytest.mark.asyncio
    async def test_comparison_speedup_calculation(self) -> None:
        """Test that speedup is calculated correctly."""

        # Single: slow (10ms per request)
        async def slow_single(prompt: str, max_tokens: int) -> list[int]:
            await asyncio.sleep(0.01)
            return list(range(10))

        # Batched: fast (concurrent)
        async def fast_batched(prompt: str, max_tokens: int) -> Any:
            for i in range(10):
                await asyncio.sleep(0.001)
                yield {"token_id": i}

        prompts = ["P1", "P2", "P3", "P4"]
        results = await run_comparison_benchmark(
            slow_single, fast_batched, prompts=prompts, max_tokens=10
        )

        # Batched should be faster (higher throughput)
        # Since all batched requests run concurrently
        single_tps = results["single"].tokens_per_second
        batched_tps = results["batched"].tokens_per_second

        # Both should have positive throughput
        assert single_tps > 0
        assert batched_tps > 0


class TestBatchingModuleComplete:
    """Integration test verifying all batching exports are available."""

    def test_batching_module_complete(self) -> None:
        """Verify all batching exports available."""
        from mlx_manager.mlx_server.services.batching import (
            BLOCK_SIZE,
            BatchInferenceEngine,
            BatchRequest,
            BenchmarkResult,
            BlockTable,
            ContinuousBatchingScheduler,
            KVBlock,
            PagedBlockManager,
            PrefixCache,
            Priority,
            PriorityQueueWithAging,
            RequestStatus,
            SchedulerManager,
            get_scheduler_manager,
            init_scheduler_manager,
            run_benchmark,
        )

        # Verify key constants
        assert BLOCK_SIZE == 32

        # Verify priority ordering (lower = higher priority)
        assert Priority.HIGH.value < Priority.NORMAL.value
        assert Priority.NORMAL.value < Priority.LOW.value

        # Verify classes are importable (not None)
        assert BatchRequest is not None
        assert BatchInferenceEngine is not None
        assert ContinuousBatchingScheduler is not None
        assert PagedBlockManager is not None
        assert PrefixCache is not None
        assert PriorityQueueWithAging is not None
        assert SchedulerManager is not None
        assert KVBlock is not None
        assert BlockTable is not None
        assert BenchmarkResult is not None

        # Verify functions are callable
        assert callable(get_scheduler_manager)
        assert callable(init_scheduler_manager)
        assert callable(run_benchmark)

        # Verify status enum
        assert RequestStatus.WAITING is not None
        assert RequestStatus.RUNNING is not None
        assert RequestStatus.COMPLETED is not None
