"""Tests for MLX memory management utilities."""

from unittest.mock import MagicMock, patch

from mlx_manager.mlx_server.utils.memory import (
    clear_cache,
    get_memory_usage,
    reset_peak_memory,
    set_memory_limit,
)


class TestGetMemoryUsage:
    """Tests for get_memory_usage function."""

    def test_returns_memory_dict(self):
        """get_memory_usage returns dict with active, peak, cache."""
        with patch("mlx_manager.mlx_server.utils.memory._get_mx") as mock_mx:
            mock_mx_mod = MagicMock()
            mock_mx_mod.get_active_memory.return_value = 1024**3  # 1 GB
            mock_mx_mod.get_peak_memory.return_value = 2 * 1024**3  # 2 GB
            mock_mx_mod.get_cache_memory.return_value = 512 * 1024**2  # 0.5 GB
            mock_mx.return_value = mock_mx_mod

            result = get_memory_usage()

            assert "active_gb" in result
            assert "peak_gb" in result
            assert "cache_gb" in result
            assert abs(result["active_gb"] - 1.0) < 0.01
            assert abs(result["peak_gb"] - 2.0) < 0.01
            assert abs(result["cache_gb"] - 0.5) < 0.01

    def test_returns_zeros_on_error(self):
        """get_memory_usage returns zeros when MLX is unavailable."""
        with patch("mlx_manager.mlx_server.utils.memory._get_mx") as mock_mx:
            mock_mx.side_effect = ImportError("No mlx")

            result = get_memory_usage()

            assert result["active_gb"] == 0.0
            assert result["peak_gb"] == 0.0
            assert result["cache_gb"] == 0.0

    def test_returns_zeros_on_runtime_error(self):
        """get_memory_usage handles runtime errors gracefully."""
        with patch("mlx_manager.mlx_server.utils.memory._get_mx") as mock_mx:
            mock_mx_mod = MagicMock()
            mock_mx_mod.get_active_memory.side_effect = RuntimeError("GPU error")
            mock_mx.return_value = mock_mx_mod

            result = get_memory_usage()

            assert result["active_gb"] == 0.0
            assert result["peak_gb"] == 0.0
            assert result["cache_gb"] == 0.0


class TestClearCache:
    """Tests for clear_cache function."""

    def test_calls_mx_clear_cache(self):
        """clear_cache calls mx.synchronize and mx.clear_cache."""
        with patch("mlx_manager.mlx_server.utils.memory._get_mx") as mock_mx:
            mock_mx_mod = MagicMock()
            mock_mx.return_value = mock_mx_mod

            clear_cache()

            mock_mx_mod.synchronize.assert_called_once()
            mock_mx_mod.clear_cache.assert_called_once()

    def test_handles_error_gracefully(self):
        """clear_cache doesn't raise on failure."""
        with patch("mlx_manager.mlx_server.utils.memory._get_mx") as mock_mx:
            mock_mx.side_effect = ImportError("No mlx")

            # Should not raise
            clear_cache()


class TestSetMemoryLimit:
    """Tests for set_memory_limit function."""

    def test_sets_limit_in_bytes(self):
        """set_memory_limit converts GB to bytes and calls mx.set_memory_limit."""
        with patch("mlx_manager.mlx_server.utils.memory._get_mx") as mock_mx:
            mock_mx_mod = MagicMock()
            mock_mx.return_value = mock_mx_mod

            set_memory_limit(16.0)

            expected_bytes = int(16.0 * 1024**3)
            mock_mx_mod.set_memory_limit.assert_called_once_with(expected_bytes)

    def test_handles_error_gracefully(self):
        """set_memory_limit doesn't raise on failure."""
        with patch("mlx_manager.mlx_server.utils.memory._get_mx") as mock_mx:
            mock_mx.side_effect = ImportError("No mlx")

            # Should not raise
            set_memory_limit(8.0)


class TestResetPeakMemory:
    """Tests for reset_peak_memory function."""

    def test_calls_mx_reset_peak(self):
        """reset_peak_memory calls mx.reset_peak_memory."""
        with patch("mlx_manager.mlx_server.utils.memory._get_mx") as mock_mx:
            mock_mx_mod = MagicMock()
            mock_mx.return_value = mock_mx_mod

            reset_peak_memory()

            mock_mx_mod.reset_peak_memory.assert_called_once()

    def test_handles_error_gracefully(self):
        """reset_peak_memory doesn't raise on failure."""
        with patch("mlx_manager.mlx_server.utils.memory._get_mx") as mock_mx:
            mock_mx.side_effect = ImportError("No mlx")

            # Should not raise
            reset_peak_memory()
