"""Tests for the health checker service.

With the embedded MLX Server, the health checker monitors the model pool
instead of external subprocess instances.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def health_checker_instance():
    """Create a fresh HealthChecker instance."""
    from mlx_manager.services.health_checker import HealthChecker

    return HealthChecker(interval=1)


class TestHealthCheckerStartStop:
    """Tests for start and stop methods."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self, health_checker_instance):
        """Test that start creates a background task."""
        with patch.object(health_checker_instance, "_health_check_loop", new_callable=AsyncMock):
            await health_checker_instance.start()

            assert health_checker_instance._running is True
            assert health_checker_instance._task is not None

            # Clean up
            await health_checker_instance.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, health_checker_instance):
        """Test that stop cancels the background task."""
        with patch.object(health_checker_instance, "_health_check_loop", new_callable=AsyncMock):
            await health_checker_instance.start()
            await health_checker_instance.stop()

            assert health_checker_instance._running is False

    @pytest.mark.asyncio
    async def test_stop_without_start(self, health_checker_instance):
        """Test that stop works even if never started."""
        await health_checker_instance.stop()

        assert health_checker_instance._running is False
        assert health_checker_instance._task is None


class TestHealthCheckerLoop:
    """Tests for the health check loop."""

    @pytest.mark.asyncio
    async def test_health_check_loop_calls_check_model_pool(self, health_checker_instance):
        """Test that the loop calls _check_model_pool."""
        call_count = 0

        async def mock_check_pool():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                health_checker_instance._running = False

        with (
            patch.object(health_checker_instance, "_check_model_pool", side_effect=mock_check_pool),
            patch("mlx_manager.services.health_checker.asyncio.sleep", new_callable=AsyncMock),
        ):
            health_checker_instance._running = True
            await health_checker_instance._health_check_loop()

        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_health_check_loop_handles_errors(self, health_checker_instance):
        """Test that the loop continues on errors."""
        call_count = 0

        async def mock_check_pool():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test error")
            if call_count >= 2:
                health_checker_instance._running = False

        with (
            patch.object(health_checker_instance, "_check_model_pool", side_effect=mock_check_pool),
            patch("mlx_manager.services.health_checker.asyncio.sleep", new_callable=AsyncMock),
        ):
            health_checker_instance._running = True
            await health_checker_instance._health_check_loop()

        # Should have continued after error
        assert call_count >= 2


class TestHealthCheckerCheckModelPool:
    """Tests for the _check_model_pool method."""

    @pytest.mark.asyncio
    async def test_check_pool_logs_status(self, health_checker_instance):
        """Test checking model pool logs status."""
        mock_pool = MagicMock()
        mock_pool.get_loaded_models.return_value = ["model1", "model2"]

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool
        ):
            with patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 8.0},
            ):
                await health_checker_instance._check_model_pool()

        # Should complete without error
        mock_pool.get_loaded_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_pool_handles_not_initialized(self, health_checker_instance):
        """Test checking pool when not initialized."""
        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            side_effect=RuntimeError("Pool not initialized"),
        ):
            # Should not raise
            await health_checker_instance._check_model_pool()


class TestHealthCheckerIntegration:
    """Integration tests for the health checker."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, health_checker_instance):
        """Test starting and stopping the health checker."""
        mock_pool = MagicMock()
        mock_pool.get_loaded_models.return_value = []

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool
        ):
            with patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 0.0},
            ):
                await health_checker_instance.start()
                # Let it run briefly
                await asyncio.sleep(0.1)
                await health_checker_instance.stop()

        assert health_checker_instance._running is False
