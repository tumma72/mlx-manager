"""Tests for the health checker service."""

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
    async def test_health_check_loop_calls_check_all(self, health_checker_instance):
        """Test that the loop calls _check_all_servers."""
        call_count = 0

        async def mock_check_all():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                health_checker_instance._running = False

        with patch.object(
            health_checker_instance, "_check_all_servers", side_effect=mock_check_all
        ):
            health_checker_instance._running = True
            await health_checker_instance._health_check_loop()

        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_health_check_loop_handles_errors(self, health_checker_instance):
        """Test that the loop continues on errors."""
        call_count = 0

        async def mock_check_all():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test error")
            if call_count >= 2:
                health_checker_instance._running = False

        with patch.object(
            health_checker_instance, "_check_all_servers", side_effect=mock_check_all
        ):
            with patch("builtins.print"):  # Suppress error output
                health_checker_instance._running = True
                await health_checker_instance._health_check_loop()

        # Should have continued after error
        assert call_count >= 2


class TestHealthCheckerCheckAllServers:
    """Tests for the _check_all_servers method."""

    @pytest.mark.asyncio
    async def test_check_all_with_no_instances(self, health_checker_instance):
        """Test checking when no instances are running."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("mlx_manager.services.health_checker.get_session", return_value=mock_context):
            await health_checker_instance._check_all_servers()

        # Should complete without error - commit is always called at the end
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_all_updates_stopped_server(self, health_checker_instance):
        """Test updating status when server has stopped."""
        mock_instance = MagicMock()
        mock_instance.profile_id = 1

        mock_profile = MagicMock()
        mock_profile.id = 1

        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.add = MagicMock()

        # First query returns instances, second returns profile
        mock_result1 = MagicMock()
        mock_result1.scalars.return_value.all.return_value = [mock_instance]

        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = mock_profile

        mock_session.execute = AsyncMock(side_effect=[mock_result1, mock_result2])

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("mlx_manager.services.health_checker.get_session", return_value=mock_context):
            with patch("mlx_manager.services.health_checker.server_manager") as mock_server_manager:
                mock_server_manager.is_running.return_value = False

                await health_checker_instance._check_all_servers()

        assert mock_instance.health_status == "stopped"
        mock_session.add.assert_called_with(mock_instance)

    @pytest.mark.asyncio
    async def test_check_all_updates_healthy_server(self, health_checker_instance):
        """Test updating status when server is healthy."""
        mock_instance = MagicMock()
        mock_instance.profile_id = 1

        mock_profile = MagicMock()
        mock_profile.id = 1

        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.add = MagicMock()

        mock_result1 = MagicMock()
        mock_result1.scalars.return_value.all.return_value = [mock_instance]

        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = mock_profile

        mock_session.execute = AsyncMock(side_effect=[mock_result1, mock_result2])

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("mlx_manager.services.health_checker.get_session", return_value=mock_context):
            with patch("mlx_manager.services.health_checker.server_manager") as mock_server_manager:
                mock_server_manager.is_running.return_value = True
                mock_server_manager.check_health = AsyncMock(return_value={"status": "healthy"})

                await health_checker_instance._check_all_servers()

        assert mock_instance.health_status == "healthy"
        mock_session.add.assert_called_with(mock_instance)

    @pytest.mark.asyncio
    async def test_check_all_skips_missing_profile(self, health_checker_instance):
        """Test that instances without profiles are skipped."""
        mock_instance = MagicMock()
        mock_instance.profile_id = 999

        mock_session = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.add = MagicMock()

        mock_result1 = MagicMock()
        mock_result1.scalars.return_value.all.return_value = [mock_instance]

        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = None  # Profile not found

        mock_session.execute = AsyncMock(side_effect=[mock_result1, mock_result2])

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("mlx_manager.services.health_checker.get_session", return_value=mock_context):
            await health_checker_instance._check_all_servers()

        # Instance should not be updated
        mock_session.add.assert_not_called()


class TestHealthCheckerIntegration:
    """Integration tests for the health checker."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, health_checker_instance):
        """Test starting, running briefly, and stopping."""
        check_count = 0

        async def mock_check():
            nonlocal check_count
            check_count += 1

        with patch.object(health_checker_instance, "_check_all_servers", side_effect=mock_check):
            await health_checker_instance.start()

            # Let it run for a brief moment
            await asyncio.sleep(0.1)

            await health_checker_instance.stop()

        assert health_checker_instance._running is False
