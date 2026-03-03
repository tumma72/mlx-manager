"""Tests for graceful shutdown with connection draining."""

import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mlx_manager.mlx_server import __version__
from mlx_manager.mlx_server.middleware.shutdown import (
    ShutdownState,
    get_shutdown_state,
    reset_shutdown_state,
)


@pytest.fixture(autouse=True)
def _reset_shutdown():
    """Reset shutdown state singleton before and after each test."""
    reset_shutdown_state()
    yield
    reset_shutdown_state()


class TestShutdownState:
    """Tests for ShutdownState tracking."""

    def test_initial_state(self):
        """Fresh state is not shutting down with zero active requests."""
        state = ShutdownState()
        assert state.is_shutting_down is False
        assert state.active_requests == 0

    def test_request_started_increments(self):
        """request_started increments active request count."""
        state = ShutdownState()
        state.request_started()
        assert state.active_requests == 1
        state.request_started()
        assert state.active_requests == 2

    def test_request_finished_decrements(self):
        """request_finished decrements active request count."""
        state = ShutdownState()
        state.request_started()
        state.request_started()
        state.request_finished()
        assert state.active_requests == 1

    def test_request_finished_clamps_to_zero(self):
        """request_finished does not go below zero."""
        state = ShutdownState()
        state._shutting_down = True
        state._drain_complete.clear()
        state.request_finished()
        assert state.active_requests == 0

    def test_start_drain_sets_flag(self):
        """start_drain sets is_shutting_down to True."""
        state = ShutdownState()
        state.start_drain()
        assert state.is_shutting_down is True

    def test_start_drain_with_no_active_requests_sets_event(self):
        """start_drain with no active requests immediately signals completion."""
        state = ShutdownState()
        state.start_drain()
        assert state._drain_complete.is_set()

    def test_start_drain_with_active_requests_clears_event(self):
        """start_drain with active requests clears the completion event."""
        state = ShutdownState()
        state.request_started()
        state.start_drain()
        assert not state._drain_complete.is_set()

    def test_drain_completes_when_last_request_finishes(self):
        """Drain event is set when the last active request finishes."""
        state = ShutdownState()
        state.request_started()
        state.request_started()
        state.start_drain()
        assert not state._drain_complete.is_set()

        state.request_finished()
        assert not state._drain_complete.is_set()

        state.request_finished()
        assert state._drain_complete.is_set()

    @pytest.mark.asyncio
    async def test_wait_for_drain_completes(self):
        """wait_for_drain returns True when all requests complete."""
        state = ShutdownState()
        state.request_started()
        state.start_drain()

        async def finish_request():
            await asyncio.sleep(0.01)
            state.request_finished()

        asyncio.create_task(finish_request())
        result = await state.wait_for_drain(timeout=5.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_drain_times_out(self):
        """wait_for_drain returns False when timeout is exceeded."""
        state = ShutdownState()
        state.request_started()
        state.start_drain()

        result = await state.wait_for_drain(timeout=0.05)
        assert result is False
        assert state.active_requests == 1

    @pytest.mark.asyncio
    async def test_wait_for_drain_immediate_when_no_requests(self):
        """wait_for_drain returns immediately when no active requests."""
        state = ShutdownState()
        state.start_drain()

        result = await state.wait_for_drain(timeout=0.05)
        assert result is True


class TestShutdownStateSingleton:
    """Tests for the module-level singleton pattern."""

    def test_get_shutdown_state_returns_same_instance(self):
        """get_shutdown_state returns the same instance on repeated calls."""
        state1 = get_shutdown_state()
        state2 = get_shutdown_state()
        assert state1 is state2

    def test_reset_shutdown_state_clears_singleton(self):
        """reset_shutdown_state allows a fresh instance to be created."""
        state1 = get_shutdown_state()
        reset_shutdown_state()
        state2 = get_shutdown_state()
        assert state1 is not state2


class TestGracefulShutdownMiddleware:
    """Tests for the middleware that enforces shutdown behavior."""

    def _make_app(self):
        """Create a test app with the graceful shutdown middleware."""
        from mlx_manager.mlx_server.main import create_app

        return create_app(embedded=True)

    def test_normal_requests_pass_through(self):
        """Requests are processed normally when not draining."""
        app = self._make_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_requests_rejected_during_drain(self):
        """Non-health requests get 503 during drain."""
        app = self._make_app()
        client = TestClient(app)

        # Start draining
        state = get_shutdown_state()
        state.start_drain()

        # Non-health endpoint returns 503
        response = client.get("/v1/models")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == 503
        assert data["title"] == "Service Unavailable"
        assert "shutting down" in data["detail"]
        assert response.headers.get("Retry-After") == "5"

    def test_health_allowed_during_drain(self):
        """Health endpoint is accessible during drain for monitoring."""
        app = self._make_app()
        client = TestClient(app)

        # Start draining
        state = get_shutdown_state()
        state.start_drain()

        # Health endpoint still works but returns 503 with draining status
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "draining"
        assert data["version"] == __version__


class TestHealthEndpointDraining:
    """Tests for health endpoint behavior during shutdown."""

    def test_health_returns_healthy_normally(self):
        """Health endpoint returns 200 with healthy status normally."""
        from mlx_manager.mlx_server.main import create_app

        app = create_app(embedded=True)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == __version__

    def test_health_returns_503_when_draining(self):
        """Health endpoint returns 503 with draining status during shutdown."""
        from mlx_manager.mlx_server.main import create_app

        app = create_app(embedded=True)
        client = TestClient(app)

        # Trigger drain
        state = get_shutdown_state()
        state.start_drain()

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "draining"
        assert data["version"] == __version__
        assert "active_requests" in data


class TestConfigDrainTimeout:
    """Tests for the drain_timeout_seconds config setting."""

    def test_default_drain_timeout(self):
        """Default drain timeout is 30 seconds."""
        from mlx_manager.mlx_server.config import MLXServerSettings

        settings = MLXServerSettings()
        assert settings.drain_timeout_seconds == 30.0

    def test_drain_timeout_bounds(self):
        """Drain timeout must be between 1 and 300 seconds."""
        from pydantic import ValidationError

        from mlx_manager.mlx_server.config import MLXServerSettings

        # Too low
        with pytest.raises(ValidationError):
            MLXServerSettings(drain_timeout_seconds=0.5)

        # Too high
        with pytest.raises(ValidationError):
            MLXServerSettings(drain_timeout_seconds=400.0)

        # Valid extremes
        settings_min = MLXServerSettings(drain_timeout_seconds=1.0)
        assert settings_min.drain_timeout_seconds == 1.0

        settings_max = MLXServerSettings(drain_timeout_seconds=300.0)
        assert settings_max.drain_timeout_seconds == 300.0


class TestLifespanWithShutdown:
    """Tests for lifespan shutdown drain integration."""

    @pytest.mark.asyncio
    async def test_lifespan_registers_sigterm_handler(self):
        """Lifespan sets up SIGTERM handler that triggers drain."""
        from unittest.mock import MagicMock

        from mlx_manager.mlx_server.main import lifespan

        mock_app = MagicMock()
        registered_handlers: dict = {}

        def mock_add_signal_handler(sig, handler):
            registered_handlers[sig] = handler

        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.max_memory_gb = 16.0
            mock_settings.max_models = 2
            mock_settings.enable_batching = False
            mock_settings.drain_timeout_seconds = 30.0

            with patch(
                "mlx_manager.mlx_server.database.init_db",
                new_callable=pytest.importorskip("unittest.mock").AsyncMock,
            ):
                with patch("mlx_manager.mlx_server.main.set_memory_limit"):
                    with patch("mlx_manager.mlx_server.main.pool") as mock_pool_mod:
                        mock_pool_mod.model_pool = None
                        with patch(
                            "mlx_manager.mlx_server.main.get_memory_usage",
                            return_value={"active_gb": 0.0},
                        ):
                            with patch("asyncio.get_running_loop") as mock_get_loop:
                                mock_loop = MagicMock()
                                mock_loop.add_signal_handler = mock_add_signal_handler
                                mock_get_loop.return_value = mock_loop

                                async with lifespan(mock_app):
                                    # SIGTERM handler should be registered
                                    import signal

                                    assert signal.SIGTERM in registered_handlers

                                    # Calling the handler should trigger drain
                                    handler = registered_handlers[signal.SIGTERM]
                                    handler()
                                    state = get_shutdown_state()
                                    assert state.is_shutting_down is True

    @pytest.mark.asyncio
    async def test_lifespan_drain_waits_on_shutdown(self):
        """Lifespan waits for drain during shutdown when drain is active."""
        from unittest.mock import AsyncMock, MagicMock

        from mlx_manager.mlx_server.main import lifespan

        mock_app = MagicMock()

        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.max_memory_gb = 16.0
            mock_settings.max_models = 2
            mock_settings.enable_batching = False
            mock_settings.drain_timeout_seconds = 1.0

            with patch(
                "mlx_manager.mlx_server.database.init_db",
                new_callable=AsyncMock,
            ):
                with patch("mlx_manager.mlx_server.main.set_memory_limit"):
                    with patch("mlx_manager.mlx_server.main.pool") as mock_pool_mod:
                        mock_pool_mod.model_pool = None
                        with patch(
                            "mlx_manager.mlx_server.main.get_memory_usage",
                            return_value={"active_gb": 0.0},
                        ):
                            with patch(
                                "mlx_manager.mlx_server.main.get_shutdown_state"
                            ) as mock_get_state:
                                mock_state = MagicMock()
                                mock_state.is_shutting_down = True
                                mock_state.wait_for_drain = AsyncMock(return_value=True)
                                mock_get_state.return_value = mock_state

                                with patch("asyncio.get_running_loop") as mock_get_loop:
                                    mock_loop = MagicMock()
                                    mock_get_loop.return_value = mock_loop

                                    async with lifespan(mock_app):
                                        pass

                                    # Drain wait should have been called
                                    mock_state.wait_for_drain.assert_called_once_with(1.0)
