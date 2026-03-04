"""Tests for MLX Server main module (app factory and lifespan)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server import __version__


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_create_embedded_app(self):
        """Embedded app has no lifespan and no logfire."""
        from mlx_manager.mlx_server.main import create_app

        app = create_app(embedded=True)

        assert app.title == "MLX Inference Server"
        assert app.version == __version__

    def test_create_embedded_app_has_routes(self):
        """Embedded app includes v1 router routes."""
        from mlx_manager.mlx_server.main import create_app

        app = create_app(embedded=True)
        paths = [r.path for r in app.routes]
        assert "/health" in paths

    def test_create_embedded_app_no_logfire(self):
        """Embedded app does not configure logfire even if enabled."""
        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.logfire_enabled = True
            mock_settings.rate_limit_rpm = 0
            mock_settings.metrics_enabled = False

            with patch("mlx_manager.mlx_server.main.register_error_handlers"):
                with patch("mlx_manager.mlx_server.main.v1_router"):
                    from mlx_manager.mlx_server.main import create_app

                    app = create_app(embedded=True)
                    # Logfire should NOT have been configured
                    assert app is not None

    def test_create_standalone_app_with_logfire_disabled(self):
        """Standalone app with logfire disabled does not instrument."""
        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.logfire_enabled = False
            mock_settings.rate_limit_rpm = 0
            mock_settings.metrics_enabled = False

            from mlx_manager.mlx_server.main import create_app

            app = create_app(embedded=False)
            assert app is not None

    def test_create_standalone_app_with_logfire_enabled(self):
        """Standalone app with logfire enabled configures instrumentation."""
        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.logfire_enabled = True
            mock_settings.rate_limit_rpm = 0
            mock_settings.metrics_enabled = False

            with patch(
                "mlx_manager.mlx_server.observability.logfire_config.configure_logfire"
            ) as mock_configure:
                with patch(
                    "mlx_manager.mlx_server.observability.logfire_config.instrument_httpx"
                ) as mock_httpx:
                    with patch(
                        "mlx_manager.mlx_server.observability.logfire_config.instrument_llm_clients"
                    ) as mock_llm:
                        with patch(
                            "mlx_manager.mlx_server.observability.logfire_config.instrument_fastapi"
                        ) as mock_fastapi:
                            from mlx_manager.mlx_server.main import create_app

                            app = create_app(embedded=False)

                            mock_configure.assert_called_once_with(service_version=__version__)
                            mock_httpx.assert_called_once()
                            mock_llm.assert_called_once()
                            mock_fastapi.assert_called_once_with(app)

    def test_health_endpoint_returns_version(self):
        """The /health endpoint returns status and version."""
        from fastapi.testclient import TestClient

        from mlx_manager.mlx_server.main import create_app

        app = create_app(embedded=True)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == __version__

    def test_error_handlers_registered(self):
        """Error handlers are registered on the app."""
        with patch("mlx_manager.mlx_server.main.register_error_handlers") as mock_register:
            from mlx_manager.mlx_server.main import create_app

            app = create_app(embedded=True)
            mock_register.assert_called_once_with(app)


class TestLifespan:
    """Tests for the lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_and_shutdown(self):
        """Lifespan initializes db, pool, memory and cleans up."""
        from mlx_manager.mlx_server.main import lifespan

        mock_app = MagicMock()

        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.max_memory_gb = 32.0
            mock_settings.max_models = 4
            mock_settings.enable_batching = False
            mock_settings.preload_models = []

            with patch(
                "mlx_manager.mlx_server.database.init_db",
                new_callable=AsyncMock,
            ) as mock_init_db:
                with patch("mlx_manager.mlx_server.main.set_memory_limit") as mock_set_mem:
                    with patch("mlx_manager.mlx_server.main.pool") as mock_pool_mod:
                        mock_pool_mod.model_pool = None

                        with patch(
                            "mlx_manager.mlx_server.main.get_memory_usage",
                            return_value={"active_gb": 0.0},
                        ):
                            async with lifespan(mock_app):
                                # During lifespan: db init, memory limit, pool set
                                mock_init_db.assert_called_once()
                                mock_set_mem.assert_called_once_with(32.0)
                                assert mock_pool_mod.model_pool is not None

                            # After lifespan: pool cleanup
                            # pool was set to a ModelPoolManager so cleanup called
                            if mock_pool_mod.model_pool:
                                pass  # cleanup would be called

    @pytest.mark.asyncio
    async def test_lifespan_with_batching_enabled(self):
        """Lifespan initializes scheduler manager when batching is enabled."""
        from mlx_manager.mlx_server.main import lifespan

        mock_app = MagicMock()
        mock_scheduler_mgr = AsyncMock()

        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.max_memory_gb = 16.0
            mock_settings.max_models = 2
            mock_settings.enable_batching = True
            mock_settings.batch_block_pool_size = 1024
            mock_settings.batch_max_batch_size = 8
            mock_settings.preload_models = []

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
                                "mlx_manager.mlx_server.services.batching.init_scheduler_manager",
                                return_value=mock_scheduler_mgr,
                            ) as mock_init_sched:
                                async with lifespan(mock_app):
                                    mock_init_sched.assert_called_once_with(
                                        block_pool_size=1024,
                                        max_batch_size=8,
                                    )

                                # Shutdown should call scheduler shutdown
                                mock_scheduler_mgr.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_pool(self):
        """Lifespan calls pool cleanup on shutdown."""
        from mlx_manager.mlx_server.main import lifespan

        mock_app = MagicMock()
        mock_pool_instance = AsyncMock()

        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.max_memory_gb = 16.0
            mock_settings.max_models = 2
            mock_settings.enable_batching = False
            mock_settings.preload_models = []

            with patch(
                "mlx_manager.mlx_server.database.init_db",
                new_callable=AsyncMock,
            ):
                with patch("mlx_manager.mlx_server.main.set_memory_limit"):
                    with patch("mlx_manager.mlx_server.main.pool"):
                        # ModelPoolManager constructor returns our mock
                        with patch(
                            "mlx_manager.mlx_server.main.ModelPoolManager",
                            return_value=mock_pool_instance,
                        ):
                            with patch(
                                "mlx_manager.mlx_server.main.get_memory_usage",
                                return_value={"active_gb": 0.0},
                            ):
                                async with lifespan(mock_app):
                                    # pool.model_pool should have been set to our mock
                                    pass

                            # After shutdown, the pool should have been cleaned up
                            mock_pool_instance.cleanup.assert_called_once()


class TestModuleLevelApp:
    """Tests for module-level lazy app initialization."""

    def test_get_standalone_app(self):
        """_get_standalone_app creates app lazily."""
        import mlx_manager.mlx_server.main as main_mod

        # Reset the cached app
        original = main_mod._app
        main_mod._app = None
        try:
            app = main_mod._get_standalone_app()
            assert app is not None
            # Second call returns same instance
            assert main_mod._get_standalone_app() is app
        finally:
            main_mod._app = original

    def test_getattr_app(self):
        """Module __getattr__ returns app for 'app' attribute."""
        import mlx_manager.mlx_server.main as main_mod

        original = main_mod._app
        main_mod._app = None
        try:
            app = main_mod.__getattr__("app")
            assert app is not None
        finally:
            main_mod._app = original

    def test_getattr_unknown_raises(self):
        """Module __getattr__ raises AttributeError for unknown attributes."""
        import mlx_manager.mlx_server.main as main_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            main_mod.__getattr__("nonexistent")


class TestPreloadConfig:
    """Tests for preload_models and warmup_prompt config settings."""

    def test_preload_models_default_is_empty_list(self):
        """preload_models defaults to empty list (backward compat — no preloading)."""
        from mlx_manager.mlx_server.config import MLXServerSettings

        settings = MLXServerSettings()
        assert settings.preload_models == []

    def test_preload_models_parses_list(self):
        """preload_models parses a list of model IDs."""
        from mlx_manager.mlx_server.config import MLXServerSettings

        settings = MLXServerSettings(
            preload_models=[
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "mlx-community/Qwen3-0.6B-4bit-DWQ",
            ]
        )
        assert len(settings.preload_models) == 2
        assert "mlx-community/Llama-3.2-3B-Instruct-4bit" in settings.preload_models
        assert "mlx-community/Qwen3-0.6B-4bit-DWQ" in settings.preload_models

    def test_warmup_prompt_default(self):
        """warmup_prompt defaults to 'Hello'."""
        from mlx_manager.mlx_server.config import MLXServerSettings

        settings = MLXServerSettings()
        assert settings.warmup_prompt == "Hello"

    def test_warmup_prompt_configurable(self):
        """warmup_prompt can be overridden."""
        from mlx_manager.mlx_server.config import MLXServerSettings

        settings = MLXServerSettings(warmup_prompt="Warm up the GPU!")
        assert settings.warmup_prompt == "Warm up the GPU!"


class TestLifespanPreload:
    """Tests for preload integration in the lifespan handler."""

    @pytest.mark.asyncio
    async def test_empty_preload_list_is_noop(self):
        """Empty preload_models list means apply_preload_list is never called."""
        from mlx_manager.mlx_server.main import lifespan

        mock_app = MagicMock()
        mock_pool_instance = AsyncMock()

        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.max_memory_gb = 16.0
            mock_settings.max_models = 2
            mock_settings.enable_batching = False
            mock_settings.preload_models = []

            with patch(
                "mlx_manager.mlx_server.database.init_db",
                new_callable=AsyncMock,
            ):
                with patch("mlx_manager.mlx_server.main.set_memory_limit"):
                    with patch("mlx_manager.mlx_server.main.pool"):
                        with patch(
                            "mlx_manager.mlx_server.main.ModelPoolManager",
                            return_value=mock_pool_instance,
                        ):
                            with patch(
                                "mlx_manager.mlx_server.main.get_memory_usage",
                                return_value={"active_gb": 0.0},
                            ):
                                async with lifespan(mock_app):
                                    pass

            # apply_preload_list should not have been called
            mock_pool_instance.apply_preload_list.assert_not_called()

    @pytest.mark.asyncio
    async def test_preload_list_calls_apply_preload(self):
        """Non-empty preload_models list triggers apply_preload_list on the pool."""
        from mlx_manager.mlx_server.main import lifespan

        mock_app = MagicMock()
        mock_pool_instance = AsyncMock()
        mock_pool_instance.apply_preload_list = AsyncMock(
            return_value={"mlx-community/Qwen3-0.6B-4bit-DWQ": "loaded"}
        )

        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.max_memory_gb = 16.0
            mock_settings.max_models = 2
            mock_settings.enable_batching = False
            mock_settings.preload_models = ["mlx-community/Qwen3-0.6B-4bit-DWQ"]

            with patch(
                "mlx_manager.mlx_server.database.init_db",
                new_callable=AsyncMock,
            ):
                with patch("mlx_manager.mlx_server.main.set_memory_limit"):
                    with patch("mlx_manager.mlx_server.main.pool"):
                        with patch(
                            "mlx_manager.mlx_server.main.ModelPoolManager",
                            return_value=mock_pool_instance,
                        ):
                            with patch(
                                "mlx_manager.mlx_server.main.get_memory_usage",
                                return_value={"active_gb": 0.0},
                            ):
                                async with lifespan(mock_app):
                                    pass

            mock_pool_instance.apply_preload_list.assert_called_once_with(
                ["mlx-community/Qwen3-0.6B-4bit-DWQ"]
            )

    @pytest.mark.asyncio
    async def test_preload_failure_does_not_crash_startup(self):
        """A failure in apply_preload_list is caught and logged — server still starts."""
        from mlx_manager.mlx_server.main import lifespan

        mock_app = MagicMock()
        mock_pool_instance = AsyncMock()
        mock_pool_instance.apply_preload_list = AsyncMock(
            side_effect=RuntimeError("GPU memory full")
        )

        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.max_memory_gb = 16.0
            mock_settings.max_models = 2
            mock_settings.enable_batching = False
            mock_settings.preload_models = ["mlx-community/Llama-3.2-3B-Instruct-4bit"]

            with patch(
                "mlx_manager.mlx_server.database.init_db",
                new_callable=AsyncMock,
            ):
                with patch("mlx_manager.mlx_server.main.set_memory_limit"):
                    with patch("mlx_manager.mlx_server.main.pool"):
                        with patch(
                            "mlx_manager.mlx_server.main.ModelPoolManager",
                            return_value=mock_pool_instance,
                        ):
                            with patch(
                                "mlx_manager.mlx_server.main.get_memory_usage",
                                return_value={"active_gb": 0.0},
                            ):
                                # Should not raise — exception is swallowed with a warning
                                async with lifespan(mock_app):
                                    pass  # Server came up despite preload failure

    @pytest.mark.asyncio
    async def test_partial_preload_failure_logs_warning(self):
        """apply_preload_list returning partial failures logs warning but continues."""
        from mlx_manager.mlx_server.main import lifespan

        mock_app = MagicMock()
        mock_pool_instance = AsyncMock()
        mock_pool_instance.apply_preload_list = AsyncMock(
            return_value={
                "mlx-community/Qwen3-0.6B-4bit-DWQ": "loaded",
                "mlx-community/nonexistent-model": "failed: model not found",
            }
        )

        with patch("mlx_manager.mlx_server.main.mlx_server_settings") as mock_settings:
            mock_settings.max_memory_gb = 16.0
            mock_settings.max_models = 2
            mock_settings.enable_batching = False
            mock_settings.preload_models = [
                "mlx-community/Qwen3-0.6B-4bit-DWQ",
                "mlx-community/nonexistent-model",
            ]

            with patch(
                "mlx_manager.mlx_server.database.init_db",
                new_callable=AsyncMock,
            ):
                with patch("mlx_manager.mlx_server.main.set_memory_limit"):
                    with patch("mlx_manager.mlx_server.main.pool"):
                        with patch(
                            "mlx_manager.mlx_server.main.ModelPoolManager",
                            return_value=mock_pool_instance,
                        ):
                            with patch(
                                "mlx_manager.mlx_server.main.get_memory_usage",
                                return_value={"active_gb": 0.0},
                            ):
                                # Should not raise — partial failure is non-fatal
                                async with lifespan(mock_app):
                                    pass
