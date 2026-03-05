"""MLX Inference Server - FastAPI Application."""

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from loguru import logger
from starlette.responses import JSONResponse

from mlx_manager.mlx_server import __version__
from mlx_manager.mlx_server.api.v1 import v1_router
from mlx_manager.mlx_server.config import mlx_server_settings, reload_settings
from mlx_manager.mlx_server.errors import register_error_handlers
from mlx_manager.mlx_server.middleware.rate_limit import RateLimitMiddleware
from mlx_manager.mlx_server.middleware.request_id import RequestIDMiddleware
from mlx_manager.mlx_server.middleware.shutdown import (
    GracefulShutdownMiddleware,
    get_shutdown_state,
)
from mlx_manager.mlx_server.models import pool
from mlx_manager.mlx_server.models.pool import ModelPoolManager
from mlx_manager.mlx_server.utils.memory import get_memory_usage, set_memory_limit

__all__ = ["create_app"]


async def _audit_cleanup_loop() -> None:
    """Background task that periodically cleans up old audit log records."""
    from mlx_manager.mlx_server.database import cleanup_old_logs

    interval = mlx_server_settings.audit_cleanup_interval_minutes * 60
    while True:
        try:
            await cleanup_old_logs()
        except Exception as exc:
            logger.warning("Audit cleanup failed: %s", exc)
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for standalone mode."""
    # Startup
    logger.info("MLX Server starting...")

    # Initialize audit database
    from mlx_manager.mlx_server.database import cleanup_old_logs
    from mlx_manager.mlx_server.database import init_db as init_audit_db

    await init_audit_db()

    # Run initial cleanup at startup
    try:
        await cleanup_old_logs()
    except Exception as exc:
        logger.warning("Initial audit cleanup failed: %s", exc)

    # Start periodic audit cleanup background task
    audit_cleanup_task = asyncio.create_task(_audit_cleanup_loop())

    # Set memory limit (auto-detect from device if not explicitly configured)
    max_memory_gb = mlx_server_settings.max_memory_gb
    if max_memory_gb <= 0:
        from mlx_manager.mlx_server.utils.memory import auto_detect_memory_limit

        max_memory_gb = auto_detect_memory_limit()
    set_memory_limit(max_memory_gb)

    # Initialize model pool
    pool.model_pool = ModelPoolManager(
        max_memory_gb=max_memory_gb,
        max_models=mlx_server_settings.max_models,
    )

    # Initialize scheduler manager if batching enabled
    scheduler_mgr = None
    if mlx_server_settings.enable_batching:
        from mlx_manager.mlx_server.services.batching import init_scheduler_manager

        scheduler_mgr = init_scheduler_manager(
            block_pool_size=mlx_server_settings.batch_block_pool_size,
            max_batch_size=mlx_server_settings.batch_max_batch_size,
        )
        logger.info("Batching scheduler initialized")

    logger.info(f"Memory usage at startup: {get_memory_usage()}")

    # Preload models if configured
    if mlx_server_settings.preload_models:
        logger.info(
            f"Preloading {len(mlx_server_settings.preload_models)} model(s): "
            f"{mlx_server_settings.preload_models}"
        )
        try:
            results = await pool.model_pool.apply_preload_list(mlx_server_settings.preload_models)
            loaded_ok = [m for m, s in results.items() if s in ("loaded", "already_loaded")]
            failed = [m for m, s in results.items() if s.startswith("failed")]
            if loaded_ok:
                logger.info(f"Preload complete: {len(loaded_ok)} model(s) ready")
            if failed:
                logger.warning(f"Preload failed for {len(failed)} model(s): {failed}")
        except Exception as exc:
            logger.warning(f"Preload step encountered an unexpected error: {exc}")

    # Set up graceful shutdown signal handler
    shutdown_state = get_shutdown_state()
    loop = asyncio.get_running_loop()

    def handle_sigterm() -> None:
        logger.info("SIGTERM received, starting graceful shutdown...")
        shutdown_state.start_drain()

    try:
        loop.add_signal_handler(signal.SIGTERM, handle_sigterm)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass

    # Set up config hot-reload signal handler (Unix only)
    def handle_sighup() -> None:
        """Reload settings from environment/config on SIGHUP.

        Runs as a callback on the event loop — safe to call async-adjacent
        code (no blocking I/O).  Immutable settings (host, port,
        database_path) that changed are logged as WARNINGs.
        """
        logger.info("SIGHUP received, reloading configuration...")
        try:
            result = reload_settings()
            changes = result["changes"]
            warnings = result["warnings"]
            if changes:
                for name, diff in changes.items():
                    logger.info(
                        "Config reloaded: %s changed from %r to %r",
                        name,
                        diff["old"],
                        diff["new"],
                    )
            else:
                logger.info("Config reloaded: no changes detected")
            for warning in warnings:
                logger.warning("Config hot-reload: %s", warning)
        except Exception as exc:
            logger.exception("Config reload failed: %s", exc)

    try:
        loop.add_signal_handler(signal.SIGHUP, handle_sighup)
    except (NotImplementedError, AttributeError):
        # Windows does not have SIGHUP
        logger.debug("SIGHUP not available on this platform; skipping signal handler")

    logger.info("MLX Server ready")

    yield

    # Shutdown
    logger.info("MLX Server shutting down...")

    # Remove SIGHUP handler on shutdown
    try:
        loop.remove_signal_handler(signal.SIGHUP)
    except (NotImplementedError, AttributeError):
        pass

    # Cancel audit cleanup background task
    audit_cleanup_task.cancel()
    try:
        await audit_cleanup_task
    except asyncio.CancelledError:
        pass

    # Graceful shutdown: wait for active requests to drain
    if shutdown_state.is_shutting_down:
        drain_timeout = mlx_server_settings.drain_timeout_seconds
        logger.info(f"Waiting up to {drain_timeout}s for active requests to drain...")
        drained = await shutdown_state.wait_for_drain(drain_timeout)
        if drained:
            logger.info("All active requests completed")
        else:
            logger.warning("Proceeding with shutdown despite active requests")

    # Shutdown scheduler manager if initialized
    if scheduler_mgr is not None:
        await scheduler_mgr.shutdown()
        logger.info("Batching scheduler shutdown complete")

    if pool.model_pool:
        await pool.model_pool.cleanup()

    # Dispose the database engine to close aiosqlite connections cleanly
    from mlx_manager.mlx_server.database import _get_engine

    await _get_engine().dispose()

    logger.info("MLX Server stopped")


def create_app(embedded: bool = False) -> FastAPI:
    """Create the MLX Server FastAPI application.

    Args:
        embedded: If True, creates app for embedding in MLX Manager.
                  Skips LogFire configuration (parent app handles it)
                  but still instruments FastAPI for request tracing.
                  If False (default), creates standalone app with
                  full lifespan and LogFire.

    Returns:
        FastAPI application instance
    """
    # Configure LogFire only for standalone mode (parent configures in embedded mode)
    if not embedded and mlx_server_settings.logfire_enabled:
        from mlx_manager.mlx_server.observability.logfire_config import (
            configure_logfire,
            instrument_httpx,
            instrument_llm_clients,
        )

        configure_logfire(service_version=__version__)
        instrument_httpx()
        instrument_llm_clients()

    # Create app with or without lifespan handler
    app_instance = FastAPI(
        title="MLX Inference Server",
        description="OpenAI-compatible inference server for MLX models",
        version=__version__,
        lifespan=None if embedded else lifespan,
    )

    # Add metrics middleware (request latency + active request tracking)
    if mlx_server_settings.metrics_enabled:
        from mlx_manager.mlx_server.middleware.metrics import MetricsMiddleware

        app_instance.add_middleware(MetricsMiddleware)  # ty: ignore[invalid-argument-type]

    # Add request ID middleware (propagates/generates X-Request-ID for every request)
    app_instance.add_middleware(RequestIDMiddleware)  # ty: ignore[invalid-argument-type]

    # Add graceful shutdown middleware (tracks active requests, returns 503 during drain)
    app_instance.add_middleware(GracefulShutdownMiddleware)  # ty: ignore[invalid-argument-type]

    # Add rate limiting middleware (per-IP token bucket, disabled when rpm=0)
    if mlx_server_settings.rate_limit_rpm > 0:
        app_instance.add_middleware(RateLimitMiddleware, rpm=mlx_server_settings.rate_limit_rpm)  # ty: ignore[invalid-argument-type]

    # Register RFC 7807 error handlers
    register_error_handlers(app_instance)

    # Include API routers
    app_instance.include_router(v1_router)

    # Instrument FastAPI with LogFire for request tracing.
    # In embedded mode, the parent has already called configure_logfire(),
    # so instrument_fastapi() will use the existing configuration.
    # In standalone mode, we only instrument if logfire is enabled.
    if embedded or mlx_server_settings.logfire_enabled:
        from mlx_manager.mlx_server.observability.logfire_config import (
            instrument_fastapi,
        )

        instrument_fastapi(app_instance)

    # Add health endpoint (shutdown-aware)
    @app_instance.get("/health", response_model=None)
    async def health() -> dict[str, Any] | JSONResponse:
        """Health check endpoint. Returns 503 with draining status during shutdown."""
        state = get_shutdown_state()
        if state.is_shutting_down:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "draining",
                    "version": __version__,
                    "active_requests": state.active_requests,
                },
            )
        return {"status": "healthy", "version": __version__}

    return app_instance


# Lazy initialization for standalone app
# This ensures LogFire is not configured until the app is actually accessed
_app: FastAPI | None = None


def _get_standalone_app() -> FastAPI:
    """Get or create the standalone app instance."""
    global _app
    if _app is None:
        _app = create_app(embedded=False)
    return _app


# Provide module-level 'app' attribute via __getattr__ for backward compatibility
# This allows `from mlx_manager.mlx_server.main import app` to work
# while deferring app creation until first access
def __getattr__(name: str) -> Any:
    """Lazy attribute access for module-level app."""
    if name == "app":
        return _get_standalone_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        _get_standalone_app(),
        host=mlx_server_settings.host,
        port=mlx_server_settings.port,
    )
