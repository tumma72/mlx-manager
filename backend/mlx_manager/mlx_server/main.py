"""MLX Inference Server - FastAPI Application."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from loguru import logger

from mlx_manager.mlx_server import __version__
from mlx_manager.mlx_server.api.v1 import v1_router
from mlx_manager.mlx_server.config import mlx_server_settings
from mlx_manager.mlx_server.errors import register_error_handlers
from mlx_manager.mlx_server.models import pool
from mlx_manager.mlx_server.models.pool import ModelPoolManager
from mlx_manager.mlx_server.utils.memory import get_memory_usage, set_memory_limit

__all__ = ["create_app"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for standalone mode."""
    # Startup
    logger.info("MLX Server starting...")

    # Initialize audit database
    from mlx_manager.mlx_server.database import init_db as init_audit_db

    await init_audit_db()

    # Set memory limit
    set_memory_limit(mlx_server_settings.max_memory_gb)

    # Initialize model pool
    pool.model_pool = ModelPoolManager(
        max_memory_gb=mlx_server_settings.max_memory_gb,
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
    logger.info("MLX Server ready")

    yield

    # Shutdown
    logger.info("MLX Server shutting down...")

    # Shutdown scheduler manager if initialized
    if scheduler_mgr is not None:
        await scheduler_mgr.shutdown()
        logger.info("Batching scheduler shutdown complete")

    if pool.model_pool:
        await pool.model_pool.cleanup()
    logger.info("MLX Server stopped")


def create_app(embedded: bool = False) -> FastAPI:
    """Create the MLX Server FastAPI application.

    Args:
        embedded: If True, creates app for embedding in MLX Manager.
                  Skips LogFire configuration and lifespan handler
                  (parent app handles these).
                  If False (default), creates standalone app with
                  full lifespan and LogFire.

    Returns:
        FastAPI application instance
    """
    # Configure LogFire only for standalone mode
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

    # Register RFC 7807 error handlers
    register_error_handlers(app_instance)

    # Include API routers
    app_instance.include_router(v1_router)

    # Instrument FastAPI with LogFire only for standalone mode
    if not embedded and mlx_server_settings.logfire_enabled:
        from mlx_manager.mlx_server.observability.logfire_config import (
            instrument_fastapi,
        )

        instrument_fastapi(app_instance)

    # Add health endpoint
    @app_instance.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint."""
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
