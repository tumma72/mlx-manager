"""MLX Inference Server - FastAPI Application."""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from mlx_manager.mlx_server import __version__
from mlx_manager.mlx_server.api.v1 import v1_router
from mlx_manager.mlx_server.config import mlx_server_settings
from mlx_manager.mlx_server.models import pool
from mlx_manager.mlx_server.models.pool import ModelPoolManager
from mlx_manager.mlx_server.utils.memory import get_memory_usage, set_memory_limit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("MLX Server starting...")

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


app = FastAPI(
    title="MLX Inference Server",
    description="OpenAI-compatible inference server for MLX models",
    version=__version__,
    lifespan=lifespan,
)

# Include API routers
app.include_router(v1_router)

# Configure LogFire instrumentation (conditional on settings)
if mlx_server_settings.logfire_enabled:
    try:
        import logfire

        logfire.configure(token=mlx_server_settings.logfire_token)
        logfire.instrument_fastapi(app)
        logger.info("LogFire instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to configure LogFire: {e}")


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=mlx_server_settings.host,
        port=mlx_server_settings.port,
    )
