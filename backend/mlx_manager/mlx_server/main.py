"""MLX Inference Server - FastAPI Application."""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from mlx_manager.mlx_server import __version__
from mlx_manager.mlx_server.api.v1 import v1_router
from mlx_manager.mlx_server.config import mlx_server_settings

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
    # TODO: Initialize model pool (Plan 03)
    logger.info("MLX Server ready")
    yield
    # Shutdown
    logger.info("MLX Server shutting down...")
    # TODO: Cleanup model pool


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
