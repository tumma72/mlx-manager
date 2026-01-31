"""Servers API router.

With the embedded MLX Server, this router provides status information about
the model pool and loaded models. Start/stop/restart endpoints are no longer
needed since the server is embedded and always running.
"""

import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from mlx_manager.dependencies import get_current_user
from mlx_manager.models import User

router = APIRouter(prefix="/api/servers", tags=["servers"])


class EmbeddedServerStatus(BaseModel):
    """Status of the embedded MLX Server."""

    status: str  # "running", "not_initialized"
    type: str = "embedded"
    uptime_seconds: float = 0.0
    loaded_models: list[str] = []
    memory_used_gb: float = 0.0
    memory_limit_gb: float = 0.0


class LoadedModelInfo(BaseModel):
    """Information about a loaded model."""

    model_id: str
    model_type: str
    size_gb: float
    loaded_at: float
    last_used: float
    preloaded: bool


class ServerHealthStatus(BaseModel):
    """Health status of the embedded server."""

    status: str  # "healthy", "degraded", "unhealthy"
    model_pool_initialized: bool = False
    loaded_model_count: int = 0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0


# Server start time for uptime calculation
_server_start_time = time.time()


@router.get("", response_model=list[EmbeddedServerStatus])
async def list_servers(
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[EmbeddedServerStatus]:
    """Return embedded MLX Server status.

    With embedded mode, there's always exactly one server (the application itself).
    Returns a list with one element for API compatibility.
    """
    try:
        from mlx_manager.mlx_server.models.pool import get_model_pool
        from mlx_manager.mlx_server.utils.memory import get_memory_usage

        pool = get_model_pool()
        loaded_models = pool.get_loaded_models()
        memory = get_memory_usage()

        return [
            EmbeddedServerStatus(
                status="running",
                type="embedded",
                uptime_seconds=time.time() - _server_start_time,
                loaded_models=loaded_models,
                memory_used_gb=memory.get("active_gb", 0.0),
                memory_limit_gb=pool.max_memory_gb,
            )
        ]
    except RuntimeError:
        # Model pool not initialized yet
        return [
            EmbeddedServerStatus(
                status="not_initialized",
                type="embedded",
                uptime_seconds=time.time() - _server_start_time,
            )
        ]


@router.get("/models", response_model=list[LoadedModelInfo])
async def list_loaded_models(
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[LoadedModelInfo]:
    """List models currently loaded in the embedded server's model pool."""
    try:
        from mlx_manager.mlx_server.models.pool import get_model_pool

        pool = get_model_pool()
        models = []

        for model_id in pool.get_loaded_models():
            if model_id in pool._models:
                loaded = pool._models[model_id]
                models.append(
                    LoadedModelInfo(
                        model_id=loaded.model_id,
                        model_type=loaded.model_type,
                        size_gb=loaded.size_gb,
                        loaded_at=loaded.loaded_at,
                        last_used=loaded.last_used,
                        preloaded=loaded.preloaded,
                    )
                )

        return models
    except RuntimeError:
        return []


@router.get("/health", response_model=ServerHealthStatus)
async def check_server_health(
    current_user: Annotated[User, Depends(get_current_user)],
) -> ServerHealthStatus:
    """Check health of the embedded MLX Server.

    Returns health status based on model pool state and memory usage.
    """
    try:
        from mlx_manager.mlx_server.models.pool import get_model_pool
        from mlx_manager.mlx_server.utils.memory import get_memory_usage

        pool = get_model_pool()
        loaded_models = pool.get_loaded_models()
        memory = get_memory_usage()

        memory_used = memory.get("active_gb", 0.0)
        memory_limit = pool.max_memory_gb
        memory_available = max(0.0, memory_limit - memory_used)

        # Determine health status
        if memory_available < 1.0 and len(loaded_models) > 0:
            status = "degraded"  # Low memory but functional
        else:
            status = "healthy"

        return ServerHealthStatus(
            status=status,
            model_pool_initialized=True,
            loaded_model_count=len(loaded_models),
            memory_used_gb=memory_used,
            memory_available_gb=memory_available,
        )
    except RuntimeError:
        return ServerHealthStatus(
            status="unhealthy",
            model_pool_initialized=False,
        )


@router.get("/memory")
async def get_memory_status(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Get detailed memory status for the embedded MLX Server."""
    try:
        from mlx_manager.mlx_server.models.pool import get_model_pool
        from mlx_manager.mlx_server.utils.memory import get_memory_usage

        pool = get_model_pool()
        memory = get_memory_usage()

        return {
            "cache_gb": memory.get("cache_gb", 0.0),
            "active_gb": memory.get("active_gb", 0.0),
            "peak_gb": memory.get("peak_gb", 0.0),
            "limit_gb": pool.max_memory_gb,
            "available_gb": max(0.0, pool.max_memory_gb - memory.get("active_gb", 0.0)),
            "max_models": pool.max_models,
            "loaded_models": len(pool.get_loaded_models()),
        }
    except RuntimeError:
        return {
            "error": "Model pool not initialized",
            "cache_gb": 0.0,
            "active_gb": 0.0,
            "peak_gb": 0.0,
            "limit_gb": 0.0,
            "available_gb": 0.0,
            "max_models": 0,
            "loaded_models": 0,
        }


# Legacy endpoint stubs for backward compatibility
# These return appropriate responses indicating embedded mode


@router.post("/{profile_id}/start")
async def start_server(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_id: int,
) -> dict:
    """Legacy endpoint - embedded server is always running.

    Returns success since the embedded server handles model loading on-demand.
    """
    return {
        "message": "Embedded server is always running. Models are loaded on-demand.",
        "profile_id": profile_id,
        "pid": os.getpid(),
    }


@router.post("/{profile_id}/stop")
async def stop_server(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_id: int,
) -> dict:
    """Legacy endpoint - embedded server cannot be stopped.

    Individual models can be unloaded via the model pool management endpoints.
    """
    return {
        "message": "Embedded server cannot be stopped. Use /v1/admin/models to manage loaded models.",
        "profile_id": profile_id,
    }


@router.post("/{profile_id}/restart")
async def restart_server(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_id: int,
) -> dict:
    """Legacy endpoint - embedded server cannot be restarted.

    To reload a model, unload it via /v1/admin/models and it will reload on next request.
    """
    return {
        "message": "Embedded server cannot be restarted. Use /v1/admin/models to manage models.",
        "profile_id": profile_id,
    }
