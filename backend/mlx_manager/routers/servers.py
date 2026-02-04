"""Servers API router.

With the embedded MLX Server, this router provides status information about
the model pool and loaded models. The start endpoint triggers model loading,
while stop/restart endpoints return informative messages since the server
is always running.
"""

import os
import time
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_manager.database import get_db
from mlx_manager.dependencies import get_current_user
from mlx_manager.mlx_server.models.pool import get_model_pool
from mlx_manager.models import ServerProfile, User

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


class RunningServer(BaseModel):
    """Running server info for UI compatibility."""

    profile_id: int
    profile_name: str
    pid: int
    port: int
    health_status: str  # "starting", "healthy", "unhealthy", "stopped"
    uptime_seconds: float
    memory_mb: float
    memory_percent: float
    memory_limit_percent: float = 0.0  # Memory as % of configured limit
    cpu_percent: float


@router.get("", response_model=list[RunningServer])
async def list_servers(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
) -> list[RunningServer]:
    """Return running servers list for UI compatibility.

    In embedded mode, a profile is "running" when its model is loaded in the
    model pool. Returns RunningServer objects for profiles with loaded models.
    """
    from sqlalchemy import select

    try:
        pool = get_model_pool()
        loaded_models = pool.get_loaded_models()

        if not loaded_models:
            return []

        # Get all profiles and find which ones have loaded models
        result = await db.execute(select(ServerProfile))
        profiles = result.scalars().all()

        memory_total_gb = pool.max_memory_gb

        running_servers: list[RunningServer] = []
        for profile in profiles:
            if profile.id is None:
                continue  # Skip profiles without IDs (shouldn't happen)
            if profile.model_path in loaded_models:
                # Get model info if available
                loaded_model = pool._models.get(profile.model_path)
                model_uptime = 0.0
                if loaded_model:
                    model_uptime = time.time() - loaded_model.loaded_at

                running_servers.append(
                    RunningServer(
                        profile_id=profile.id,
                        profile_name=profile.name,
                        pid=os.getpid(),
                        port=8080,  # Embedded server always on main port
                        health_status="healthy",
                        uptime_seconds=model_uptime,
                        memory_mb=loaded_model.size_gb * 1024 if loaded_model else 0.0,
                        memory_percent=(
                            (loaded_model.size_gb / memory_total_gb * 100)
                            if memory_total_gb > 0 and loaded_model
                            else 0.0
                        ),
                        memory_limit_percent=(
                            (loaded_model.size_gb / pool.max_memory_gb * 100)
                            if pool.max_memory_gb > 0 and loaded_model
                            else 0.0
                        ),
                        cpu_percent=0.0,  # Not tracked in embedded mode
                    )
                )

        return running_servers

    except RuntimeError:
        # Model pool not initialized
        return []


@router.get("/embedded", response_model=EmbeddedServerStatus)
async def get_embedded_status(
    current_user: Annotated[User, Depends(get_current_user)],
) -> EmbeddedServerStatus:
    """Return embedded MLX Server status.

    With embedded mode, there's always exactly one server (the application itself).
    This endpoint provides the actual status of the embedded server.
    """
    try:
        from mlx_manager.mlx_server.models.pool import get_model_pool
        from mlx_manager.mlx_server.utils.memory import get_memory_usage

        pool = get_model_pool()
        loaded_models = pool.get_loaded_models()
        memory = get_memory_usage()

        return EmbeddedServerStatus(
            status="running",
            type="embedded",
            uptime_seconds=time.time() - _server_start_time,
            loaded_models=loaded_models,
            memory_used_gb=memory.get("active_gb", 0.0),
            memory_limit_gb=pool.max_memory_gb,
        )
    except RuntimeError:
        # Model pool not initialized yet
        return EmbeddedServerStatus(
            status="not_initialized",
            type="embedded",
            uptime_seconds=time.time() - _server_start_time,
        )


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
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict:
    """Load the profile's model into the embedded server.

    Triggers model loading in the background. The model will be available
    for inference once loading completes. Use the health endpoint to check
    if the model is loaded.
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool

    # Get the profile to find which model it uses
    profile = await db.get(ServerProfile, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    model_id = profile.model_path

    try:
        pool = get_model_pool()

        # Check if already loaded
        if pool.is_loaded(model_id):
            return {"status": "already_loaded", "model": model_id, "pid": os.getpid()}

        # Start loading in background
        async def load_model() -> None:
            try:
                await pool.get_model(model_id)
                logger.info(f"Model loaded successfully: {model_id}")
            except Exception as e:
                logger.exception(f"Failed to load model {model_id}: {e}")

        background_tasks.add_task(load_model)
        return {"status": "loading", "model": model_id, "pid": os.getpid()}

    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Model pool not initialized. Server may still be starting.",
        )


@router.post("/{profile_id}/stop")
async def stop_server(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Unload model associated with profile from memory.

    In embedded mode, "stopping" a profile means unloading its model from the pool.
    Preloaded models are protected and cannot be unloaded.
    """
    # Get the profile to find which model it uses
    profile = await db.get(ServerProfile, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        pool = get_model_pool()
        model_id = profile.model_path

        # Check if model is loaded
        if not pool.is_loaded(model_id):
            return {
                "success": True,
                "message": f"Model {model_id} is not currently loaded",
            }

        # Check if model is preloaded (protected)
        loaded = pool._models.get(model_id)
        if loaded and loaded.preloaded:
            return {
                "success": False,
                "message": f"Model {model_id} is preloaded and protected from unload",
            }

        # Unload the model
        await pool.unload_model(model_id)

        return {
            "success": True,
            "message": f"Model {model_id} unloaded successfully",
        }
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Model pool not initialized. Server may still be starting.",
        )


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


class ProfileServerStatus(BaseModel):
    """Server status for a profile - for frontend polling compatibility.

    In embedded mode, the server is always running. This response format
    matches what the frontend expects from the old profile-based server model.
    """

    profile_id: int
    running: bool = True  # Embedded server is always running
    pid: int | None = None
    exit_code: int | None = None
    failed: bool = False
    error_message: str | None = None


@router.get("/{profile_id}/status", response_model=ProfileServerStatus)
async def get_server_status(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_id: int,
) -> ProfileServerStatus:
    """Get server status for a profile.

    In embedded mode, the server is always running. Models are loaded on-demand
    when a request is made. This endpoint returns a status that indicates the
    embedded server is operational.

    The frontend uses this endpoint to poll for server startup completion.
    With embedded mode, startup is always "complete" since the server is embedded.
    """
    return ProfileServerStatus(
        profile_id=profile_id,
        running=True,
        pid=os.getpid(),
        exit_code=None,
        failed=False,
        error_message=None,
    )


class ProfileHealthStatus(BaseModel):
    """Health status for a profile - for frontend polling compatibility.

    In embedded mode, the server is always healthy. The model_loaded field
    indicates whether the profile's model is currently loaded in the pool.
    """

    status: str  # "healthy", "unhealthy", "starting", "stopped"
    response_time_ms: int | None = None
    model_loaded: bool = False
    error: str | None = None


@router.get("/{profile_id}/health", response_model=ProfileHealthStatus)
async def get_server_health(
    current_user: Annotated[User, Depends(get_current_user)],
    profile_id: int,
    db: AsyncSession = Depends(get_db),
) -> ProfileHealthStatus:
    """Check if profile's model is loaded in the embedded server.

    Returns health status based on whether the profile's specific model
    is currently loaded in the model pool.
    """
    # Get the profile to find which model it uses
    profile = await db.get(ServerProfile, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    model_id = profile.model_path

    try:
        from mlx_manager.mlx_server.models.pool import get_model_pool

        pool = get_model_pool()
        is_loaded = pool.is_loaded(model_id)

        return ProfileHealthStatus(
            status="healthy",
            model_loaded=is_loaded,
            response_time_ms=1,
            error=None,
        )
    except RuntimeError:
        return ProfileHealthStatus(
            status="unhealthy",
            model_loaded=False,
            response_time_ms=0,
            error="Model pool not initialized",
        )
