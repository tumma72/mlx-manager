"""Admin API endpoints for model pool management.

These endpoints allow administrators to:
- Preload models (protected from LRU eviction)
- Unload models to free memory
- Monitor pool status and memory usage
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mlx_manager.mlx_server.models.pool import get_model_pool
from mlx_manager.mlx_server.utils.memory import get_memory_usage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# --- Response Models ---


class ModelStatus(BaseModel):
    """Status of a loaded model."""

    model_id: str
    model_type: str
    size_gb: float
    preloaded: bool
    last_used: float
    loaded_at: float


class PoolStatusResponse(BaseModel):
    """Model pool status response."""

    loaded_models: list[ModelStatus]
    total_models: int
    memory: dict[str, Any]
    max_memory_gb: float
    max_models: int


class ModelLoadResponse(BaseModel):
    """Response for model load operation."""

    status: str
    model_id: str
    model_type: str
    size_gb: float
    preloaded: bool


class ModelUnloadResponse(BaseModel):
    """Response for model unload operation."""

    status: str
    model_id: str


# --- Endpoints ---


@router.get("/models/status", response_model=PoolStatusResponse)
async def pool_status() -> PoolStatusResponse:
    """Get current model pool status.

    Returns list of loaded models with their metadata, memory usage,
    and pool configuration.
    """
    pool = get_model_pool()
    memory = get_memory_usage()

    models = [
        ModelStatus(
            model_id=model_id,
            model_type=m.model_type,
            size_gb=m.size_gb,
            preloaded=m.preloaded,
            last_used=m.last_used,
            loaded_at=m.loaded_at,
        )
        for model_id, m in pool._models.items()
    ]

    return PoolStatusResponse(
        loaded_models=models,
        total_models=len(models),
        memory=memory,
        max_memory_gb=pool.max_memory_gb,
        max_models=pool.max_models,
    )


@router.post("/models/load/{model_id:path}", response_model=ModelLoadResponse)
async def preload_model(model_id: str) -> ModelLoadResponse:
    """Preload a model into the pool.

    Preloaded models are protected from LRU eviction. Use this to ensure
    a model stays loaded even under memory pressure.

    Args:
        model_id: HuggingFace model ID (e.g., mlx-community/Llama-3.2-3B-4bit)
    """
    pool = get_model_pool()

    logger.info(f"Admin: Preloading model {model_id}")

    try:
        loaded = await pool.preload_model(model_id)

        return ModelLoadResponse(
            status="loaded",
            model_id=model_id,
            model_type=loaded.model_type,
            size_gb=loaded.size_gb,
            preloaded=True,
        )

    except Exception as e:
        logger.error(f"Admin: Failed to preload {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/unload/{model_id:path}", response_model=ModelUnloadResponse)
async def unload_model(model_id: str) -> ModelUnloadResponse:
    """Unload a model from the pool.

    Frees memory by removing the model from the pool. This works for both
    preloaded and on-demand loaded models.

    Args:
        model_id: HuggingFace model ID to unload
    """
    pool = get_model_pool()

    logger.info(f"Admin: Unloading model {model_id}")

    success = await pool.unload_model(model_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Model not loaded: {model_id}",
        )

    return ModelUnloadResponse(
        status="unloaded",
        model_id=model_id,
    )


@router.get("/health")
async def admin_health() -> dict[str, str]:
    """Admin health check endpoint."""
    return {"status": "healthy"}
