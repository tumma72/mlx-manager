"""v1 API router."""

from fastapi import APIRouter

from mlx_manager.mlx_server.api.v1.models import router as models_router

v1_router = APIRouter()
v1_router.include_router(models_router)

__all__ = ["v1_router"]
