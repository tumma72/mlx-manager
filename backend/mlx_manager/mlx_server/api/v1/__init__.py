"""v1 API router."""

from fastapi import APIRouter

from mlx_manager.mlx_server.api.v1.admin import router as admin_router
from mlx_manager.mlx_server.api.v1.chat import router as chat_router
from mlx_manager.mlx_server.api.v1.completions import router as completions_router
from mlx_manager.mlx_server.api.v1.embeddings import router as embeddings_router
from mlx_manager.mlx_server.api.v1.messages import router as messages_router
from mlx_manager.mlx_server.api.v1.models import router as models_router

v1_router = APIRouter()
v1_router.include_router(models_router)
v1_router.include_router(chat_router)
v1_router.include_router(completions_router)
v1_router.include_router(embeddings_router)
v1_router.include_router(messages_router)
v1_router.include_router(admin_router)

__all__ = ["v1_router"]
