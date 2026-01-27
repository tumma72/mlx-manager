"""Models listing endpoint."""

from fastapi import APIRouter, HTTPException

from mlx_manager.mlx_server.config import get_settings
from mlx_manager.mlx_server.schemas.openai import ModelInfo, ModelListResponse

router = APIRouter(prefix="/v1", tags=["models"])


def get_available_models() -> list[str]:
    """Get list of available model IDs.

    Returns both:
    - Currently loaded models (hot) - from ModelPool (Plan 03)
    - Configured loadable models - from settings

    For now, returns configured models. Plan 03 will add loaded model detection.
    """
    settings = get_settings()

    # Start with configured available models
    model_ids = set(settings.available_models)

    # TODO: Plan 03 will add loaded models from ModelPoolManager
    # loaded_models = pool.get_loaded_models()
    # model_ids.update(loaded_models)

    return sorted(model_ids)


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List all available models.

    Returns models that are:
    - Currently loaded (hot) - ready for immediate inference
    - Available for loading (configured) - will be loaded on first request

    This satisfies API-04: /v1/models returns hot + loadable models.
    """
    model_ids = get_available_models()

    models = [ModelInfo(id=model_id, owned_by="mlx-community") for model_id in model_ids]

    return ModelListResponse(data=models)


@router.get("/models/{model_id:path}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """Get information about a specific model.

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")

    Returns:
        ModelInfo if model is available (loaded or loadable)

    Raises:
        404 if model is not in available_models list
    """
    available = get_available_models()
    if model_id not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available models: {available}",
        )

    return ModelInfo(id=model_id, owned_by="mlx-community")
