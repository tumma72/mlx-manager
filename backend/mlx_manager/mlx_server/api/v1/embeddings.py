"""Embeddings API endpoint.

OpenAI-compatible endpoint for generating text embeddings.
Reference: https://platform.openai.com/docs/api-reference/embeddings
"""

import asyncio
import uuid

from fastapi import APIRouter, HTTPException
from loguru import logger

from mlx_manager.mlx_server.config import get_settings
from mlx_manager.mlx_server.errors import TimeoutHTTPException
from mlx_manager.mlx_server.models.detection import detect_model_type
from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.mlx_server.schemas.openai import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from mlx_manager.mlx_server.services.audit import audit_service
from mlx_manager.mlx_server.services.embeddings import generate_embeddings

router = APIRouter(tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for the input text(s).

    Accepts a single string or array of strings and returns embeddings
    in OpenAI-compatible format. Embeddings are L2-normalized.
    """
    request_id = f"emb-{uuid.uuid4().hex[:12]}"

    # Normalize input to list
    texts = [request.input] if isinstance(request.input, str) else list(request.input)

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    # Validate model type
    model_type = detect_model_type(request.model)
    if model_type != ModelType.EMBEDDINGS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is type '{model_type.value}', not 'embeddings'. "
            f"Use an embedding model (e.g., all-MiniLM-L6-v2).",
        )

    settings = get_settings()
    timeout = settings.timeout_embeddings_seconds

    async with audit_service.track_request(
        request_id=request_id,
        model=request.model,
        endpoint="/v1/embeddings",
        backend_type="local",
    ) as audit_ctx:
        try:
            # Generate embeddings with timeout
            embeddings_list, total_tokens = await asyncio.wait_for(
                generate_embeddings(
                    model_id=request.model,
                    texts=texts,
                ),
                timeout=timeout,
            )

            # Update audit context with token count
            audit_ctx.prompt_tokens = total_tokens
            audit_ctx.total_tokens = total_tokens

            # Build response
            return EmbeddingResponse(
                data=[
                    EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings_list)
                ],
                model=request.model,
                usage=EmbeddingUsage(
                    prompt_tokens=total_tokens,
                    total_tokens=total_tokens,
                ),
            )

        except TimeoutError:
            logger.warning(f"Embeddings generation timed out after {timeout}s")
            raise TimeoutHTTPException(
                timeout_seconds=timeout,
                detail=f"Embeddings generation timed out after {int(timeout)} seconds. "
                f"Consider reducing batch size or using a smaller model.",
            )
        except Exception as e:
            logger.exception(f"Embeddings generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e
