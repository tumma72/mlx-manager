"""Models API router."""

import asyncio
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_manager.models import LocalModel, ModelSearchResult
from mlx_manager.services.hf_client import hf_client
from mlx_manager.services.parser_options import get_parser_options
from mlx_manager.utils.model_detection import get_model_detection_info

router = APIRouter(prefix="/api/models", tags=["models"])


class DownloadRequest(BaseModel):
    """Request body for model download."""

    model_id: str


# Store for download tasks
download_tasks: dict[str, dict] = {}


@router.get("/search", response_model=list[ModelSearchResult])
async def search_models(
    query: str = Query(..., min_length=1),
    max_size_gb: float | None = None,
    limit: int = Query(default=20, le=100),
):
    """Search MLX models on HuggingFace."""
    try:
        results = await hf_client.search_mlx_models(
            query=query, max_size_gb=max_size_gb, limit=limit
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/local", response_model=list[LocalModel])
async def list_local_models():
    """List locally downloaded MLX models."""
    return hf_client.list_local_models()


@router.post("/download")
async def start_download(request: DownloadRequest):
    """Start downloading a model from HuggingFace."""
    task_id = str(uuid.uuid4())

    # Store task info
    download_tasks[task_id] = {
        "model_id": request.model_id,
        "status": "starting",
        "progress": 0,
    }

    return {"task_id": task_id, "model_id": request.model_id}


@router.get("/download/{task_id}/progress")
async def get_download_progress(task_id: str):
    """SSE endpoint for download progress."""

    async def generate() -> AsyncGenerator[str, None]:
        if task_id not in download_tasks:
            yield "data: {'error': 'Task not found'}\n\n"
            return

        task = download_tasks[task_id]
        model_id = task["model_id"]

        try:
            async for progress in hf_client.download_model(model_id):
                download_tasks[task_id].update(progress)
                import json

                yield f"data: {json.dumps(progress)}\n\n"

                if progress["status"] in ("completed", "failed"):
                    break
        except Exception as e:
            import json

            yield f"data: {json.dumps({'status': 'failed', 'error': str(e)})}\n\n"
        finally:
            # Clean up task after a delay
            await asyncio.sleep(60)
            download_tasks.pop(task_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.delete("/{model_id:path}")
async def delete_model(model_id: str):
    """Delete a local model."""
    success = await hf_client.delete_model(model_id)

    if not success:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"deleted": True}


@router.get("/detect-options/{model_id:path}")
async def detect_model_options(model_id: str):
    """
    Detect recommended parser options for a model.

    This endpoint works OFFLINE - it reads config.json from the local
    HuggingFace cache or falls back to model path name matching.

    Returns:
        model_family: Detected model family (minimax, qwen, glm) or null
        recommended_options: Parser options for the model family
        is_downloaded: Whether the model is locally available
    """
    return get_model_detection_info(model_id)


@router.get("/available-parsers")
async def get_available_parsers():
    """
    Get list of available parser options for dropdowns.

    DEPRECATED: Use /api/system/parser-options instead for separate
    tool_call_parsers, reasoning_parsers, and message_converters lists.

    Returns a combined list of all parser identifiers for backwards
    compatibility.
    """
    options = get_parser_options()
    # Combine all unique parsers for backwards compatibility
    all_parsers = set(options["tool_call_parsers"])
    all_parsers.update(options["reasoning_parsers"])
    all_parsers.update(options["message_converters"])
    return {"parsers": sorted(all_parsers)}
