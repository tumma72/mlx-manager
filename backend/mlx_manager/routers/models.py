"""Models API router."""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from mlx_manager.database import get_db
from mlx_manager.dependencies import get_current_user
from mlx_manager.models import Download, LocalModel, ModelSearchResult, User
from mlx_manager.services.hf_client import hf_client
from mlx_manager.utils.model_detection import get_model_detection_info

router = APIRouter(prefix="/api/models", tags=["models"])


class DownloadRequest(BaseModel):
    """Request body for model download."""

    model_id: str


# Store for download tasks
download_tasks: dict[str, dict] = {}


@router.get("/search", response_model=list[ModelSearchResult])
async def search_models(
    current_user: Annotated[User, Depends(get_current_user)],
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
async def list_local_models(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """List locally downloaded MLX models."""
    return hf_client.list_local_models()


@router.post("/download")
async def start_download(
    current_user: Annotated[User, Depends(get_current_user)],
    request: DownloadRequest,
    db: AsyncSession = Depends(get_db),
):
    """Start downloading a model from HuggingFace."""
    task_id = str(uuid.uuid4())

    # Check if there's already an active download for this model
    result = await db.execute(
        select(Download).where(
            Download.model_id == request.model_id,
            Download.status.in_(["pending", "downloading"]),  # type: ignore[attr-defined]
        )
    )
    existing = result.scalars().first()

    if existing:
        # Return existing download's task_id if one exists
        existing_task_id = None
        for tid, task in download_tasks.items():
            if task.get("model_id") == request.model_id:
                existing_task_id = tid
                break
        if existing_task_id:
            return {"task_id": existing_task_id, "model_id": request.model_id}

    # Create DB record for the download
    download_record = Download(
        model_id=request.model_id,
        status="pending",
        started_at=datetime.now(tz=UTC),
    )
    db.add(download_record)
    await db.flush()  # Get the ID

    # Store task info with download_id reference
    download_tasks[task_id] = {
        "model_id": request.model_id,
        "download_id": download_record.id,
        "status": "starting",
        "progress": 0,
    }

    return {"task_id": task_id, "model_id": request.model_id}


@router.get("/download/{task_id}/progress")
async def get_download_progress(
    current_user: Annotated[User, Depends(get_current_user)],
    task_id: str,
):
    """SSE endpoint for download progress."""

    async def generate() -> AsyncGenerator[str, None]:
        if task_id not in download_tasks:
            yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
            return

        task = download_tasks[task_id]
        model_id = task["model_id"]
        download_id = task.get("download_id")
        last_db_update = time.time()

        try:
            async for progress in hf_client.download_model(model_id):
                download_tasks[task_id].update(progress)

                # Serialize progress dict properly
                progress_dict = {
                    "status": progress.get("status"),
                    "progress": progress.get("progress", 0),
                    "downloaded_bytes": progress.get("downloaded_bytes", 0),
                    "total_bytes": progress.get("total_bytes", 0),
                    "error": progress.get("error"),
                }
                yield f"data: {json.dumps(progress_dict)}\n\n"

                # Update DB record periodically (every 5 seconds) or on status change
                current_time = time.time()
                status = progress.get("status")
                is_final = status in ("completed", "failed")

                if download_id and (is_final or current_time - last_db_update >= 5):
                    await _update_download_record(
                        download_id,
                        status=status or "downloading",
                        downloaded_bytes=progress.get("downloaded_bytes", 0),
                        total_bytes=progress.get("total_bytes"),
                        error=progress.get("error"),
                        completed=is_final and status == "completed",
                    )
                    last_db_update = current_time

                if is_final:
                    break
        except Exception as e:
            error_msg = str(e)
            yield f"data: {json.dumps({'status': 'failed', 'error': error_msg})}\n\n"
            # Update DB with failure
            if download_id:
                await _update_download_record(download_id, status="failed", error=error_msg)
        finally:
            # Clean up task after a delay
            await asyncio.sleep(60)
            download_tasks.pop(task_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/download/{task_id}/status")
async def get_download_status(
    current_user: Annotated[User, Depends(get_current_user)],
    task_id: str,
) -> dict:
    """Get download status without SSE (polling fallback).

    Use this endpoint for debugging or as a fallback when SSE is not available.
    Returns the current status from the in-memory task store.
    """
    if task_id not in download_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = download_tasks[task_id]
    return {
        "task_id": task_id,
        "model_id": task.get("model_id"),
        "status": task.get("status"),
        "progress": task.get("progress", 0),
        "downloaded_bytes": task.get("downloaded_bytes", 0),
        "total_bytes": task.get("total_bytes", 0),
        "error": task.get("error"),
    }


async def _update_download_record(
    download_id: int,
    status: str,
    downloaded_bytes: int = 0,
    total_bytes: int | None = None,
    error: str | None = None,
    completed: bool = False,
) -> None:
    """Update a download record in the database."""
    from mlx_manager.database import get_session

    async with get_session() as session:
        result = await session.execute(select(Download).where(Download.id == download_id))
        download = result.scalars().first()
        if download:
            download.status = status
            download.downloaded_bytes = downloaded_bytes
            if total_bytes is not None:
                download.total_bytes = total_bytes
            if error:
                download.error = error
            if completed:
                download.completed_at = datetime.now(tz=UTC)
            session.add(download)
            await session.commit()


@router.get("/downloads/active")
async def get_active_downloads(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Get all active/in-progress downloads with their task IDs.

    Returns downloads that are currently in progress or pending.
    The frontend can use this to reconnect to SSE streams after navigation.
    Combines in-memory tasks with DB-backed downloads for persistence across restarts.
    """
    active = []
    seen_model_ids = set()

    # First, get in-memory tasks (these are authoritative for active connections)
    for task_id, task in download_tasks.items():
        status = task.get("status", "")
        model_id = task.get("model_id")
        if status in ("starting", "pending", "downloading") and model_id:
            seen_model_ids.add(model_id)
            active.append(
                {
                    "task_id": task_id,
                    "model_id": model_id,
                    "status": status,
                    "progress": task.get("progress", 0),
                    "downloaded_bytes": task.get("downloaded_bytes", 0),
                    "total_bytes": task.get("total_bytes", 0),
                }
            )

    # Also include DB-backed downloads that aren't in memory
    # (these are downloads that need to be resumed)
    result = await db.execute(
        select(Download).where(
            Download.status.in_(["pending", "downloading"])  # type: ignore[attr-defined]
        )
    )
    db_downloads = result.scalars().all()

    for download in db_downloads:
        if download.model_id not in seen_model_ids:
            # This is a download from a previous server session that needs resuming
            # Generate a new task_id for it
            task_id = f"resume-{download.id}"
            active.append(
                {
                    "task_id": task_id,
                    "model_id": download.model_id,
                    "status": download.status,
                    "progress": (
                        int((download.downloaded_bytes / download.total_bytes) * 100)
                        if download.total_bytes
                        else 0
                    ),
                    "downloaded_bytes": download.downloaded_bytes,
                    "total_bytes": download.total_bytes or 0,
                    "needs_resume": True,  # Signal to frontend this needs resuming
                }
            )

    return active


@router.delete("/{model_id:path}")
async def delete_model(
    current_user: Annotated[User, Depends(get_current_user)],
    model_id: str,
):
    """Delete a local model."""
    success = await hf_client.delete_model(model_id)

    if not success:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"deleted": True}


@router.get("/config/{model_id:path}")
async def get_model_config(
    current_user: Annotated[User, Depends(get_current_user)],
    model_id: str,
    tags: str | None = Query(None, description="Comma-separated HuggingFace tags"),
):
    """Get model characteristics from config.json.

    For local models: reads from HuggingFace cache.
    For remote models: fetches via HuggingFace resolve API.

    Returns:
        ModelCharacteristics with architecture, context window, quantization, etc.
        204 No Content if config is not available.
    """
    from mlx_manager.services.hf_api import fetch_remote_config
    from mlx_manager.utils.model_detection import (
        extract_characteristics,
        extract_characteristics_from_model,
    )

    # Parse tags into list
    tag_list = tags.split(",") if tags else None

    # Try local first
    chars = extract_characteristics_from_model(model_id, tags=tag_list)
    if chars:
        return chars

    # Fetch remote
    config = await fetch_remote_config(model_id)
    if config:
        return extract_characteristics(config, tags=tag_list)

    # Return 204 No Content instead of 404 to avoid browser console errors
    return Response(status_code=204)


@router.get("/detect-options/{model_id:path}")
async def detect_model_options(
    current_user: Annotated[User, Depends(get_current_user)],
    model_id: str,
):
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
async def get_available_parsers(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Get list of available parser options for dropdowns.

    DEPRECATED: Parser options are no longer used with the embedded MLX Server.
    Returns empty list for backwards compatibility.
    """
    return {"parsers": []}
