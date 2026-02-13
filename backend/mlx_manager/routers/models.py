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
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from mlx_manager.database import get_db
from mlx_manager.dependencies import get_current_user, get_current_user_from_token
from mlx_manager.models import (
    Download,
    LocalModel,
    Model,
    ModelResponse,
    ModelSearchResult,
    User,
)
from mlx_manager.models.capabilities import capabilities_to_response
from mlx_manager.models.dto.models import DownloadRequest
from mlx_manager.models.enums import DownloadStatusEnum
from mlx_manager.services.hf_client import (
    cleanup_cancel_event,
    cleanup_partial_download,
    hf_client,
    register_cancel_event,
    request_cancel,
)
from mlx_manager.utils.model_detection import get_model_detection_info

router = APIRouter(prefix="/api/models", tags=["models"])


def _serialize_model_response(model: Model) -> ModelResponse:
    """Build a ModelResponse from a Model with its capabilities relationship."""
    caps_response = None
    if model.capabilities:
        caps_response = capabilities_to_response(model.capabilities)

    return ModelResponse(
        id=model.id,  # type: ignore[arg-type]
        repo_id=model.repo_id,
        model_type=model.model_type,
        local_path=model.local_path,
        size_bytes=model.size_bytes,
        size_gb=round(model.size_bytes / (1024**3), 2) if model.size_bytes else None,
        downloaded_at=model.downloaded_at,
        last_used_at=model.last_used_at,
        capabilities=caps_response,
    )


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


@router.get("/downloaded", response_model=list[ModelResponse])
async def list_downloaded_models(
    current_user: Annotated[User, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
):
    """List all downloaded models from the unified Model table."""
    from sqlalchemy.orm import selectinload
    from sqlmodel import col, desc

    result = await session.execute(
        select(Model)
        .options(selectinload(Model.capabilities))  # type: ignore[arg-type]
        .order_by(desc(col(Model.downloaded_at)))
    )
    models = result.scalars().all()
    return [_serialize_model_response(m) for m in models]


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
    current_user: Annotated[User, Depends(get_current_user_from_token)],
    task_id: str,
):
    """SSE endpoint for download progress.

    Uses query-parameter JWT auth (?token=<jwt>) because browser EventSource
    cannot send custom Authorization headers.
    """

    async def generate() -> AsyncGenerator[str, None]:
        if task_id not in download_tasks:
            yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
            return

        task = download_tasks[task_id]
        model_id = task["model_id"]
        download_id = task.get("download_id")
        last_db_update = time.time()

        # Register cancel event so pause/cancel endpoints can signal this download
        cancel_event = None
        if download_id:
            cancel_event = register_cancel_event(str(download_id))

        try:
            async for progress in hf_client.download_model(model_id, cancel_event=cancel_event):
                download_tasks[task_id].update(progress.model_dump(exclude_none=True))

                # Serialize progress dict properly
                progress_dict = {
                    "status": progress.status,
                    "progress": progress.progress or 0,
                    "downloaded_bytes": progress.downloaded_bytes or 0,
                    "total_bytes": progress.total_bytes or 0,
                    "error": progress.error,
                }
                yield f"data: {json.dumps(progress_dict)}\n\n"

                # Update DB record periodically (every 5 seconds) or on status change
                current_time = time.time()
                status = progress.status
                is_final = status in ("completed", "failed", "cancelled")

                if download_id and (is_final or current_time - last_db_update >= 5):
                    await _update_download_record(
                        download_id,
                        status=status or "downloading",
                        downloaded_bytes=progress.downloaded_bytes or 0,
                        total_bytes=progress.total_bytes,
                        error=progress.error,
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
            # Clean up cancel event
            if download_id:
                cleanup_cancel_event(str(download_id))
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
            download.status = DownloadStatusEnum(status)
            download.downloaded_bytes = downloaded_bytes
            if total_bytes is not None:
                download.total_bytes = total_bytes
            if error:
                download.error = error
            if completed:
                download.completed_at = datetime.now(tz=UTC)
                # Register in unified Model table
                from mlx_manager.services.hf_client import hf_client
                from mlx_manager.services.model_registry import register_model_from_download

                local_path = hf_client.get_local_path(download.model_id)
                await register_model_from_download(
                    repo_id=download.model_id,
                    local_path=local_path or "",
                    size_bytes=total_bytes,
                )
            session.add(download)
            await session.commit()


@router.get("/downloads/active")
async def get_active_downloads(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Get all active/in-progress downloads with their task IDs.

    Returns downloads that are currently in progress, pending, or paused.
    The frontend can use this to reconnect to SSE streams after navigation,
    and to show paused downloads with Resume buttons.
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
                    "download_id": task.get("download_id"),
                    "status": status,
                    "progress": task.get("progress", 0),
                    "downloaded_bytes": task.get("downloaded_bytes", 0),
                    "total_bytes": task.get("total_bytes", 0),
                }
            )

    # Also include DB-backed downloads that aren't in memory
    # (these are downloads that need to be resumed, or paused downloads)
    result = await db.execute(
        select(Download).where(
            Download.status.in_(["pending", "downloading", "paused"])  # type: ignore[attr-defined]
        )
    )
    db_downloads = result.scalars().all()

    for download in db_downloads:
        if download.model_id not in seen_model_ids:
            # This is a download from a previous server session that needs resuming
            # or a paused download that should show in the UI
            task_id = f"resume-{download.id}"
            active.append(
                {
                    "task_id": task_id,
                    "download_id": download.id,
                    "model_id": download.model_id,
                    "status": download.status,
                    "progress": (
                        int((download.downloaded_bytes / download.total_bytes) * 100)
                        if download.total_bytes
                        else 0
                    ),
                    "downloaded_bytes": download.downloaded_bytes,
                    "total_bytes": download.total_bytes or 0,
                    "needs_resume": download.status != "paused",
                }
            )

    return active


@router.post("/download/{download_id}/pause")
async def pause_download(
    current_user: Annotated[User, Depends(get_current_user)],
    download_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Pause an active download.

    Signals the download thread to stop via cancel event.
    HF Hub leaves .incomplete files in place for later resume.
    """
    result = await db.execute(select(Download).where(Download.id == download_id))
    download = result.scalars().first()
    if not download:
        raise HTTPException(status_code=404, detail="Download not found")

    if download.status != "downloading":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot pause download in '{download.status}' state",
        )

    # Signal the download thread to stop
    request_cancel(str(download_id))

    # Update status to paused
    download.status = DownloadStatusEnum.PAUSED
    db.add(download)
    await db.flush()

    return {
        "id": download.id,
        "model_id": download.model_id,
        "status": "paused",
        "downloaded_bytes": download.downloaded_bytes,
        "total_bytes": download.total_bytes,
    }


@router.post("/download/{download_id}/resume")
async def resume_download(
    current_user: Annotated[User, Depends(get_current_user)],
    download_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Resume a paused download.

    Creates a new background task. HF Hub's snapshot_download()
    automatically resumes from .incomplete files.
    """
    result = await db.execute(select(Download).where(Download.id == download_id))
    download = result.scalars().first()
    if not download:
        raise HTTPException(status_code=404, detail="Download not found")

    if download.status != "paused":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot resume download in '{download.status}' state",
        )

    # Update status to downloading
    download.status = DownloadStatusEnum.DOWNLOADING
    db.add(download)
    await db.flush()

    # Generate a new task_id for SSE reconnection
    task_id = str(uuid.uuid4())
    download_tasks[task_id] = {
        "model_id": download.model_id,
        "download_id": download.id,
        "status": "downloading",
        "progress": (
            int((download.downloaded_bytes / download.total_bytes) * 100)
            if download.total_bytes
            else 0
        ),
        "downloaded_bytes": download.downloaded_bytes,
        "total_bytes": download.total_bytes or 0,
    }

    return {
        "task_id": task_id,
        "model_id": download.model_id,
        "download_id": download.id,
        "progress": download_tasks[task_id]["progress"],
        "downloaded_bytes": download.downloaded_bytes,
        "total_bytes": download.total_bytes or 0,
    }


@router.post("/download/{download_id}/cancel")
async def cancel_download(
    current_user: Annotated[User, Depends(get_current_user)],
    download_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Cancel a download and clean up partial files.

    Works for both active (downloading) and paused downloads.
    Removes partial files from HF cache.
    """
    result = await db.execute(select(Download).where(Download.id == download_id))
    download = result.scalars().first()
    if not download:
        raise HTTPException(status_code=404, detail="Download not found")

    if download.status not in ("downloading", "paused", "pending"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel download in '{download.status}' state",
        )

    # Signal active download to stop
    if download.status == "downloading":
        request_cancel(str(download_id))

    # Update status to cancelled
    download.status = DownloadStatusEnum.CANCELLED
    download.completed_at = datetime.now(tz=UTC)
    db.add(download)
    await db.flush()

    # Clean up partial files from HF cache
    cleanup_success = False
    try:
        cleanup_success = cleanup_partial_download(download.model_id)
    except Exception as e:
        # Log but don't fail - download is already marked cancelled
        from loguru import logger

        logger.error(f"Failed to clean up partial download for {download.model_id}: {e}")

    return {
        "id": download.id,
        "model_id": download.model_id,
        "status": "cancelled",
        "cleanup_success": cleanup_success,
    }


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


@router.get("/probe/supported-types")
async def get_probeable_types(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Return model types that have a registered probe strategy."""
    from mlx_manager.services.probe import registered_model_types

    return [mt.value for mt in registered_model_types()]


@router.get("/capabilities", response_model=list[ModelResponse])
async def get_all_capabilities(
    current_user: Annotated[User, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
):
    """Get all stored model capabilities (only models that have been probed)."""
    from sqlalchemy.orm import selectinload

    from mlx_manager.models.capabilities import ModelCapabilities

    result = await session.execute(
        select(Model)
        .join(ModelCapabilities, Model.id == ModelCapabilities.model_id)  # type: ignore[arg-type]
        .options(selectinload(Model.capabilities))  # type: ignore[arg-type]
    )
    models = result.scalars().all()
    return [_serialize_model_response(m) for m in models]


@router.get("/capabilities/{model_id:path}", response_model=ModelResponse | None)
async def get_model_capabilities(
    model_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
):
    """Get stored capabilities for a specific model."""
    from sqlalchemy.orm import selectinload

    result = await session.execute(
        select(Model).where(Model.repo_id == model_id).options(selectinload(Model.capabilities))  # type: ignore[arg-type]
    )
    model = result.scalar_one_or_none()
    if not model:
        return None
    return _serialize_model_response(model)


@router.post("/probe/{model_id:path}")
async def probe_model_capabilities(
    model_id: str,
    current_user: Annotated[User, Depends(get_current_user_from_token)],
):
    """Probe a model's capabilities (SSE stream).

    Loads the model temporarily, tests for native tool support,
    thinking support, and practical max tokens. Results are stored
    in the database for future use.

    Requires JWT token as query parameter (same pattern as download progress).
    """
    from mlx_manager.services.probe import probe_model

    async def generate():
        async for step in probe_model(model_id):
            yield step.to_sse()
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
