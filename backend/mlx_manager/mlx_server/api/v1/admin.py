"""Admin API endpoints for model pool management and audit logs.

These endpoints allow administrators to:
- Preload models (protected from LRU eviction)
- Unload models to free memory
- Monitor pool status and memory usage
- View and export audit logs
- Subscribe to live audit log updates via WebSocket
"""

import asyncio
import csv
import io
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from mlx_manager.mlx_server.database import get_session
from mlx_manager.mlx_server.models.audit import AuditLog, AuditLogResponse
from mlx_manager.mlx_server.models.pool import get_model_pool
from mlx_manager.mlx_server.services.audit import audit_service
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


# ============================================================================
# Audit Log Endpoints
# ============================================================================


@router.get("/audit-logs", response_model=list[AuditLogResponse])
async def get_audit_logs(
    model: str | None = Query(default=None, description="Filter by model"),
    backend_type: str | None = Query(default=None, description="Filter by backend type"),
    status: str | None = Query(default=None, description="Filter by status"),
    start_time: datetime | None = Query(default=None, description="Start of time range"),
    end_time: datetime | None = Query(default=None, description="End of time range"),
    limit: int = Query(default=100, le=1000, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> list[AuditLogResponse]:
    """Get audit logs with optional filtering.

    Supports filtering by:
    - model: Exact model name match
    - backend_type: local, openai, anthropic
    - status: success, error, timeout
    - start_time/end_time: Time range filter

    Returns most recent first (descending timestamp).
    """
    async with get_session() as session:
        query = select(AuditLog)

        # Apply filters
        if model:
            query = query.where(AuditLog.model == model)
        if backend_type:
            query = query.where(AuditLog.backend_type == backend_type)
        if status:
            query = query.where(AuditLog.status == status)
        if start_time:
            query = query.where(AuditLog.timestamp >= start_time)
        if end_time:
            query = query.where(AuditLog.timestamp <= end_time)

        # Order by timestamp descending (most recent first)
        query = query.order_by(AuditLog.timestamp.desc())

        # Apply pagination
        query = query.offset(offset).limit(limit)

        result = await session.execute(query)
        logs = result.scalars().all()

        return [AuditLogResponse.model_validate(log) for log in logs]


@router.get("/audit-logs/stats")
async def get_audit_stats() -> dict[str, Any]:
    """Get aggregate statistics for audit logs."""
    async with get_session() as session:
        # Total count
        total_result = await session.execute(select(func.count(AuditLog.id)))
        total = total_result.scalar() or 0

        # Count by status
        status_result = await session.execute(
            select(AuditLog.status, func.count(AuditLog.id)).group_by(AuditLog.status)
        )
        by_status = dict(status_result.all())

        # Count by backend
        backend_result = await session.execute(
            select(AuditLog.backend_type, func.count(AuditLog.id)).group_by(AuditLog.backend_type)
        )
        by_backend = dict(backend_result.all())

        # Unique models
        models_result = await session.execute(select(func.count(func.distinct(AuditLog.model))))
        unique_models = models_result.scalar() or 0

        return {
            "total_requests": total,
            "by_status": by_status,
            "by_backend": by_backend,
            "unique_models": unique_models,
        }


@router.get("/audit-logs/export")
async def export_audit_logs(
    model: str | None = Query(default=None),
    backend_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    format: str = Query(default="jsonl", description="Export format: jsonl or csv"),
) -> PlainTextResponse:
    """Export audit logs in JSONL or CSV format.

    JSONL format is standard and easily parseable by log tools.
    CSV format is useful for spreadsheet analysis.
    """
    async with get_session() as session:
        query = select(AuditLog)

        if model:
            query = query.where(AuditLog.model == model)
        if backend_type:
            query = query.where(AuditLog.backend_type == backend_type)
        if status:
            query = query.where(AuditLog.status == status)
        if start_time:
            query = query.where(AuditLog.timestamp >= start_time)
        if end_time:
            query = query.where(AuditLog.timestamp <= end_time)

        query = query.order_by(AuditLog.timestamp.desc())
        result = await session.execute(query)
        logs = result.scalars().all()

        if format == "csv":
            output = io.StringIO()
            if logs:
                fieldnames = [
                    "timestamp",
                    "request_id",
                    "model",
                    "backend_type",
                    "endpoint",
                    "duration_ms",
                    "status",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "error_type",
                    "error_message",
                ]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for log in logs:
                    row = log.model_dump()
                    row["timestamp"] = row["timestamp"].isoformat()
                    writer.writerow({k: row.get(k) for k in fieldnames})

            return PlainTextResponse(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=audit-logs.csv"},
            )
        else:
            # JSONL format (default)
            lines = []
            for log in logs:
                data = log.model_dump()
                data["timestamp"] = data["timestamp"].isoformat()
                lines.append(json.dumps(data))

            return PlainTextResponse(
                content="\n".join(lines),
                media_type="application/jsonl",
                headers={"Content-Disposition": "attachment; filename=audit-logs.jsonl"},
            )


# ============================================================================
# WebSocket for Live Log Updates
# ============================================================================


@router.websocket("/ws/audit-logs")
async def audit_log_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time audit log streaming.

    Sends recent logs on connect, then streams new entries.
    """
    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)

    def on_new_log(log: dict[str, Any]) -> None:
        """Callback when new log entry is created."""
        try:
            queue.put_nowait(log)
        except asyncio.QueueFull:
            pass  # Drop if queue is full

    # Subscribe to new logs
    audit_service.subscribe(on_new_log)

    try:
        # Send recent logs first
        for log in audit_service.get_recent_logs():
            await websocket.send_json({"type": "log", "data": log})

        # Stream new logs
        while True:
            try:
                log = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json({"type": "log", "data": log})
            except TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    finally:
        audit_service.unsubscribe(on_new_log)
