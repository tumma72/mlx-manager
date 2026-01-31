"""System API router."""

import asyncio
import logging
import platform
import subprocess
import sys
from datetime import datetime
from typing import Annotated, Any

import httpx
import psutil
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_manager.config import settings
from mlx_manager.database import get_db
from mlx_manager.dependencies import get_current_user, get_profile_or_404
from mlx_manager.models import LaunchdStatus, ServerProfile, SystemInfo, SystemMemory, User
from mlx_manager.services.launchd import launchd_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])


def get_physical_memory_bytes() -> int:
    """
    Get actual physical memory in bytes.

    On macOS, psutil.virtual_memory().total can return inflated values
    (e.g., 137GB on a 128GB Mac) due to including compressed memory or swap.
    This function uses sysctl on macOS for accurate physical RAM reporting.
    """
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            logger.debug(f"sysctl failed, falling back to psutil: {e}")
    # Fallback to psutil for other platforms or on error
    return psutil.virtual_memory().total


@router.get("/memory", response_model=SystemMemory)
async def get_memory(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get system memory information."""
    mem = psutil.virtual_memory()

    # Use accurate physical memory for total (sysctl on macOS)
    # Convert bytes to GiB using binary (1024^3), not decimal (1e9)
    total_bytes = get_physical_memory_bytes()
    bytes_to_gib = 1024**3
    total_gb = total_bytes / bytes_to_gib
    available_gb = mem.available / bytes_to_gib
    used_gb = (total_bytes - mem.available) / bytes_to_gib
    percent_used = ((total_bytes - mem.available) / total_bytes) * 100

    # MLX recommended is 80% of total
    mlx_recommended_gb = total_gb * (settings.max_memory_percent / 100)

    return SystemMemory(
        total_gb=round(total_gb, 2),
        available_gb=round(available_gb, 2),
        used_gb=round(used_gb, 2),
        percent_used=round(percent_used, 2),
        mlx_recommended_gb=round(mlx_recommended_gb, 2),
    )


@router.get("/info", response_model=SystemInfo)
async def get_system_info(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get system information."""
    # Get OS info
    os_version = f"{platform.system()} {platform.release()}"

    # Get chip info (macOS specific)
    chip = "Unknown"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            chip = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug(f"Could not get chip info: {e}")

    # Get memory (use accurate physical memory, convert to GiB)
    memory_gb = round(get_physical_memory_bytes() / (1024**3), 0)

    # Get Python version
    python_version = sys.version.split()[0]

    # Try to get MLX version
    mlx_version = None
    try:
        import mlx  # type: ignore[import-not-found]

        mlx_version = getattr(mlx, "__version__", "installed")
    except ImportError as e:
        logger.debug(f"MLX not available: {e}")

    # Try to get mlx-openai-server version
    mlx_openai_server_version = None
    try:
        import mlx_openai_server  # type: ignore[import-not-found]

        mlx_openai_server_version = getattr(mlx_openai_server, "__version__", "installed")
    except ImportError as e:
        logger.debug(f"mlx-openai-server not available: {e}")

    return SystemInfo(
        os_version=os_version,
        chip=chip,
        memory_gb=memory_gb,
        python_version=python_version,
        mlx_version=mlx_version,
        mlx_openai_server_version=mlx_openai_server_version,
    )


@router.get("/parser-options")
async def get_available_parser_options(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, list[str]]:
    """
    Get available parser options.

    DEPRECATED: Parser options were used for mlx-openai-server CLI arguments.
    The embedded MLX Server doesn't use these. Returns empty lists for
    backwards compatibility.
    """
    return {
        "tool_call_parsers": [],
        "reasoning_parsers": [],
        "message_converters": [],
    }


@router.post("/launchd/install/{profile_id}")
async def install_launchd_service(
    current_user: Annotated[User, Depends(get_current_user)],
    profile: ServerProfile = Depends(get_profile_or_404),
    session: AsyncSession = Depends(get_db),
):
    """Install profile as launchd service."""
    try:
        plist_path = launchd_manager.install(profile)

        # Update profile
        profile.launchd_installed = True
        session.add(profile)
        await session.commit()

        return {"plist_path": plist_path, "label": launchd_manager.get_label(profile)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/launchd/uninstall/{profile_id}", status_code=204)
async def uninstall_launchd_service(
    current_user: Annotated[User, Depends(get_current_user)],
    profile: ServerProfile = Depends(get_profile_or_404),
    session: AsyncSession = Depends(get_db),
):
    """Uninstall launchd service."""
    launchd_manager.uninstall(profile)

    # Update profile
    profile.launchd_installed = False
    session.add(profile)
    await session.commit()


@router.get("/launchd/status/{profile_id}", response_model=LaunchdStatus)
async def get_launchd_status(
    current_user: Annotated[User, Depends(get_current_user)],
    profile: ServerProfile = Depends(get_profile_or_404),
):
    """Get launchd service status."""
    status = launchd_manager.get_status(profile)
    return LaunchdStatus(**status)


# ============================================================================
# Audit Log Proxy Endpoints
# ============================================================================

# MLX Server base URL (default port 10242)
MLX_SERVER_URL = "http://localhost:10242"


@router.get("/audit-logs")
async def get_audit_logs(
    current_user: Annotated[User, Depends(get_current_user)],
    model: str | None = Query(default=None),
    backend_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    """Proxy audit logs from MLX Server."""
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if model:
        params["model"] = model
    if backend_type:
        params["backend_type"] = backend_type
    if status:
        params["status"] = status
    if start_time:
        params["start_time"] = start_time.isoformat()
    if end_time:
        params["end_time"] = end_time.isoformat()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MLX_SERVER_URL}/admin/audit-logs", params=params, timeout=10.0
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e)) from e
    except Exception as e:
        logger.warning(f"Failed to fetch audit logs: {e}")
        raise HTTPException(status_code=503, detail="MLX Server not available") from e


@router.get("/audit-logs/stats")
async def get_audit_stats(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, Any]:
    """Proxy audit stats from MLX Server."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MLX_SERVER_URL}/admin/audit-logs/stats", timeout=10.0
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e)) from e
    except Exception as e:
        logger.warning(f"Failed to fetch audit stats: {e}")
        raise HTTPException(status_code=503, detail="MLX Server not available") from e


@router.get("/audit-logs/export")
async def export_audit_logs(
    current_user: Annotated[User, Depends(get_current_user)],
    model: str | None = Query(default=None),
    backend_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    format: str = Query(default="jsonl"),
) -> StreamingResponse:
    """Proxy audit log export from MLX Server."""
    params: dict[str, Any] = {"format": format}
    if model:
        params["model"] = model
    if backend_type:
        params["backend_type"] = backend_type
    if status:
        params["status"] = status
    if start_time:
        params["start_time"] = start_time.isoformat()
    if end_time:
        params["end_time"] = end_time.isoformat()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MLX_SERVER_URL}/admin/audit-logs/export", params=params, timeout=60.0
            )
            response.raise_for_status()

            media_type = "text/csv" if format == "csv" else "application/jsonl"
            filename = f"audit-logs.{format}"

            return StreamingResponse(
                content=iter([response.content]),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e)) from e
    except Exception as e:
        logger.warning(f"Failed to export audit logs: {e}")
        raise HTTPException(status_code=503, detail="MLX Server not available") from e


# ============================================================================
# WebSocket Proxy for Audit Logs
# ============================================================================


@router.websocket("/ws/audit-logs")
async def proxy_audit_log_stream(websocket: WebSocket) -> None:
    """Proxy WebSocket connection to MLX Server for audit log streaming.

    The frontend connects to the manager API (port 8080), not directly to
    the MLX Server (port 10242). This endpoint proxies the WebSocket
    connection to the MLX Server's audit log stream.
    """
    await websocket.accept()

    # MLX Server WebSocket URL (default port 10242)
    mlx_server_ws_url = "ws://localhost:10242/admin/ws/audit-logs"

    try:
        import websockets

        async with websockets.connect(mlx_server_ws_url) as mlx_ws:

            async def receive_from_mlx() -> None:
                """Forward messages from MLX Server to frontend."""
                async for message in mlx_ws:
                    await websocket.send_text(message)

            async def receive_from_client() -> None:
                """Forward messages from frontend to MLX Server."""
                while True:
                    try:
                        data = await websocket.receive_text()
                        await mlx_ws.send(data)
                    except WebSocketDisconnect:
                        return

            # Run both directions concurrently
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(receive_from_mlx()),
                    asyncio.create_task(receive_from_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except Exception as e:
        logger.warning(f"Audit log WebSocket proxy failed: {e}")
        try:
            await websocket.send_json(
                {"type": "error", "message": "MLX Server not available for audit logs"}
            )
        except Exception:
            pass  # Client may have already disconnected
        await websocket.close()
