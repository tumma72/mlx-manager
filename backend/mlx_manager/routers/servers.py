"""Servers API router."""

import asyncio
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from mlx_manager.database import get_db
from mlx_manager.models import (
    HealthStatus,
    RunningInstance,
    RunningServerResponse,
    ServerProfile,
)
from mlx_manager.services.server_manager import server_manager

router = APIRouter(prefix="/api/servers", tags=["servers"])


@router.get("", response_model=list[RunningServerResponse])
async def list_running_servers(session: AsyncSession = Depends(get_db)):
    """List all running server instances."""
    # Get all running processes from server manager
    running = server_manager.get_all_running()
    running_profile_ids = {r["profile_id"] for r in running}

    # Get profile info for running servers
    if not running_profile_ids:
        return []

    result = await session.execute(
        select(ServerProfile).where(ServerProfile.id.in_(running_profile_ids))
    )
    profiles = {p.id: p for p in result.scalars().all()}

    # Build response
    servers = []
    for r in running:
        profile = profiles.get(r["profile_id"])
        if not profile:
            continue

        # Calculate uptime
        uptime = time.time() - r.get("create_time", time.time())

        servers.append(
            RunningServerResponse(
                profile_id=profile.id,
                profile_name=profile.name,
                pid=r["pid"],
                port=profile.port,
                health_status=r.get("status", "unknown"),
                uptime_seconds=uptime,
                memory_mb=r.get("memory_mb", 0),
            )
        )

    return servers


@router.post("/{profile_id}/start")
async def start_server(profile_id: int, session: AsyncSession = Depends(get_db)):
    """Start a server for a profile."""
    # Get profile
    result = await session.execute(select(ServerProfile).where(ServerProfile.id == profile_id))
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        pid = await server_manager.start_server(profile)

        # Record in database
        instance = RunningInstance(profile_id=profile.id, pid=pid, health_status="starting")
        session.add(instance)
        await session.commit()

        return {"pid": pid, "port": profile.port}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{profile_id}/stop")
async def stop_server(
    profile_id: int, force: bool = False, session: AsyncSession = Depends(get_db)
):
    """Stop a running server."""
    success = await server_manager.stop_server(profile_id, force=force)

    if not success:
        raise HTTPException(status_code=404, detail="Server not running")

    # Remove from database
    result = await session.execute(
        select(RunningInstance).where(RunningInstance.profile_id == profile_id)
    )
    instance = result.scalar_one_or_none()
    if instance:
        await session.delete(instance)
        await session.commit()

    return {"stopped": True}


@router.post("/{profile_id}/restart")
async def restart_server(profile_id: int, session: AsyncSession = Depends(get_db)):
    """Restart a server."""
    # Get profile
    result = await session.execute(select(ServerProfile).where(ServerProfile.id == profile_id))
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Stop if running
    await server_manager.stop_server(profile_id, force=False)
    await asyncio.sleep(1)

    # Start again
    try:
        pid = await server_manager.start_server(profile)

        # Update database
        result = await session.execute(
            select(RunningInstance).where(RunningInstance.profile_id == profile_id)
        )
        instance = result.scalar_one_or_none()

        if instance:
            instance.pid = pid
            instance.health_status = "starting"
            instance.started_at = datetime.utcnow()
        else:
            instance = RunningInstance(profile_id=profile.id, pid=pid, health_status="starting")

        session.add(instance)
        await session.commit()

        return {"pid": pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{profile_id}/health", response_model=HealthStatus)
async def check_server_health(profile_id: int, session: AsyncSession = Depends(get_db)):
    """Check server health."""
    # Get profile
    result = await session.execute(select(ServerProfile).where(ServerProfile.id == profile_id))
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if not server_manager.is_running(profile_id):
        return HealthStatus(status="stopped")

    health = await server_manager.check_health(profile)
    return HealthStatus(**health)


@router.get("/{profile_id}/logs")
async def stream_logs(profile_id: int, lines: int = 100):
    """SSE endpoint for live logs."""

    async def generate():
        import json

        while True:
            log_lines = server_manager.get_log_lines(profile_id)

            for line in log_lines:
                yield f"data: {json.dumps({'line': line})}\n\n"

            if not server_manager.is_running(profile_id):
                yield f"data: {json.dumps({'status': 'stopped'})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")
