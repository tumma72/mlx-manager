"""System API router."""

import platform
import sys

import psutil
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from mlx_manager.config import settings
from mlx_manager.database import get_db
from mlx_manager.models import LaunchdStatus, ServerProfile, SystemInfo, SystemMemory
from mlx_manager.services.launchd import launchd_manager

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/memory", response_model=SystemMemory)
async def get_memory():
    """Get system memory information."""
    mem = psutil.virtual_memory()

    total_gb = mem.total / 1e9
    available_gb = mem.available / 1e9
    used_gb = mem.used / 1e9
    percent_used = mem.percent

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
async def get_system_info():
    """Get system information."""
    # Get OS info
    os_version = f"{platform.system()} {platform.release()}"

    # Get chip info (macOS specific)
    chip = "Unknown"
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            chip = result.stdout.strip()
    except Exception:
        pass

    # Get memory
    mem = psutil.virtual_memory()
    memory_gb = round(mem.total / 1e9, 0)

    # Get Python version
    python_version = sys.version.split()[0]

    # Try to get MLX version
    mlx_version = None
    try:
        import mlx

        mlx_version = mlx.__version__
    except ImportError:
        pass

    # Try to get mlx-openai-server version
    mlx_openai_server_version = None
    try:
        import mlx_openai_server

        mlx_openai_server_version = getattr(mlx_openai_server, "__version__", "installed")
    except ImportError:
        pass

    return SystemInfo(
        os_version=os_version,
        chip=chip,
        memory_gb=memory_gb,
        python_version=python_version,
        mlx_version=mlx_version,
        mlx_openai_server_version=mlx_openai_server_version,
    )


@router.post("/launchd/install/{profile_id}")
async def install_launchd_service(profile_id: int, session: AsyncSession = Depends(get_db)):
    """Install profile as launchd service."""
    # Get profile
    result = await session.execute(select(ServerProfile).where(ServerProfile.id == profile_id))
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

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
async def uninstall_launchd_service(profile_id: int, session: AsyncSession = Depends(get_db)):
    """Uninstall launchd service."""
    # Get profile
    result = await session.execute(select(ServerProfile).where(ServerProfile.id == profile_id))
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    launchd_manager.uninstall(profile)

    # Update profile
    profile.launchd_installed = False
    session.add(profile)
    await session.commit()


@router.get("/launchd/status/{profile_id}", response_model=LaunchdStatus)
async def get_launchd_status(profile_id: int, session: AsyncSession = Depends(get_db)):
    """Get launchd service status."""
    # Get profile
    result = await session.execute(select(ServerProfile).where(ServerProfile.id == profile_id))
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    status = launchd_manager.get_status(profile)
    return LaunchdStatus(**status)
