"""System API router."""

import platform
import subprocess
import sys

import psutil
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from mlx_manager.config import settings
from mlx_manager.database import get_db
from mlx_manager.dependencies import get_profile_or_404
from mlx_manager.models import LaunchdStatus, ServerProfile, SystemInfo, SystemMemory
from mlx_manager.services.launchd import launchd_manager

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
        except Exception:
            pass
    # Fallback to psutil for other platforms or on error
    return psutil.virtual_memory().total


@router.get("/memory", response_model=SystemMemory)
async def get_memory():
    """Get system memory information."""
    mem = psutil.virtual_memory()

    # Use accurate physical memory for total (sysctl on macOS)
    total_bytes = get_physical_memory_bytes()
    total_gb = total_bytes / 1e9
    available_gb = mem.available / 1e9
    used_gb = (total_bytes - mem.available) / 1e9
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
async def get_system_info():
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
    except Exception:
        pass

    # Get memory (use accurate physical memory)
    memory_gb = round(get_physical_memory_bytes() / 1e9, 0)

    # Get Python version
    python_version = sys.version.split()[0]

    # Try to get MLX version
    mlx_version = None
    try:
        import mlx  # type: ignore[import-not-found]

        mlx_version = getattr(mlx, "__version__", "installed")
    except ImportError:
        pass

    # Try to get mlx-openai-server version
    mlx_openai_server_version = None
    try:
        import mlx_openai_server  # type: ignore[import-not-found]

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
async def install_launchd_service(
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
async def get_launchd_status(profile: ServerProfile = Depends(get_profile_or_404)):
    """Get launchd service status."""
    status = launchd_manager.get_status(profile)
    return LaunchdStatus(**status)
