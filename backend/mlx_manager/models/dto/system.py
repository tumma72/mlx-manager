"""System DTOs - system info, memory, and launchd status."""

from pydantic import BaseModel

__all__ = [
    "SystemMemory",
    "SystemInfo",
    "LaunchdStatus",
]


class SystemMemory(BaseModel):
    """System memory information."""

    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    mlx_recommended_gb: float


class SystemInfo(BaseModel):
    """System information."""

    os_version: str
    chip: str
    memory_gb: float
    python_version: str
    mlx_version: str | None = None


class LaunchdStatus(BaseModel):
    """Launchd service status."""

    installed: bool
    running: bool
    pid: int | None = None
    label: str
    plist_path: str | None = None
