"""Type definitions for MLX Manager."""

from typing import TypedDict


class HealthCheckResult(TypedDict, total=False):
    """Result from server health check."""

    status: str
    response_time_ms: float
    model_loaded: bool
    error: str


class ServerStats(TypedDict):
    """Statistics for a running server process."""

    pid: int
    memory_mb: float
    cpu_percent: float
    status: str
    create_time: float


class RunningServerInfo(TypedDict):
    """Information about a running server."""

    profile_id: int
    pid: int
    memory_mb: float
    cpu_percent: float
    status: str
    create_time: float


class ModelSearchResult(TypedDict, total=False):
    """Search result from HuggingFace Hub."""

    model_id: str
    author: str
    downloads: int
    likes: int
    estimated_size_gb: float
    tags: list[str]
    is_downloaded: bool
    last_modified: str | None


class LocalModelInfo(TypedDict):
    """Information about a locally downloaded model."""

    model_id: str
    local_path: str
    size_bytes: int
    size_gb: float


class DownloadStatus(TypedDict, total=False):
    """Status update for model download."""

    status: str
    model_id: str
    total_size_gb: float
    local_path: str
    progress: int
    error: str


class LaunchdStatus(TypedDict, total=False):
    """Status of a launchd service."""

    installed: bool
    running: bool
    label: str
    plist_path: str
    pid: int | None
