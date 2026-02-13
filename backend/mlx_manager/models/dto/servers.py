"""Server DTOs - running server status, health, and embedded server info."""

from pydantic import BaseModel

__all__ = [
    "RunningServerResponse",
    "HealthStatus",
    "ServerStatus",
    "EmbeddedServerStatus",
    "LoadedModelInfo",
    "ServerHealthStatus",
]


class RunningServerResponse(BaseModel):
    """Response model for running server status."""

    profile_id: int
    profile_name: str
    pid: int
    health_status: str
    uptime_seconds: float
    memory_mb: float
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    port: int = 0
    memory_limit_percent: float = 0.0


class HealthStatus(BaseModel):
    """Server health status."""

    status: str
    response_time_ms: float | None = None
    model_loaded: bool | None = None
    error: str | None = None


class ServerStatus(BaseModel):
    """Detailed server process status."""

    profile_id: int
    running: bool
    pid: int | None = None
    exit_code: int | None = None
    failed: bool = False
    error_message: str | None = None


class EmbeddedServerStatus(BaseModel):
    """Status of the embedded MLX Server."""

    status: str  # "running", "not_initialized"
    type: str = "embedded"
    uptime_seconds: float = 0.0
    loaded_models: list[str] = []
    memory_used_gb: float = 0.0
    memory_limit_gb: float = 0.0


class LoadedModelInfo(BaseModel):
    """Information about a loaded model."""

    model_id: str
    model_type: str
    size_gb: float
    loaded_at: float
    last_used: float
    preloaded: bool


class ServerHealthStatus(BaseModel):
    """Health status of the embedded server."""

    status: str  # "healthy", "degraded", "unhealthy"
    model_pool_initialized: bool = False
    loaded_model_count: int = 0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
