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
    memory_percent: float
    cpu_percent: float
    status: str
    create_time: float


class RunningServerInfo(TypedDict):
    """Information about a running server."""

    profile_id: int
    pid: int
    memory_mb: float
    memory_percent: float
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


class ModelCharacteristics(TypedDict, total=False):
    """Model characteristics extracted from config.json."""

    model_type: str  # Raw model_type from config
    architecture_family: str  # Normalized: "Llama", "Qwen", "Mistral", etc.
    max_position_embeddings: int  # Context window size
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int | None  # GQA heads (if different from attention heads)
    quantization_bits: int | None  # 2, 3, 4, 8, or None for fp16/bf16
    quantization_group_size: int | None
    is_multimodal: bool
    multimodal_type: str | None  # "vision" or None
    use_cache: bool  # KV cache enabled (default True)


class LocalModelInfo(TypedDict, total=False):
    """Information about a locally downloaded model."""

    model_id: str
    local_path: str
    size_bytes: int
    size_gb: float
    characteristics: ModelCharacteristics | None


class DownloadStatus(TypedDict, total=False):
    """Status update for model download."""

    status: str  # "starting", "downloading", "completed", "failed"
    model_id: str
    total_size_gb: float  # Deprecated, use total_bytes
    total_bytes: int
    downloaded_bytes: int
    local_path: str
    progress: int  # 0-100
    speed_mbps: float
    error: str


class LaunchdStatus(TypedDict, total=False):
    """Status of a launchd service."""

    installed: bool
    running: bool
    label: str
    plist_path: str
    pid: int | None
