"""Model DTOs - catalog responses and download requests."""

from datetime import datetime

from pydantic import BaseModel, computed_field

from mlx_manager.models.capabilities import CapabilitiesResponse
from mlx_manager.models.enums import DownloadStatusEnum

__all__ = [
    "ModelResponse",
    "ModelCharacteristics",
    "DownloadStatus",
    "ModelSearchResult",
    "LocalModel",
    "DownloadRequest",
]


class ModelResponse(BaseModel):
    """Response model for Model entity with nested capabilities."""

    id: int
    repo_id: str
    model_type: str | None = None
    local_path: str | None = None
    size_bytes: int | None = None
    size_gb: float | None = None
    downloaded_at: datetime | None = None
    last_used_at: datetime | None = None

    # Capabilities from STI hierarchy (None = not probed)
    capabilities: CapabilitiesResponse | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_id(self) -> str:
        """Alias for repo_id used by the frontend capabilities mapping."""
        return self.repo_id

    # Backward-compatible flat accessors for frontend transition
    @computed_field  # type: ignore[prop-decorator]
    @property
    def probed_at(self) -> datetime | None:
        return self.capabilities.probed_at if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def probe_version(self) -> int | None:
        return self.capabilities.probe_version if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_family(self) -> str | None:
        return self.capabilities.model_family if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def supports_native_tools(self) -> bool | None:
        return self.capabilities.supports_native_tools if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def supports_thinking(self) -> bool | None:
        return self.capabilities.supports_thinking if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_format(self) -> str | None:
        return self.capabilities.tool_format if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def practical_max_tokens(self) -> int | None:
        return self.capabilities.practical_max_tokens if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_parser_id(self) -> str | None:
        return self.capabilities.tool_parser_id if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def thinking_parser_id(self) -> str | None:
        return self.capabilities.thinking_parser_id if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def supports_multi_image(self) -> bool | None:
        return self.capabilities.supports_multi_image if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def supports_video(self) -> bool | None:
        return self.capabilities.supports_video if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embedding_dimensions(self) -> int | None:
        return self.capabilities.embedding_dimensions if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_sequence_length(self) -> int | None:
        return self.capabilities.max_sequence_length if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_normalized(self) -> bool | None:
        return self.capabilities.is_normalized if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def supports_tts(self) -> bool | None:
        return self.capabilities.supports_tts if self.capabilities else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def supports_stt(self) -> bool | None:
        return self.capabilities.supports_stt if self.capabilities else None


class ModelCharacteristics(BaseModel):
    """Model characteristics extracted from config.json."""

    model_type: str | None = None
    architecture_family: str | None = None
    max_position_embeddings: int | None = None
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    quantization_bits: int | None = None
    quantization_group_size: int | None = None
    is_multimodal: bool | None = None
    multimodal_type: str | None = None
    use_cache: bool | None = None
    is_tool_use: bool | None = None


class DownloadStatus(BaseModel):
    """Status update for model download progress."""

    status: DownloadStatusEnum | None = None
    model_id: str | None = None
    total_size_gb: float | None = None
    total_bytes: int | None = None
    downloaded_bytes: int | None = None
    local_path: str | None = None
    progress: int | None = None
    speed_mbps: float | None = None
    error: str | None = None


class ModelSearchResult(BaseModel):
    """Model search result from HuggingFace."""

    model_id: str
    author: str
    downloads: int
    likes: int
    estimated_size_gb: float
    tags: list[str]
    is_downloaded: bool
    last_modified: str | None = None


class LocalModel(BaseModel):
    """Locally downloaded model."""

    model_id: str
    local_path: str
    size_bytes: int
    size_gb: float
    characteristics: ModelCharacteristics | None = None


class DownloadRequest(BaseModel):
    """Request body for model download."""

    model_id: str
