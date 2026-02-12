"""Consolidated domain enums for MLX Manager."""

from enum import StrEnum


class UserStatus(StrEnum):
    """User account status."""

    PENDING = "pending"
    APPROVED = "approved"
    DISABLED = "disabled"


class ApiType(StrEnum):
    """API protocol type for cloud providers."""

    OPENAI = "openai"  # OpenAI-compatible API
    ANTHROPIC = "anthropic"  # Anthropic-compatible API


class BackendType(StrEnum):
    """Backend types for routing."""

    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # Generic providers
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC_COMPATIBLE = "anthropic_compatible"
    # Common providers (convenience)
    TOGETHER = "together"
    GROQ = "groq"
    FIREWORKS = "fireworks"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"


class ModelType(StrEnum):
    """Supported model types for the MLX server.

    Each type corresponds to different inference capabilities:
    - TEXT_GEN: Text generation models (e.g., Llama, Mistral)
    - VISION: Vision-language models (e.g., LLaVA, Qwen-VL)
    - EMBEDDINGS: Embedding models (e.g., BGE, E5)
    - AUDIO: Audio models for TTS and STT (e.g., Kokoro, Whisper)
    """

    TEXT_GEN = "text-gen"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    AUDIO = "audio"


class PatternType(StrEnum):
    """Pattern matching type for backend mappings."""

    EXACT = "exact"
    PREFIX = "prefix"
    REGEX = "regex"


class MemoryLimitMode(StrEnum):
    """Memory limit configuration mode."""

    PERCENT = "percent"
    GB = "gb"


class EvictionPolicy(StrEnum):
    """Model pool eviction policy."""

    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"


class DownloadStatusEnum(StrEnum):
    """Download operation status."""

    PENDING = "pending"
    STARTING = "starting"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProfileType(StrEnum):
    """Server profile type for future polymorphic support."""

    BASE = "base"
    INFERENCE = "inference"
    AUDIO = "audio"
