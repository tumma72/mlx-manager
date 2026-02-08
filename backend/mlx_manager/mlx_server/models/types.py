"""Model types for the MLX server."""

from dataclasses import dataclass
from enum import StrEnum


@dataclass
class AdapterInfo:
    """Information about a loaded LoRA adapter."""

    adapter_path: str
    base_model: str | None = None  # From adapter_config.json if available
    description: str | None = None


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
