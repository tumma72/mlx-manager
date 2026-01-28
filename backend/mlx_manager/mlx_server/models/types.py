"""Model types for the MLX server."""

from enum import Enum


class ModelType(str, Enum):
    """Supported model types for the MLX server.

    Each type corresponds to different inference capabilities:
    - TEXT_GEN: Text generation models (e.g., Llama, Mistral)
    - VISION: Vision-language models (e.g., LLaVA, Qwen-VL)
    - EMBEDDINGS: Embedding models (e.g., BGE, E5)
    """

    TEXT_GEN = "text-gen"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
