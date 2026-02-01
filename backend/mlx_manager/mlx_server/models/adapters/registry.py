"""Model adapter registry and family detection."""

import logging

from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter, ModelAdapter
from mlx_manager.mlx_server.models.adapters.gemma import GemmaAdapter
from mlx_manager.mlx_server.models.adapters.glm4 import GLM4Adapter
from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter
from mlx_manager.mlx_server.models.adapters.mistral import MistralAdapter
from mlx_manager.mlx_server.models.adapters.qwen import QwenAdapter

logger = logging.getLogger(__name__)

# Adapter instances (singletons)
_ADAPTERS: dict[str, ModelAdapter] = {
    "llama": LlamaAdapter(),
    "qwen": QwenAdapter(),
    "mistral": MistralAdapter(),
    "gemma": GemmaAdapter(),
    "glm4": GLM4Adapter(),
    "default": DefaultAdapter(),
}


def detect_model_family(model_id: str) -> str:
    """Detect model family from HuggingFace model ID.

    Args:
        model_id: e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit"

    Returns:
        Family name: "llama", "qwen", "mistral", "gemma", or "default"
    """
    model_id_lower = model_id.lower()

    # Llama family (including CodeLlama)
    if "llama" in model_id_lower or "codellama" in model_id_lower:
        return "llama"

    # Qwen family (Phase 8)
    if "qwen" in model_id_lower:
        return "qwen"

    # Mistral family (Phase 8)
    if "mistral" in model_id_lower or "mixtral" in model_id_lower:
        return "mistral"

    # Gemma family (Phase 8)
    if "gemma" in model_id_lower:
        return "gemma"

    # GLM4 family (includes ChatGLM)
    if "glm" in model_id_lower or "chatglm" in model_id_lower:
        return "glm4"

    # Phi family
    if "phi" in model_id_lower:
        return "phi"

    # Default fallback
    logger.info("Unknown model family for %s, using default adapter", model_id)
    return "default"


def get_adapter(model_id: str) -> ModelAdapter:
    """Get the appropriate adapter for a model.

    Args:
        model_id: HuggingFace model ID

    Returns:
        ModelAdapter instance for the model's family
    """
    family = detect_model_family(model_id)
    adapter = _ADAPTERS.get(family, _ADAPTERS["default"])
    logger.debug("Using %s adapter for %s", adapter.family, model_id)
    return adapter


def register_adapter(family: str, adapter: ModelAdapter) -> None:
    """Register a custom adapter for a model family.

    Args:
        family: Family name (e.g., "qwen")
        adapter: ModelAdapter instance
    """
    _ADAPTERS[family] = adapter
    logger.info("Registered adapter for family: %s", family)


def get_supported_families() -> list[str]:
    """Get list of supported model families."""
    return list(_ADAPTERS.keys())
