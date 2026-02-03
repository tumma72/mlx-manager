"""Model type detection for MLX server.

This module determines the model TYPE (text-gen, vision, embeddings) to select
the correct loading strategy. It reuses config reading utilities from the
existing model_detection module but applies MLX server-specific logic.
"""

import logging
from typing import Any

from mlx_manager.mlx_server.models.types import ModelType

logger = logging.getLogger(__name__)


def detect_model_type(model_id: str, config: dict[str, Any] | None = None) -> ModelType:
    """Detect model type using the decision chain.

    Decision priority:
    1. config.json fields (most reliable)
    2. Model name patterns (fallback)
    3. Default to TEXT_GEN

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/Qwen2-VL-2B-Instruct-4bit")
        config: Optional pre-loaded config.json dict (loads from cache if not provided)

    Returns:
        ModelType enum value
    """
    # Try to load config if not provided
    if config is None:
        try:
            from mlx_manager.utils.model_detection import read_model_config

            config = read_model_config(model_id)
            if config:
                logger.debug(
                    f"Loaded config for {model_id}: keys={list(config.keys())}"
                )
            else:
                logger.debug(f"No config found for {model_id} (model not downloaded)")
        except Exception as e:
            logger.warning(f"Could not load config for {model_id}: {e}")
            config = None

    if config:
        # Vision: has vision_config, image_token_id, or image_token_index
        # (Gemma 3 uses image_token_index instead of image_token_id)
        vision_keys = ("vision_config", "image_token_id", "image_token_index")
        if any(key in config for key in vision_keys):
            logger.debug(f"Detected VISION model from config: {model_id}")
            return ModelType.VISION

        # Check model_type for vision indicators
        model_type = config.get("model_type", "").lower()
        if any(ind in model_type for ind in ("vl", "vision", "multimodal")):
            logger.debug(f"Detected VISION model from model_type: {model_id}")
            return ModelType.VISION

        # Embeddings: specific architectures
        arch_list = config.get("architectures", [])
        if arch_list:
            arch = arch_list[0].lower() if arch_list else ""
            embedding_indicators = ("embedding", "sentence", "bert", "roberta", "e5", "bge")
            if any(ind in arch for ind in embedding_indicators):
                logger.debug(f"Detected EMBEDDINGS model from architecture: {model_id}")
                return ModelType.EMBEDDINGS

        # Check model_type for embeddings indicators
        if any(ind in model_type for ind in ("embedding", "sentence", "bert")):
            logger.debug(f"Detected EMBEDDINGS model from model_type: {model_id}")
            return ModelType.EMBEDDINGS

    # Name-based fallback
    name_lower = model_id.lower()

    # Vision patterns
    vision_patterns = (
        "-vl",
        "vlm",
        "vision",
        "qwen2-vl",
        "qwen2.5-vl",
        "llava",
        "pixtral",
        "gemma-3",  # Gemma 3 multimodal (gemma-3-*-it models)
    )
    if any(pattern in name_lower for pattern in vision_patterns):
        logger.debug(f"Detected VISION model from name pattern: {model_id}")
        return ModelType.VISION

    # Embeddings patterns
    embed_patterns = ("embed", "minilm", "sentence", "e5-", "bge-", "gte-")
    if any(pattern in name_lower for pattern in embed_patterns):
        logger.debug(f"Detected EMBEDDINGS model from name pattern: {model_id}")
        return ModelType.EMBEDDINGS

    # Default to text generation
    logger.debug(f"Defaulting to TEXT_GEN model: {model_id}")
    return ModelType.TEXT_GEN
