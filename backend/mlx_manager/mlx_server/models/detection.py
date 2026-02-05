"""Model type detection for MLX server.

This module determines the model TYPE (text-gen, vision, embeddings, audio) to select
the correct loading strategy. It reuses detection utilities from the existing
model_detection module to ensure consistency between badge display and loading.
"""

from typing import Any

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType


def detect_model_type(model_id: str, config: dict[str, Any] | None = None) -> ModelType:
    """Detect model type using the decision chain.

    Decision priority:
    1. config.json fields via shared detect_multimodal() (most reliable)
    2. Model name patterns (fallback)
    3. Default to TEXT_GEN

    Uses the shared detect_multimodal() function from utils/model_detection.py
    to ensure badge display and model loading use identical detection logic.

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/Qwen2-VL-2B-Instruct-4bit")
        config: Optional pre-loaded config.json dict (loads from cache if not provided)

    Returns:
        ModelType enum value
    """
    from mlx_manager.utils.model_detection import detect_multimodal, read_model_config

    # Try to load config if not provided
    if config is None:
        try:
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
        # Use shared multimodal detection (same logic as badge display)
        is_multimodal, multimodal_type = detect_multimodal(config)
        if is_multimodal and multimodal_type == "vision":
            logger.debug(f"Detected VISION model from config: {model_id}")
            return ModelType.VISION

        # Audio: config fields that indicate TTS/STT models
        audio_config_indicators = (
            "audio_config",
            "tts_config",
            "stt_config",
            "vocoder_config",
            "codec_config",
        )
        if any(key in config for key in audio_config_indicators):
            logger.debug(f"Detected AUDIO model from config field: {model_id}")
            return ModelType.AUDIO

        # Audio: architecture-based detection
        arch_list = config.get("architectures", [])
        if arch_list:
            arch = arch_list[0].lower() if arch_list else ""
            audio_arch_indicators = (
                "kokoro",
                "whisper",
                "bark",
                "speecht5",
                "parler",
                "sesame",
                "spark",
                "dia",
                "outetts",
                "chatterbox",
                "parakeet",
                "voxtral",
                "vibevoice",
                "voxcpm",
                "soprano",
            )
            if any(ind in arch for ind in audio_arch_indicators):
                logger.debug(f"Detected AUDIO model from architecture: {model_id}")
                return ModelType.AUDIO

        # Audio: model_type field detection
        config_model_type = config.get("model_type", "").lower()
        audio_model_type_indicators = (
            "kokoro",
            "whisper",
            "bark",
            "speecht5",
            "parler",
            "dia",
            "outetts",
            "spark",
            "chatterbox",
            "soprano",
            "parakeet",
            "qwen3_tts",
            "qwen3_asr",
            "glm",
        )
        if any(ind in config_model_type for ind in audio_model_type_indicators):
            logger.debug(f"Detected AUDIO model from model_type: {model_id}")
            return ModelType.AUDIO

        # Embeddings: specific architectures
        if arch_list:
            arch = arch_list[0].lower() if arch_list else ""
            embedding_indicators = ("embedding", "sentence", "bert", "roberta", "e5", "bge")
            if any(ind in arch for ind in embedding_indicators):
                logger.debug(f"Detected EMBEDDINGS model from architecture: {model_id}")
                return ModelType.EMBEDDINGS

        # Check model_type for embeddings indicators
        model_type = config.get("model_type", "").lower()
        if any(ind in model_type for ind in ("embedding", "sentence", "bert")):
            logger.debug(f"Detected EMBEDDINGS model from model_type: {model_id}")
            return ModelType.EMBEDDINGS

    # Name-based fallback
    name_lower = model_id.lower()

    # Audio name patterns (check before vision/embeddings since audio is more specific)
    audio_name_patterns = (
        "kokoro",
        "whisper",
        "tts",
        "stt",
        "speech",
        "bark",
        "speecht5",
        "parler",
        "chatterbox",
        "dia-",
        "outetts",
        "spark-tts",
        "parakeet",
        "voxtral",
        "vibevoice",
        "voxcpm",
        "soprano",
    )
    if any(pattern in name_lower for pattern in audio_name_patterns):
        logger.debug(f"Detected AUDIO model from name pattern: {model_id}")
        return ModelType.AUDIO

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
