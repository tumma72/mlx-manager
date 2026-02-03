"""
Model family detection for configuring parser options.

This module provides OFFLINE-FIRST detection of model families
from locally cached models. It reads config.json directly from
the HuggingFace cache filesystem without any network calls.
"""

import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from mlx_manager.config import settings
from mlx_manager.types import ModelCharacteristics

logger = logging.getLogger(__name__)

# Mapping from model_type patterns to normalized architecture family names
ARCHITECTURE_FAMILIES: dict[str, str] = {
    "llama": "Llama",
    "mllama": "Llama",  # Multi-modal Llama
    "qwen": "Qwen",
    "qwen2": "Qwen",
    "qwen3": "Qwen",
    "mistral": "Mistral",
    "mixtral": "Mixtral",
    "gemma": "Gemma",
    "gemma2": "Gemma",
    "phi": "Phi",
    "phi3": "Phi",
    "starcoder": "StarCoder",
    "starcoder2": "StarCoder",
    "deepseek": "DeepSeek",
    "deepseek_v3": "DeepSeek",
    "glm": "GLM",
    "chatglm": "GLM",
    "minimax": "MiniMax",
    "falcon": "Falcon",
    "yi": "Yi",
    "internlm": "InternLM",
    "internlm2": "InternLM",
    "baichuan": "Baichuan",
    "cohere": "Cohere",
    "command": "Cohere",
    "olmo": "OLMo",
    "nemotron": "Nemotron",
    "granite": "Granite",
}

# Model families that support tool-use/function-calling by default
# These families have native tool-use support regardless of HuggingFace tags
TOOL_CAPABLE_FAMILIES: set[str] = {
    "qwen",  # Qwen models have native function calling
    "qwen2",
    "qwen3",
    "glm",  # GLM-4 family supports tool use
    "chatglm",
    "minimax",  # MiniMax models support tool use
    "deepseek",  # DeepSeek family supports function calling
    "deepseek_v3",
    "hermes",  # Hermes models fine-tuned for tool use
    "command",  # Cohere Command-R supports tools
    "cohere",
    "mistral",  # Mistral-Instruct models support function calling
}

# Minimum mlx-lm version requirements for model families
# MiniMax support was added in mlx-lm v0.28.4
MODEL_FAMILY_MIN_VERSIONS: dict[str, str] = {
    "minimax": "0.28.4",
}


def get_mlx_lm_version() -> str | None:
    """Get the installed mlx-lm version.

    Returns:
        Version string (e.g., "0.30.2") or None if not installed.
    """
    try:
        return version("mlx-lm")
    except PackageNotFoundError:
        return None


def parse_version(v: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers for comparison.

    Args:
        v: Version string like "0.28.4" or "0.30.2"

    Returns:
        Tuple of integers like (0, 28, 4)
    """
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


def check_mlx_lm_support(model_family: str) -> dict[str, Any]:
    """Check if installed mlx-lm supports a model family.

    Args:
        model_family: The model family (e.g., "minimax", "qwen")

    Returns:
        Dictionary with 'supported', 'installed_version',
        'required_version', and optional 'upgrade_command'.
    """
    installed = get_mlx_lm_version()
    required = MODEL_FAMILY_MIN_VERSIONS.get(model_family)

    if not installed:
        return {
            "supported": False,
            "installed_version": None,
            "required_version": required,
            "error": "mlx-lm is not installed. Install with: pip install mlx-lm",
        }

    if not required:
        # No version requirement for this family
        return {
            "supported": True,
            "installed_version": installed,
            "required_version": None,
        }

    installed_tuple = parse_version(installed)
    required_tuple = parse_version(required)

    if installed_tuple >= required_tuple:
        return {
            "supported": True,
            "installed_version": installed,
            "required_version": required,
        }

    return {
        "supported": False,
        "installed_version": installed,
        "required_version": required,
        "error": f"{model_family.title()} models require mlx-lm >= {required} "
        f"(installed: {installed}). Upgrade with: pip install -U mlx-lm",
        "upgrade_command": "pip install -U mlx-lm",
    }


# NOTE: Parser options are no longer used with the embedded MLX Server.
# The fuzzy matcher in fuzzy_matcher.py is kept for backwards compatibility
# but is not actively used for parser option selection.


def get_local_model_path(model_id: str) -> Path | None:
    """
    Get path to locally downloaded model files.

    Works OFFLINE - no network access required.
    Directly reads HuggingFace cache structure:
    ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{revision}/

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/MiniMax-M2.1-3bit")

    Returns:
        Path to the model snapshot directory, or None if not downloaded.
    """
    cache_dir = settings.hf_cache_path
    cache_name = f"models--{model_id.replace('/', '--')}"
    model_path = cache_dir / cache_name / "snapshots"

    if not model_path.exists():
        return None

    try:
        # Get the most recent snapshot (usually just one)
        snapshots = sorted(
            model_path.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return snapshots[0] if snapshots else None
    except Exception as e:
        logger.warning(f"Error reading model snapshots for {model_id}: {e}")
        return None


def read_model_config(model_id: str) -> dict[str, Any] | None:
    """
    Read config.json from a locally downloaded model.

    Works OFFLINE - reads directly from filesystem.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Parsed config.json contents, or None if not available.
    """
    local_path = get_local_model_path(model_id)
    if not local_path:
        return None

    config_path = local_path / "config.json"
    if not config_path.exists():
        return None

    try:
        config: dict[str, Any] = json.loads(config_path.read_text())
        return config
    except Exception as e:
        logger.warning(f"Error reading config.json for {model_id}: {e}")
        return None


def detect_model_family(model_id: str) -> str | None:
    """
    Detect model family from config.json architecture field.

    Works OFFLINE - reads directly from local filesystem.
    Falls back to model path name matching if not downloaded.

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/MiniMax-M2.1-3bit")

    Returns:
        Model family name (e.g., "minimax", "qwen3", "glm") or None if unknown.
    """

    def _match_family_from_string(text: str) -> str | None:
        """Match model family from a text string (model_type, architecture, or path)."""
        text_lower = text.lower()

        # Check specific variants first (more specific matches)
        # Order matters: check specific variants before base families

        # Qwen variants - check specific types first
        if "qwen" in text_lower:
            if "coder" in text_lower:
                return "qwen3_coder"
            if "moe" in text_lower:
                return "qwen3_moe"
            if "vl" in text_lower:
                return "qwen3_vl"
            return "qwen3"

        # GLM models
        if "glm" in text_lower:
            return "glm"

        # MiniMax models
        if "minimax" in text_lower:
            return "minimax"

        # Nemotron models
        if "nemotron" in text_lower:
            return "nemotron"

        # Harmony models
        if "harmony" in text_lower:
            return "harmony"

        # Hermes models
        if "hermes" in text_lower:
            return "hermes"

        # Solar models
        if "solar" in text_lower:
            return "solar"

        return None

    # Try to read config.json from local cache (OFFLINE - no API calls)
    config = read_model_config(model_id)
    if config:
        # Check model_type field first
        model_type = config.get("model_type", "")
        family = _match_family_from_string(model_type)
        if family:
            logger.debug(f"Detected {family} from model_type: {model_type}")
            return family

        # Check architectures field
        architectures = config.get("architectures", [])
        for arch in architectures:
            family = _match_family_from_string(arch)
            if family:
                logger.debug(f"Detected {family} from architecture: {arch}")
                return family

    # Fallback: check model path name (works even if not downloaded)
    family = _match_family_from_string(model_id)
    if family:
        logger.debug(f"Detected {family} from model path: {model_id}")
        return family

    return None


def get_parser_options(model_id: str) -> dict[str, str]:
    """
    Get recommended parser options for a model using fuzzy matching.

    DEPRECATED: Parser options are no longer used with the embedded MLX Server.
    This function is kept for backwards compatibility but returns empty results.

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/Qwen3-8B-4bit")

    Returns:
        Dictionary with tool_call_parser, reasoning_parser, message_converter
        values (only matched options), or empty dict if no matches found.
    """
    from mlx_manager.utils.fuzzy_matcher import find_parser_options

    return find_parser_options(model_id)


def get_model_detection_info(model_id: str) -> dict:
    """
    Get full model detection information for API response.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Dictionary with model_family, recommended_options, is_downloaded,
        and version_support info.
    """
    family = detect_model_family(model_id)
    options = get_parser_options(model_id)  # Use filtered options
    is_downloaded = get_local_model_path(model_id) is not None

    # Check if mlx-lm supports this model family
    version_support = (
        check_mlx_lm_support(family)
        if family
        else {
            "supported": True,
            "installed_version": get_mlx_lm_version(),
        }
    )

    return {
        "model_family": family,
        "recommended_options": options,
        "is_downloaded": is_downloaded,
        "version_support": version_support,
    }


def detect_tool_use(config: dict[str, Any], tags: list[str] | None = None) -> bool:
    """
    Detect if a model supports tool-use / function-calling.

    Uses three detection strategies:
    1. Tag-based (primary): Check HuggingFace tags for tool-use indicators
    2. Family-based (secondary): Check if model_type matches known tool-capable families
    3. Config-based (fallback): Check config.json for tool_call_parser

    Args:
        config: Parsed config.json dictionary
        tags: Optional list of HuggingFace tags for the model

    Returns:
        True if model supports tool-use, False otherwise
    """
    # Tag-based detection (primary)
    if tags:
        tags_lower = [tag.lower() for tag in tags]
        tool_indicators = [
            "tool-use",
            "tool_use",
            "function-calling",
            "function_calling",
            "tool-calling",
            "tools",
        ]
        if any(indicator in tag for tag in tags_lower for indicator in tool_indicators):
            return True

    # Family-based detection (secondary) - check model_type against known families
    model_type = config.get("model_type", "").lower()
    if model_type in TOOL_CAPABLE_FAMILIES:
        return True
    # Also check for partial matches (e.g., "qwen2_vl" contains "qwen")
    for family in TOOL_CAPABLE_FAMILIES:
        if family in model_type:
            return True

    # Config-based detection (fallback)
    # Check for tool_call_parser in config (some models have this)
    if "tool_call_parser" in config:
        return True

    return False


def detect_multimodal(config: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Detect if a model is multimodal and its type.

    Checks for common multimodal indicators in config.json:
    - vision_config key (most reliable)
    - image_token_id, image_token_index, or video_token_id keys
    - "vl", "vision", or "multimodal" in model_type or architectures

    Args:
        config: Parsed config.json dictionary

    Returns:
        Tuple of (is_multimodal, multimodal_type) where multimodal_type
        is "vision" for vision-language models, None otherwise.
    """
    # Check for vision_config key (most common for VL models)
    if "vision_config" in config:
        return (True, "vision")

    # Check for image/video token IDs
    # (Gemma 3 uses image_token_index instead of image_token_id)
    if any(key in config for key in ("image_token_id", "image_token_index", "video_token_id")):
        return (True, "vision")

    # Check model_type for multimodal indicators
    model_type = config.get("model_type", "").lower()
    if any(indicator in model_type for indicator in ("vl", "vision", "multimodal")):
        return (True, "vision")

    # Check architectures list
    architectures = config.get("architectures", [])
    for arch in architectures:
        arch_lower = arch.lower()
        if any(indicator in arch_lower for indicator in ("vl", "vision", "multimodal")):
            return (True, "vision")

    return (False, None)


def normalize_architecture(config: dict[str, Any]) -> str:
    """
    Normalize model architecture to a family name.

    Extracts model_type from config and maps to a standardized family name
    using ARCHITECTURE_FAMILIES. Falls back to architectures[0] or "Unknown".

    Args:
        config: Parsed config.json dictionary

    Returns:
        Normalized architecture family name (e.g., "Llama", "Qwen", "Mistral")
    """
    model_type = config.get("model_type", "").lower()

    # Try exact match first
    if model_type in ARCHITECTURE_FAMILIES:
        return ARCHITECTURE_FAMILIES[model_type]

    # Try partial match (for variants like "qwen2_vl")
    for key, family in ARCHITECTURE_FAMILIES.items():
        if key in model_type:
            return family

    # Fall back to architectures field
    architectures = config.get("architectures", [])
    if architectures:
        arch = architectures[0].lower()
        # Try to match from architecture class name (e.g., "Qwen2ForCausalLM")
        for key, family in ARCHITECTURE_FAMILIES.items():
            if key in arch:
                return family

    return "Unknown"


def extract_characteristics(
    config: dict[str, Any], tags: list[str] | None = None
) -> ModelCharacteristics:
    """
    Extract model characteristics from config.json.

    Builds a ModelCharacteristics TypedDict from the config dictionary,
    handling missing fields gracefully with defaults.

    Args:
        config: Parsed config.json dictionary
        tags: Optional HuggingFace tags for tag-based detection

    Returns:
        ModelCharacteristics with all available fields populated
    """
    is_multimodal, multimodal_type = detect_multimodal(config)
    is_tool_use = detect_tool_use(config, tags)

    # Extract quantization info (MLX models often have this)
    quantization = config.get("quantization", {})
    quantization_bits = quantization.get("bits") if isinstance(quantization, dict) else None
    quantization_group_size = (
        quantization.get("group_size") if isinstance(quantization, dict) else None
    )

    characteristics: ModelCharacteristics = {
        "model_type": config.get("model_type", ""),
        "architecture_family": normalize_architecture(config),
        "is_multimodal": is_multimodal,
        "multimodal_type": multimodal_type,
        "use_cache": config.get("use_cache", True),
        "is_tool_use": is_tool_use,
    }

    # Add optional numeric fields if present
    if "max_position_embeddings" in config:
        characteristics["max_position_embeddings"] = config["max_position_embeddings"]

    if "num_hidden_layers" in config:
        characteristics["num_hidden_layers"] = config["num_hidden_layers"]

    if "hidden_size" in config:
        characteristics["hidden_size"] = config["hidden_size"]

    if "vocab_size" in config:
        characteristics["vocab_size"] = config["vocab_size"]

    if "num_attention_heads" in config:
        characteristics["num_attention_heads"] = config["num_attention_heads"]

    if "num_key_value_heads" in config:
        characteristics["num_key_value_heads"] = config["num_key_value_heads"]

    if quantization_bits is not None:
        characteristics["quantization_bits"] = quantization_bits

    if quantization_group_size is not None:
        characteristics["quantization_group_size"] = quantization_group_size

    return characteristics


def extract_characteristics_from_model(
    model_id: str, tags: list[str] | None = None
) -> ModelCharacteristics | None:
    """
    Extract characteristics from a locally downloaded model.

    Reads config.json from the HuggingFace cache and extracts characteristics.
    Returns None if the model is not downloaded or config.json is not available.

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        tags: Optional HuggingFace tags for tag-based detection

    Returns:
        ModelCharacteristics or None if model not available
    """
    config = read_model_config(model_id)
    if config is None:
        return None
    return extract_characteristics(config, tags)
