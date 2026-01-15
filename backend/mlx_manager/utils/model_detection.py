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

logger = logging.getLogger(__name__)

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


# Parser options for different model families.
# These correspond to --tool-call-parser, --reasoning-parser, and --message-converter
# options in mlx-openai-server.
MODEL_PARSER_CONFIGS: dict[str, dict[str, str]] = {
    "minimax": {
        "tool_call_parser": "minimax_m2",
        "reasoning_parser": "minimax_m2",
        "message_converter": "minimax_m2",
    },
    "qwen": {
        "tool_call_parser": "qwen3",
        "reasoning_parser": "qwen3",
        "message_converter": "qwen3",
    },
    "glm": {
        "tool_call_parser": "glm4",
        "reasoning_parser": "glm4",
        "message_converter": "glm4",
    },
}

# Available parser options for frontend dropdown
AVAILABLE_PARSERS = [
    "minimax_m2",
    "qwen3",
    "glm4",
    "hermes",
    "llama",
    "mistral",
]


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
        Model family name (e.g., "minimax", "qwen", "glm") or None if unknown.
    """
    # Try to read config.json from local cache (OFFLINE - no API calls)
    config = read_model_config(model_id)
    if config:
        # Check model_type field
        model_type = config.get("model_type", "").lower()
        for family in MODEL_PARSER_CONFIGS:
            if family in model_type:
                logger.debug(f"Detected {family} from model_type: {model_type}")
                return family

        # Check architectures field
        architectures = config.get("architectures", [])
        for family in MODEL_PARSER_CONFIGS:
            if any(family in arch.lower() for arch in architectures):
                logger.debug(f"Detected {family} from architectures: {architectures}")
                return family

    # Fallback: check model path name (works even if not downloaded)
    path_lower = model_id.lower()
    for family in MODEL_PARSER_CONFIGS:
        if family in path_lower:
            logger.debug(f"Detected {family} from model path: {model_id}")
            return family

    return None


def get_parser_options(model_id: str) -> dict[str, str]:
    """
    Get recommended parser options for a model.

    Works OFFLINE - no network calls required.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Dictionary with tool_call_parser, reasoning_parser, message_converter
        values, or empty dict if no special configuration needed.
    """
    family = detect_model_family(model_id)
    if family:
        return MODEL_PARSER_CONFIGS.get(family, {})
    return {}


def get_model_detection_info(model_id: str) -> dict:
    """
    Get full model detection information for API response.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Dictionary with model_family, recommended_options, is_downloaded,
        available_parsers, and version_support info.
    """
    family = detect_model_family(model_id)
    options = MODEL_PARSER_CONFIGS.get(family, {}) if family else {}
    is_downloaded = get_local_model_path(model_id) is not None

    # Check if mlx-lm supports this model family
    version_support = check_mlx_lm_support(family) if family else {
        "supported": True,
        "installed_version": get_mlx_lm_version(),
    }

    return {
        "model_family": family,
        "recommended_options": options,
        "is_downloaded": is_downloaded,
        "available_parsers": AVAILABLE_PARSERS,
        "version_support": version_support,
    }
