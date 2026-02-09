"""Model family detection.

This module provides model family detection from HuggingFace model IDs.
The actual adapters are now in composable.py.
"""

from loguru import logger

# Family pattern mappings (used by pool.py and probe utilities)
FAMILY_PATTERNS = {
    "llama": ["llama", "codellama"],
    "qwen": ["qwen"],
    "mistral": ["mistral", "mixtral"],
    "gemma": ["gemma"],
    "glm4": ["glm", "chatglm"],
    "phi": ["phi"],
}


def detect_model_family(model_id: str) -> str:
    """Detect model family from HuggingFace model ID.

    Args:
        model_id: e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit"

    Returns:
        Family name: "llama", "qwen", "mistral", "gemma", "glm4", "phi", or
        "default"
    """
    model_id_lower = model_id.lower()

    # Check each family's patterns
    for family, patterns in FAMILY_PATTERNS.items():
        if any(pattern in model_id_lower for pattern in patterns):
            return family

    # Default fallback
    logger.info("Unknown model family for {}, using default adapter", model_id)
    return "default"
