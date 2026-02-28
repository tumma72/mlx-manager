"""Model family detection.

This module provides model family detection from HuggingFace model IDs.
The actual adapters are now in composable.py.
"""

from loguru import logger

# Family pattern mappings (used by pool.py and probe utilities)
FAMILY_PATTERNS: dict[str, list[str]] = {
    "llama": ["llama", "codellama"],
    # nemotron before qwen: Qwen3-Coder models use Qwen3CoderXmlParser (nemotron family)
    "nemotron": ["nemotron", "qwen3-coder", "qwen3_coder"],
    "qwen": ["qwen", "iquest"],
    "mistral": ["mistral", "mixtral", "devstral", "magistral"],
    "functiongemma": ["functiongemma"],
    "gemma": ["gemma"],
    "glm4": ["glm", "chatglm"],
    "phi": ["phi"],
    "liquid": ["liquid", "lfm"],
    "whisper": ["whisper"],
    "kokoro": ["kokoro"],
}

# Architecture class → family mapping for fallback when name patterns fail.
# Only includes unambiguous architectures — LlamaForCausalLM is intentionally
# excluded because too many unrelated fine-tunes use it (EuroLLM, Yi, Vicuna, etc.).
ARCHITECTURE_TO_FAMILY: dict[str, str] = {
    "Qwen2ForCausalLM": "qwen",
    "Qwen2VLForConditionalGeneration": "qwen",
    "Qwen3ForCausalLM": "qwen",
    "MistralForCausalLM": "mistral",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma",
    "Gemma3ForCausalLM": "gemma",
    "Gemma3ForConditionalGeneration": "gemma",
    "PhiForCausalLM": "phi",
    "Phi3ForCausalLM": "phi",
    "ChatGLMModel": "glm4",
    "WhisperForConditionalGeneration": "whisper",
}


def detect_model_family(model_id: str, architecture: str | None = None) -> str:
    """Detect model family from HuggingFace model ID.

    Uses a two-tier strategy:
    1. Name pattern matching (highest priority — most specific)
    2. Architecture class lookup (fallback for unambiguous architectures)
    3. "default" if neither matches

    Args:
        model_id: e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit"
        architecture: Optional architecture class from config.json
            (e.g., "Qwen2ForCausalLM"). Used as fallback when name
            patterns don't match.

    Returns:
        Family name: "llama", "qwen", "mistral", "gemma", "glm4", "phi", or
        "default"
    """
    model_id_lower = model_id.lower()

    # Tier 1: Name pattern matching (most specific)
    for family, patterns in FAMILY_PATTERNS.items():
        if any(pattern in model_id_lower for pattern in patterns):
            return family

    # Tier 2: Architecture class fallback
    if architecture and architecture in ARCHITECTURE_TO_FAMILY:
        family = ARCHITECTURE_TO_FAMILY[architecture]
        logger.info(
            "Family for {} detected via architecture {} → {}",
            model_id,
            architecture,
            family,
        )
        return family

    # Default fallback
    logger.info("Unknown model family for {}, using default adapter", model_id)
    return "default"
