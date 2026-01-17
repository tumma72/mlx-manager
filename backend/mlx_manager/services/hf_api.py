"""HuggingFace REST API wrapper.

This module provides direct access to the HuggingFace Hub REST API,
bypassing the huggingface_hub SDK for search operations. This allows us to:
1. Use expand=safetensors to get model sizes in a single request
2. Avoid N+1 API calls that made search slow
3. Have more control over request parameters

The SDK is still used for snapshot_download which handles complex
download logic (resumable, LFS, caching, etc.).
"""

import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# HuggingFace API base URL
HF_API_BASE = "https://huggingface.co/api"

# Default timeout for API requests (seconds)
DEFAULT_TIMEOUT = 30.0


@dataclass
class ModelInfo:
    """Model information from HuggingFace API."""

    model_id: str
    author: str | None
    downloads: int
    likes: int
    tags: list[str]
    last_modified: str | None
    size_bytes: int | None  # From safetensors.total if available


def estimate_size_from_name(model_id: str) -> float | None:
    """Estimate model size in GiB from model name patterns.

    MLX models typically follow naming conventions like:
    - "Qwen3-8B-4bit" -> 8B params at 4-bit ≈ 4.1 GiB
    - "Llama-3.1-70B-8bit" -> 70B params at 8-bit ≈ 65 GiB
    - "Mistral-7B-bf16" -> 7B params at bf16 ≈ 13 GiB

    Args:
        model_id: HuggingFace model ID

    Returns:
        Estimated size in GiB (binary, 1024^3), or None if cannot estimate.
    """
    model_name = model_id.split("/")[-1].lower()

    # Extract parameter count (e.g., "8b", "70b", "1.7b")
    param_match = re.search(r"(\d+\.?\d*)b(?![a-z])", model_name)
    if not param_match:
        return None

    params_billions = float(param_match.group(1))

    # Determine bytes per parameter based on quantization
    # Default to 4-bit if not specified (most common for MLX)
    if "bf16" in model_name or "fp16" in model_name or "16bit" in model_name:
        bytes_per_param = 2.0
    elif "8bit" in model_name or "8-bit" in model_name:
        bytes_per_param = 1.0
    elif "4bit" in model_name or "4-bit" in model_name:
        bytes_per_param = 0.5
    elif "3bit" in model_name or "3-bit" in model_name:
        bytes_per_param = 0.375
    elif "2bit" in model_name or "2-bit" in model_name:
        bytes_per_param = 0.25
    else:
        # Default to 4-bit for MLX models
        bytes_per_param = 0.5

    # Calculate total bytes: params * bytes_per_param + 10% overhead
    total_bytes = params_billions * 1e9 * bytes_per_param * 1.1

    # Convert to GiB (binary)
    size_gib = total_bytes / (1024**3)

    return round(size_gib, 2)


async def search_models(
    query: str,
    author: str | None = None,
    sort: str = "downloads",
    limit: int = 20,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[ModelInfo]:
    """Search for MLX-optimized models on HuggingFace Hub.

    Uses the REST API directly to search for models with the MLX library tag,
    finding models from any author (mlx-community, lmstudio-community, etc.).

    Args:
        query: Search query string
        author: Optional filter by author/organization (None = all authors)
        sort: Sort field (downloads, likes, lastModified)
        limit: Maximum number of results
        timeout: Request timeout in seconds

    Returns:
        List of ModelInfo objects with model metadata.
    """
    url = f"{HF_API_BASE}/models"
    params: dict[str, str | list[str]] = {
        "search": query,
        "filter": "mlx",  # Filter by MLX library - finds models from any author
        "sort": sort,
        "limit": str(limit),
        # Expand safetensors to get model weights size
        # Note: usedStorage is not a valid expand option (causes 400 error)
        "expand[]": ["safetensors"],
    }

    # Optionally filter by specific author
    if author:
        params["author"] = author

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=timeout)
            response.raise_for_status()
        except httpx.TimeoutException:
            logger.warning(f"HuggingFace API timeout after {timeout}s")
            return []
        except httpx.HTTPStatusError as e:
            logger.warning(f"HuggingFace API error: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            logger.warning(f"HuggingFace API request failed: {e}")
            return []

    results: list[ModelInfo] = []

    for item in response.json():
        # Get size from safetensors.total (total size of model weights)
        size_bytes = None
        safetensors = item.get("safetensors")
        if safetensors and isinstance(safetensors, dict):
            size_bytes = safetensors.get("total")

        results.append(
            ModelInfo(
                model_id=item.get("id", item.get("modelId", "")),
                author=item.get("author"),
                downloads=item.get("downloads", 0),
                likes=item.get("likes", 0),
                tags=item.get("tags", []),
                last_modified=item.get("lastModified"),
                size_bytes=size_bytes,
            )
        )

    return results


def get_model_size_gb(model: ModelInfo) -> float:
    """Get model size in GiB, using safetensors data or name estimation.

    Args:
        model: ModelInfo object from search

    Returns:
        Size in GiB (binary, 1024^3) - estimated if not available from API.
    """
    # Use safetensors total if available (convert to binary GiB)
    if model.size_bytes:
        return round(model.size_bytes / (1024**3), 2)

    # Fall back to name-based estimation
    estimated = estimate_size_from_name(model.model_id)
    if estimated:
        return estimated

    # Unknown size
    return 0.0
