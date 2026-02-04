"""HuggingFace REST API wrapper.

This module provides direct access to the HuggingFace Hub REST API,
bypassing the huggingface_hub SDK for search operations. This allows us to:
1. Fetch accurate model sizes via usedStorage (requires per-model API calls
   since usedStorage is not available as an expand option in the list endpoint)
2. Run parallel requests for efficient batch size fetching
3. Have more control over request parameters

The SDK is still used for snapshot_download which handles complex
download logic (resumable, LFS, caching, etc.).
"""

import asyncio
import re
from dataclasses import dataclass

import httpx
from loguru import logger

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
    size_bytes: int | None  # From usedStorage (accurate total repository size)


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


async def _fetch_model_size(
    client: httpx.AsyncClient,
    model_id: str,
    timeout: float,
) -> tuple[str, int | None]:
    """Fetch the usedStorage for a single model.

    Args:
        client: Shared httpx client
        model_id: HuggingFace model ID
        timeout: Request timeout

    Returns:
        Tuple of (model_id, size_bytes or None if failed)
    """
    url = f"{HF_API_BASE}/models/{model_id}"
    try:
        response = await client.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return (model_id, data.get("usedStorage"))
    except Exception as e:
        logger.debug(f"Failed to fetch size for {model_id}: {e}")
    return (model_id, None)


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

    After the initial search, fetches accurate sizes via parallel API calls
    to get usedStorage for each model. This provides accurate download sizes
    rather than estimates.

    Args:
        query: Search query string
        author: Optional filter by author/organization (None = all authors)
        sort: Sort field (downloads, likes, lastModified)
        limit: Maximum number of results
        timeout: Request timeout in seconds

    Returns:
        List of ModelInfo objects with model metadata and accurate sizes.
    """
    url = f"{HF_API_BASE}/models"
    params: dict[str, str | list[str]] = {
        "search": query,
        "filter": "mlx",  # Filter by MLX library - finds models from any author
        "sort": sort,
        "limit": str(limit),
        # Note: usedStorage is not available as an expand option in list endpoint
        # We fetch it separately via parallel per-model API calls
    }

    # Optionally filter by specific author
    if author:
        params["author"] = author

    async with httpx.AsyncClient() as client:
        # Step 1: Fetch search results
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

        search_results = response.json()

        # Step 2: Fetch accurate sizes in parallel
        # Use shorter timeout for individual size fetches to avoid slowing down search
        size_timeout = min(timeout, 10.0)
        size_tasks = [
            _fetch_model_size(client, item.get("id", item.get("modelId", "")), size_timeout)
            for item in search_results
        ]
        size_results = await asyncio.gather(*size_tasks)
        size_map = {model_id: size for model_id, size in size_results}

    # Step 3: Build results with accurate sizes
    results: list[ModelInfo] = []

    for item in search_results:
        model_id = item.get("id", item.get("modelId", ""))
        # Use usedStorage from parallel fetch (accurate total repository size)
        size_bytes = size_map.get(model_id)

        results.append(
            ModelInfo(
                model_id=model_id,
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
    """Get model size in GiB, using usedStorage or name estimation as fallback.

    Args:
        model: ModelInfo object from search

    Returns:
        Size in GiB (binary, 1024^3) - estimated if not available from API.
    """
    # Use usedStorage if available (accurate total repository size)
    if model.size_bytes:
        return round(model.size_bytes / (1024**3), 2)

    # Fall back to name-based estimation
    estimated = estimate_size_from_name(model.model_id)
    if estimated:
        return estimated

    # Unknown size
    return 0.0


async def fetch_remote_config(
    model_id: str,
    timeout: float = 10.0,
) -> dict[str, object] | None:
    """Fetch config.json from a HuggingFace model repository.

    Uses the resolve API to fetch the config.json file directly.

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        timeout: Request timeout in seconds

    Returns:
        Parsed config.json as a dictionary, or None if not available.
    """
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout)
            if response.status_code == 200:
                data: dict[str, object] = response.json()
                return data
            logger.debug(f"Config fetch for {model_id} returned status {response.status_code}")
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching config for {model_id}")
    except httpx.RequestError as e:
        logger.warning(f"Error fetching config for {model_id}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error fetching config for {model_id}: {e}")

    return None
