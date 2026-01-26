---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/services/hf_api.py
type: service
updated: 2026-01-21
status: active
---

# hf_api.py

## Purpose

Direct REST API wrapper for HuggingFace Hub, bypassing the huggingface_hub SDK for search operations. Provides model search with accurate size data via parallel usedStorage API calls, name-based size estimation as fallback, and remote config.json fetching. The SDK is still used for downloads which handles complex caching and LFS.

## Exports

- `HF_API_BASE` - HuggingFace API base URL constant
- `DEFAULT_TIMEOUT` - Default request timeout
- `ModelInfo` - Dataclass for model metadata
- `estimate_size_from_name(model_id: str) -> float | None` - Estimate size from naming conventions
- `search_models(query, author, sort, limit, timeout) -> list[ModelInfo]` - Search MLX models with accurate sizes
- `get_model_size_gb(model: ModelInfo) -> float` - Get size using usedStorage or estimation
- `fetch_remote_config(model_id, timeout) -> dict | None` - Fetch config.json from HuggingFace

## Dependencies

- httpx - Async HTTP client

## Used By

TBD

## Notes

Parallel API calls for usedStorage provide accurate total repository sizes. Size estimation from model names follows MLX naming conventions (e.g., "8B-4bit" -> ~4.1 GiB).
