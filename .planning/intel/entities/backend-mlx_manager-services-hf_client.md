---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/services/hf_client.py
type: service
updated: 2026-01-21
status: active
---

# hf_client.py

## Purpose

Service for interacting with HuggingFace Hub for model operations. Provides model search with accurate sizes, local model listing with characteristics extraction, model downloads with progress streaming, and model deletion. Uses the REST API for search and huggingface_hub SDK for downloads which handles complex caching and LFS.

## Exports

- `SilentProgress` - tqdm subclass that suppresses console output
- `HuggingFaceClient` - Main client class
- `hf_client` - Singleton instance

## Dependencies

- [[backend-mlx_manager-config]] - Settings for cache path, organization, offline mode
- [[backend-mlx_manager-services-hf_api]] - REST API wrapper for search
- [[backend-mlx_manager-types]] - Type definitions
- [[backend-mlx_manager-utils-model_detection]] - Characteristics extraction
- huggingface_hub - SDK for snapshot_download

## Used By

TBD

## Notes

Download progress is tracked by polling directory size since snapshot_download doesn't provide callbacks. Dry run is used first to get total size. Respects offline_mode setting to prevent network access.
