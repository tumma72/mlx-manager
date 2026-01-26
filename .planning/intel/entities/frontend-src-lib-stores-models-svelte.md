---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/stores/models.svelte.ts
type: hook
updated: 2026-01-21
status: active
---

# models.svelte.ts

## Purpose

Svelte 5 runes-based store for lazy-loading and caching model characteristics (config.json). Fetches model configuration on demand via the API and caches results to avoid redundant requests. Falls back to parsing model name/tags when config.json is unavailable from the backend.

## Exports

- `ConfigState` - Interface for cached config state
- `ARCHITECTURE_PATTERNS` - Regex patterns for detecting architecture family
- `QUANTIZATION_PATTERNS` - Regex patterns for detecting quantization bits
- `MULTIMODAL_PATTERNS` - Regex patterns for detecting multimodal models
- `parseCharacteristicsFromName(modelId, tags?) -> ModelCharacteristics` - Fallback name parsing
- `ModelConfigStore` - Class with getConfig, fetchConfig, clearConfig, clearAll methods
- `modelConfigStore` - Singleton store instance

## Dependencies

- [[frontend-src-lib-api-client]] - API client for model config fetching
- [[frontend-src-lib-api-types]] - ModelCharacteristics type

## Used By

TBD

## Notes

Uses Map with reassignment pattern for Svelte 5 reactivity. Prevents infinite loops by tracking all fetch attempts, not just successful ones.
