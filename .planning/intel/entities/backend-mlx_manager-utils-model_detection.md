---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/utils/model_detection.py
type: util
updated: 2026-01-21
status: active
---

# model_detection.py

## Purpose

Provides offline-first detection of model families and characteristics from locally cached models. Reads config.json directly from the HuggingFace cache filesystem without network calls. Used to recommend parser options, detect multimodal capabilities, normalize architecture families, and check mlx-lm version compatibility for specific model families.

## Exports

- `ARCHITECTURE_FAMILIES` - Mapping of model types to normalized family names
- `MODEL_FAMILY_MIN_VERSIONS` - Minimum mlx-lm versions required for model families
- `get_mlx_lm_version() -> str | None` - Get installed mlx-lm version
- `parse_version(v: str) -> tuple[int, ...]` - Parse version string for comparison
- `check_mlx_lm_support(model_family: str) -> dict` - Check version compatibility
- `get_local_model_path(model_id: str) -> Path | None` - Get cached model path
- `read_model_config(model_id: str) -> dict | None` - Read config.json from cache
- `detect_model_family(model_id: str) -> str | None` - Detect model family
- `get_parser_options(model_id: str) -> dict[str, str]` - Get recommended parsers via fuzzy matching
- `get_model_detection_info(model_id: str) -> dict` - Full detection info for API
- `detect_multimodal(config: dict) -> tuple[bool, str | None]` - Detect VL models
- `normalize_architecture(config: dict) -> str` - Normalize architecture family
- `extract_characteristics(config: dict) -> ModelCharacteristics` - Extract model characteristics
- `extract_characteristics_from_model(model_id: str) -> ModelCharacteristics | None` - Extract from local model

## Dependencies

- [[backend-mlx_manager-config]] - Settings for HF cache path
- [[backend-mlx_manager-types]] - ModelCharacteristics TypedDict
- [[backend-mlx_manager-utils-fuzzy_matcher]] - Fuzzy matching for parser options

## Used By

TBD
