---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/utils/security.py
type: util
updated: 2026-01-21
status: active
---

# security.py

## Purpose

Security utilities for validating user inputs. Currently provides model path validation to ensure paths are within allowed directories, preventing path traversal attacks. Also allows HuggingFace model IDs which contain a slash but don't start with one.

## Exports

- `validate_model_path(path: str) -> bool` - Validate model path is within allowed directories

## Dependencies

- [[backend-mlx_manager-config]] - Settings for allowed_model_dirs

## Used By

TBD

## Notes

Resolves paths to absolute form before checking against allowed directories. Allows HuggingFace model ID format (contains "/" but doesn't start with "/").
