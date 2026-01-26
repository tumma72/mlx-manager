---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/utils/__init__.py
type: module
updated: 2026-01-21
status: active
---

# __init__.py (utils)

## Purpose

Re-exports utility functions from the utils submodules for convenient access. Provides a single import point for command building, fuzzy matching, model detection, and security validation utilities.

## Exports

- `build_mlx_server_command` - Build mlx-openai-server command from profile
- `find_mlx_openai_server` - Locate mlx-openai-server executable
- `get_server_log_path` - Get log file path for a profile
- `find_parser_options` - Fuzzy match parser options for a model
- `MODEL_FAMILY_MIN_VERSIONS` - Version requirements by model family
- `check_mlx_lm_support` - Check mlx-lm version compatibility
- `detect_model_family` - Detect model family from config
- `get_mlx_lm_version` - Get installed mlx-lm version
- `get_model_detection_info` - Full model detection info
- `get_parser_options` - Get recommended parser options
- `validate_model_path` - Security validation for model paths

## Dependencies

- [[backend-mlx_manager-utils-command_builder]] - Command building utilities
- [[backend-mlx_manager-utils-fuzzy_matcher]] - Parser option fuzzy matching
- [[backend-mlx_manager-utils-model_detection]] - Model family detection
- [[backend-mlx_manager-utils-security]] - Path validation

## Used By

TBD
