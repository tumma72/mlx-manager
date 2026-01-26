---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/utils/fuzzy_matcher.py
type: util
updated: 2026-01-21
status: active
---

# fuzzy_matcher.py

## Purpose

Implements fuzzy string matching to dynamically find the best parser option for a given model name. Replaces static model-to-parser mappings with dynamic matching that scales to 100+ model variants. Uses substring containment with token matching and prefers more specific options (e.g., qwen3_coder over qwen3).

## Exports

- `FuzzyMatcher` - Abstract base class for fuzzy matchers
- `RapidfuzzMatcher` - Implementation using rapidfuzz library (C++ optimized, ~10x faster)
- `DifflibMatcher` - Fallback using Python's built-in difflib
- `get_matcher() -> FuzzyMatcher` - Get configured matcher (prefers rapidfuzz)
- `find_parser_options(model_id: str) -> dict[str, str]` - Main entry point for fuzzy matching

## Dependencies

- [[backend-mlx_manager-services-parser_options]] - Available parser options
- rapidfuzz - Fast fuzzy matching library (optional)
- difflib - Python stdlib fallback

## Used By

TBD

## Notes

Matching strategy: 1) Check for base family name in model name, 2) If option has variant (coder, moe, vl), variant must also be present, 3) Base options only match if no known variant is in model name. This prevents base Qwen3 from matching when Qwen3-Coder should match.
