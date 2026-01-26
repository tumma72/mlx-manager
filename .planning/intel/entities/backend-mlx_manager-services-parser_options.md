---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/services/parser_options.py
type: service
updated: 2026-01-21
status: active
---

# parser_options.py

## Purpose

Dynamically discovers available parser options from the installed mlx-openai-server package. Imports TOOL_PARSER_MAP, REASONING_PARSER_MAP, UNIFIED_PARSER_MAP, and MESSAGE_CONVERTER_MAP at runtime to stay in sync with the installed version. Falls back to known values if the package is not installed.

## Exports

- `ParserOptions` - TypedDict for parser option lists
- `FALLBACK_TOOL_PARSERS` - Fallback tool parser set
- `FALLBACK_REASONING_PARSERS` - Fallback reasoning parser set
- `FALLBACK_MESSAGE_CONVERTERS` - Fallback message converter set
- `get_parser_options() -> ParserOptions` - Get available parser options (cached)

## Dependencies

- app.parsers (mlx-openai-server) - Parser maps (optional)
- app.message_converters (mlx-openai-server) - Converter map (optional)

## Used By

TBD

## Notes

Results are cached via lru_cache for process lifetime. Unified parsers are added to both tool and reasoning parser lists.
