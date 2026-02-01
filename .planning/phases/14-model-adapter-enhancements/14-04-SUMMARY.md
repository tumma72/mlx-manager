---
phase: 14
plan: 04
subsystem: mlx-server-inference
tags: [jsonschema, validation, structured-output, json-schema]

dependency-graph:
  requires: ["14-01"]
  provides: ["StructuredOutputValidator", "ValidationResult"]
  affects: ["14-05"]

tech-stack:
  added: ["jsonschema>=4.0.0"]
  patterns: ["Draft202012Validator", "type-coercion", "json-extraction"]

key-files:
  created:
    - backend/mlx_manager/mlx_server/services/structured_output.py
  modified:
    - backend/pyproject.toml
    - backend/uv.lock

decisions:
  - "Draft202012Validator for modern JSON Schema support"
  - "Error paths use dot.notation for objects and [N] for arrays"
  - "Type coercion for common LLM issues (string->int, string->bool)"

metrics:
  duration: "3 min"
  completed: "2026-02-01"
---

# Phase 14 Plan 04: Structured Output Validation Summary

**One-liner:** JSON Schema validation service with type coercion and JSON extraction for LLM outputs using jsonschema library.

## What Changed

### New Files

**backend/mlx_manager/mlx_server/services/structured_output.py**

Structured output validation service with three main capabilities:

1. **ValidationResult dataclass** - Result container with:
   - `success: bool` - Whether validation passed
   - `data: dict | list | None` - Parsed JSON if valid
   - `error: str | None` - Error message if invalid
   - `error_path: str | None` - Path to failing element (e.g., `user.age`, `items[1].id`)

2. **StructuredOutputValidator.validate()** - Core validation:
   - Parses output as JSON
   - Validates against provided JSON Schema
   - Returns detailed error paths for nested failures
   - Uses `$` for root-level errors

3. **StructuredOutputValidator.extract_json()** - JSON extraction:
   - Finds JSON in text with surrounding content
   - Handles nested brackets with depth tracking
   - Useful for models that output explanatory text with JSON

4. **StructuredOutputValidator.validate_and_coerce()** - Type coercion:
   - Converts string `"5"` to integer `5` for integer fields
   - Converts string `"3.14"` to float for number fields
   - Converts string `"true"`/`"false"` to boolean
   - Handles `"null"` to `None` for nullable fields

### Modified Files

**backend/pyproject.toml**
- Added `jsonschema>=4.0.0` dependency for JSON Schema validation

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 5eef98a | chore | Add jsonschema dependency for structured output validation |
| 0e53677 | feat | Implement StructuredOutputValidator service |
| 99a214b | style | Format structured_output.py with ruff |

## Decisions Made

1. **Draft202012Validator** - Used the latest JSON Schema draft validator for modern schema support including advanced features like `$ref` and `additionalProperties`.

2. **Error path format** - Used dot notation for object properties (`user.age`) and bracket notation for array indices (`items[1].id`) for familiar JSONPath-like syntax.

3. **Type coercion strategy** - Implemented schema-guided coercion that only converts types when the schema explicitly expects a different type, preventing unintended conversions.

## Verification Results

All criteria verified:

- [x] jsonschema>=4.0.0 in pyproject.toml dependencies
- [x] jsonschema importable after uv sync (v4.26.0)
- [x] StructuredOutputValidator.validate() returns ValidationResult
- [x] Valid JSON passes validation
- [x] Invalid JSON returns error with path
- [x] Malformed JSON returns parse error
- [x] extract_json() finds JSON in mixed content
- [x] Quality gates pass (ruff check, ruff format)

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

Plan 04 provides the StructuredOutputValidator service that can be used by:
- **Plan 05** (if it exists) - Integration with inference service for response_format validation
- **Inference service** - Validating model outputs against user-provided schemas
- **Chat completion router** - Enforcing structured output requirements

The service is complete and ready for integration.
