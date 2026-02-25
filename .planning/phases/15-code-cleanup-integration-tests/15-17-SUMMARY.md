---
phase: 15
plan: 17
subsystem: probe-system
tags: [probe, parser-selection, tdd, family-detection, nemotron, qwen3-coder]

dependency:
  requires: [15-16]
  provides: [family-aware-parser-priority, qwen3-coder-family-routing]
  affects: []

tech-stack:
  added: []
  patterns: [tdd-red-green, family-aware-routing, parser-prioritization]

key-files:
  created:
    - backend/tests/test_probe_parser_priority.py
  modified:
    - backend/mlx_manager/mlx_server/models/adapters/registry.py
    - backend/mlx_manager/services/probe/base.py
    - backend/mlx_manager/services/probe/coordinator.py

decisions:
  - nemotron-before-qwen: FAMILY_PATTERNS dict ordering puts nemotron entry before qwen so Qwen3-Coder model IDs match nemotron (more specific) before qwen (broader)
  - qwen3-coder-patterns: Added "qwen3-coder" and "qwen3_coder" to nemotron patterns to catch both hyphen and underscore variants
  - lazy-imports-in-methods: _prioritize_parsers and get_family_*_parser_id imported lazily inside _sweep_* methods to avoid circular imports
  - stdlib-logger-format: coordinator.py uses standard library logger; format strings must use %s not {} to avoid TypeError in InterceptHandler

metrics:
  duration: 52m 14s
  completed: 2026-02-25
---

# Phase 15 Plan 17: Family-Aware Parser Selection (TDD) Summary

Family-aware parser prioritization fixes Qwen3-Coder model routing to wrong parsers. Root causes: FAMILY_PATTERNS dict ordering let "qwen" match Qwen3-Coder before "nemotron", and alphabetical parser validation order caused false-positives when multiple parsers share the same stream marker (`<tool_call>`).

## Tasks Completed

| Task | Description | Commit | Key Files |
|------|-------------|--------|-----------|
| RED  | Write failing tests for family detection, parser prioritization, and family lookup | d138a38 | tests/test_probe_parser_priority.py |
| GREEN | Fix FAMILY_PATTERNS ordering + add helper functions + wire coordinator sweeps | 4a8b50a | registry.py, base.py, coordinator.py |

## Changes Summary

### FAMILY_PATTERNS reordering (registry.py)

Moved `"nemotron"` entry before `"qwen"` in `FAMILY_PATTERNS`. Added `"qwen3-coder"` and `"qwen3_coder"` as nemotron patterns. Since `detect_model_family()` iterates the dict in insertion order, more-specific families must come before broader ones.

Before: `Qwen3-Coder-7B` → matched `"qwen"` (wrong parser: HermesJsonParser)
After: `Qwen3-Coder-7B` → matches `"nemotron"` (correct parser: Qwen3CoderXmlParser)

### Helper functions added to base.py

Three new module-level functions:

- `_prioritize_parsers(candidates, family_parser_id)` — returns family-declared parser first, then sorted alphabetically. Used in Phase 3 VALIDATE of both sweeps.
- `get_family_tool_parser_id(family)` — looks up the tool parser ID from `FamilyConfig.tool_parser_factory`.
- `get_family_thinking_parser_id(family)` — looks up the thinking parser ID from `FamilyConfig.thinking_parser_factory`.

### Coordinator wiring (coordinator.py)

- `_sweep_thinking()`: added `family: str | None = None` parameter; Phase 3 VALIDATE uses `_prioritize_parsers` with `get_family_thinking_parser_id(family)` instead of `sorted()`.
- `_sweep_tools()`: added `family: str | None = None` parameter; Phase 3 VALIDATE uses `_prioritize_parsers` with `get_family_tool_parser_id(family)` instead of `sorted()`.
- `_sweep_generative_capabilities()`: passes `result.model_family` to both sweep calls.

### Bug fix (Rule 1 - coordinator.py)

Pre-existing bug: `logger.warning()` used `{}` format string with the standard library logger (which uses `%s` format). The `InterceptHandler` in `logging_config.py` calls `record.getMessage()` which applies `%` formatting, causing `TypeError: not all arguments converted`. Fixed format string to use `%s`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed logger.warning format string in coordinator.py**

- **Found during:** Running full probe test suite (STEP 5)
- **Issue:** `logger.warning("...{}", value)` fails in `InterceptHandler` because stdlib logger uses `%` formatting; `{}` format leaves the arg unformatted
- **Fix:** Changed `{}` to `%s` in the tokenization artifacts warning message
- **Files modified:** `backend/mlx_manager/services/probe/coordinator.py`
- **Commit:** 4a8b50a

## Test Results

- Tests written: 19 (3 classes: TestFamilyDetection, TestParserPrioritization, TestFamilyParserLookup)
- RED phase: 16/19 failed (3 pre-existing tests for nemotron/qwen already passed)
- GREEN phase: 19/19 pass
- Full probe suite: 260/260 pass (was 259/260 due to pre-existing logger bug now fixed)

## Next Phase Readiness

No blockers. Parser selection is now correct for Qwen3-Coder models. The fix is backward-compatible: all other families unaffected, `family` parameter defaults to `None` in sweep methods.
