---
phase: 15-code-cleanup-integration-tests
plan: 18
subsystem: testing
tags: [probe, tdd, generative-probe, sweep-capabilities, refactoring]

# Dependency graph
requires:
  - phase: 15-17
    provides: family-aware parser selection + _prioritize_parsers logic in base.py
provides:
  - GenerativeProbe.sweep_capabilities() public method owning all sweep logic
  - ProbingCoordinator delegates to strategy via isinstance(strategy, GenerativeProbe)
  - _sweep_generative_capabilities() removed from ProbingCoordinator
affects: [probe package consumers, future probe strategy implementations]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Strategy owns its own sweep logic — GenerativeProbe.sweep_capabilities() is called by coordinator via isinstance check, eliminating the 'strategy passed as argument' antipattern"

key-files:
  created:
    - backend/tests/test_probe_sweep_capabilities.py
  modified:
    - backend/mlx_manager/services/probe/base.py
    - backend/mlx_manager/services/probe/coordinator.py
    - backend/tests/test_probe_coverage_new.py
    - backend/tests/test_probe_package.py

key-decisions:
  - "GenerativeProbe.sweep_capabilities() passes self as the strategy to sweep_thinking/sweep_tools — self IS the strategy so no separate parameter needed"
  - "Coordinator uses isinstance(strategy, GenerativeProbe) check instead of ModelType enum check — cleaner design that works automatically for any future GenerativeProbe subclasses"
  - "Pre-existing mypy errors in sweeps.py and base.py (3 errors) are unchanged — not introduced by this plan"

patterns-established:
  - "Delegation pattern: coordinator delegates to strategy.sweep_capabilities() for GenerativeProbe subclasses, non-generative strategies (embeddings, audio) get no sweep"

# Metrics
duration: 22min
completed: 2026-02-25
---

# Phase 15 Plan 18: GenerativeProbe.sweep_capabilities() Summary

**Moved `_sweep_generative_capabilities()` from `ProbingCoordinator` into `GenerativeProbe.sweep_capabilities()` using TDD — the strategy now owns its own sweep logic instead of receiving itself as an argument.**

## Performance

- **Duration:** 22 min
- **Started:** 2026-02-25T13:18:21Z
- **Completed:** 2026-02-25T13:40:35Z
- **Tasks:** 4 (RED test, GREEN impl, update existing tests, full suite)
- **Files modified:** 4 (+ 1 created)

## Accomplishments

- Added `sweep_capabilities(model_id, loaded, result)` to `GenerativeProbe` in `base.py` — moves the entire 136-line sweep body from `coordinator.py`
- The method passes `self` as the strategy to `sweep_thinking`/`sweep_tools` — eliminates the odd "strategy passed to its own method" antipattern
- `ProbingCoordinator` step 6 now uses `isinstance(strategy, GenerativeProbe)` and calls `strategy.sweep_capabilities()` — 3 lines replacing 4+ lines
- Removed `_sweep_generative_capabilities()` from `ProbingCoordinator` entirely (-146 lines in coordinator)
- Updated 6 tests in `test_probe_coverage_new.py` from calling `coordinator._sweep_generative_capabilities()` to `strategy.sweep_capabilities()`
- Updated 3 docstrings in `test_probe_package.py` to reference new method location
- 15 new tests in `test_probe_sweep_capabilities.py` covering all sweep_capabilities behaviors
- Full test suite: 2740 backend tests pass, 1070 total pass

## TDD Commits

| Phase | Commit | Description |
|-------|--------|-------------|
| RED | 9c590d4 | test(15-18): add failing tests for GenerativeProbe.sweep_capabilities() |
| GREEN | 18c9ba3 | feat(15-18): move sweep_capabilities from coordinator into GenerativeProbe |
| Update | 982cce9 | refactor(15-18): update existing tests to call strategy.sweep_capabilities() |

## Deviations from Plan

None — plan executed exactly as written.

## Next Phase Readiness

All quality gates pass:
- ruff: clean
- mypy: 3 pre-existing errors (unchanged, not introduced here)
- pytest: 2740 backend tests pass
- coverage: 96% backend
