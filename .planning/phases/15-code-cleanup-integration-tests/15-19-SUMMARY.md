---
phase: 15-code-cleanup-integration-tests
plan: 19
subsystem: probe
tags: [probe, tdd, timeout, asyncio, generative-probe, vision-probe]

# Dependency graph
requires:
  - phase: 15-18
    provides: GenerativeProbe.sweep_capabilities() public method
provides:
  - Timeout protection in GenerativeProbe._generate() (default 60s)
  - Timeout protection in VisionProbe._generate() (same pattern)
  - Descriptive TimeoutError with timeout value and max_tokens in message
affects: [probe execution, sweeps.py callers, any code calling _generate()]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "asyncio.wait_for() wraps blocking adapter.generate() calls in probe methods to prevent indefinite hangs"
    - "Built-in TimeoutError (not asyncio.TimeoutError) in except clauses per ruff UP041"

key-files:
  created:
    - backend/tests/test_probe_generation_timeout.py
  modified:
    - backend/mlx_manager/services/probe/base.py
    - backend/mlx_manager/services/probe/vision.py

key-decisions:
  - "timeout=60.0 default keeps sweeps.py callers unchanged — no param threading needed"
  - "Catch TimeoutError (builtin) not asyncio.TimeoutError — ruff UP041 prefers builtin alias"
  - "VisionProbe overrides _generate() so must independently add timeout — cannot inherit it"
  - "PIL.Image patched at PIL module level in tests since it is imported locally inside _generate() not at module level"

patterns-established:
  - "asyncio.wait_for() + TimeoutError re-raise pattern for probe generation timeout"

# Metrics
duration: 8min
completed: 2026-02-25
---

# Phase 15 Plan 19: Generation Timeouts (TDD) Summary

**Added `timeout` parameter (default 60s) to `GenerativeProbe._generate()` and `VisionProbe._generate()` using `asyncio.wait_for()` — prevents endless model generation from blocking probe indefinitely.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-25T14:16:34Z
- **Completed:** 2026-02-25T14:24:21Z
- **Tasks:** 2 TDD commits (RED + GREEN)
- **Files modified:** 3 (+ 1 created)

## Accomplishments

- Added `timeout: float = 60.0` parameter to `GenerativeProbe._generate()` in `base.py`
- Wrapped `adapter.generate()` call in `asyncio.wait_for()` — catches `TimeoutError` and re-raises as descriptive `TimeoutError(f"Generation timed out after {timeout}s (max_tokens={max_tokens})")`
- `finally` block still resets `template_options` via `adapter.configure(template_options=None)` even when timeout fires
- Applied the same pattern to `VisionProbe._generate()` in `vision.py` (overrides base method)
- Added `import asyncio` to both files
- 8 new tests in `test_probe_generation_timeout.py` covering all timeout behaviors:
  - Default parameter value (60.0s)
  - Timeout raises `TimeoutError` with descriptive message
  - Custom timeout is respected (fires within ~0.1s for 0.1s timeout)
  - Template options reset even on timeout (finally block)
  - Successful generation still works (regression guard)
  - `VisionProbe` has timeout parameter with same default
  - `VisionProbe` raises `TimeoutError` on hang
  - Error message includes `max_tokens` for diagnostics
- Fixed ruff `UP041`: use builtin `TimeoutError` not `asyncio.TimeoutError` in `except` clauses

## TDD Commits

| Phase | Commit | Description |
|-------|--------|-------------|
| RED | f5b4c87 | test(6a-01): add failing tests for generation timeout in GenerativeProbe |
| GREEN | 2085bb7 | feat(6a-01): add timeout protection to GenerativeProbe._generate() |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] PIL.Image patch path in vision test**

- **Found during:** GREEN phase (test_vision_generate_timeout)
- **Issue:** Test used `patch("mlx_manager.services.probe.vision.Image")` but `Image` is imported locally inside `_generate()`, not at module level — patch target doesn't exist
- **Fix:** Changed to `patch("PIL.Image")` to patch at the PIL module level
- **Files modified:** `backend/tests/test_probe_generation_timeout.py`
- **Commit:** 2085bb7 (included in GREEN commit)

**2. [Rule 1 - Bug] ruff UP041 lint error**

- **Found during:** GREEN phase lint check
- **Issue:** `except asyncio.TimeoutError:` triggers UP041 — ruff prefers builtin `TimeoutError`
- **Fix:** Auto-fixed with `ruff check --fix` + `ruff format`
- **Files modified:** `backend/mlx_manager/services/probe/base.py`, `backend/mlx_manager/services/probe/vision.py`
- **Commit:** 2085bb7 (included in GREEN commit)

## Next Phase Readiness

All quality gates pass:
- ruff: clean on changed files
- mypy: 3 pre-existing errors (unchanged, not introduced here)
- pytest: 8 new tests pass, no regressions in full suite (2757 pass, 8 pre-existing failures)
