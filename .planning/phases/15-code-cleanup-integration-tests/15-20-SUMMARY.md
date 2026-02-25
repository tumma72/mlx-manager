---
phase: 15
plan: 20
name: verbose-flag-utilization
subsystem: probe-pipeline
tags: [probe, verbose, tdd, debugging, timing, diagnostics]
requires: [15-18]
provides: [verbose-probe-details]
affects: []
tech-stack:
  added: []
  patterns: [tdd-red-green-refactor, verbose-flag-threading, time.monotonic-timing]
key-files:
  created:
    - backend/tests/test_probe_verbose.py
  modified:
    - backend/mlx_manager/services/probe/steps.py
    - backend/mlx_manager/services/probe/sweeps.py
    - backend/mlx_manager/services/probe/base.py
    - backend/mlx_manager/services/probe/coordinator.py
    - backend/tests/test_probe_sweep_capabilities.py
decisions:
  - name: verbose-only-in-details-dict
    choice: verbose info only added to ProbeStep.details — never changes status/value/capability
    rationale: minimal surface area; no impact on SSE protocol or downstream consumers
  - name: time-monotonic-for-elapsed
    choice: time.monotonic() for elapsed_ms in StepContext
    rationale: monotonic clock not affected by system time changes; round() gives clean int ms
  - name: info-level-verbose-diagnostics
    choice: verbose adds INFO-level ProbeDiagnostic entries to sweep returns
    rationale: doesn't pollute WARNING/ACTION_NEEDED diagnostic channels; clearly labeled as verbose
metrics:
  duration: "15 min"
  completed: "2026-02-25"
  tests-added: 17
  tests-passing: 2765
---

# Phase 15 Plan 20: Verbose Flag Utilization Summary

**One-liner:** Thread verbose through probe pipeline — StepContext adds elapsed_ms timing, sweep functions add raw output samples and parser trial details in INFO diagnostics.

## What Was Built

The `verbose` flag previously passed to `ProbingCoordinator.probe()` was silently unused throughout the probe pipeline. This plan wires it through from coordinator to all steps.

### Changes Made

**`steps.py` — StepContext + probe_step:**
- Added `verbose: bool = False` parameter to `StepContext.__init__()`
- Added `_verbose` and `_start_time` to `__slots__` (using `time.monotonic()`)
- Updated `result` property: when `verbose=True`, computes `elapsed_ms = round((time.monotonic() - _start_time) * 1000)` and adds to `step.details` dict (creates dict if None)
- Timing is added on both successful and failed steps when verbose
- Updated `probe_step()` async context manager to accept `verbose: bool = False` and pass to `StepContext`

**`sweeps.py` — sweep_thinking + sweep_tools:**
- Added `verbose: bool = False` to `sweep_thinking()` signature
- When verbose=True and generation succeeded: appends INFO `ProbeDiagnostic` with `raw_output_sample`, `parser_trials`, and `discovered_tags` to diagnostics
- Added `verbose: bool = False` to `sweep_tools()` signature
- When verbose=True: appends INFO `ProbeDiagnostic` with `parser_trials`, `raw_output_sample`, and `discovered_tags` to diagnostics

**`base.py` — GenerativeProbe.sweep_capabilities:**
- Added `*, verbose: bool = False` keyword-only parameter
- All three `probe_step()` calls now pass `verbose=verbose`
- Both sweep function calls pass `verbose=verbose`

**`coordinator.py` — ProbingCoordinator.probe:**
- `strategy.sweep_capabilities(...)` call now passes `verbose=verbose`

**`tests/test_probe_sweep_capabilities.py`:**
- Updated `fake_sweep` in `test_coordinator_delegates_to_generative_probe` to accept `*, verbose=False` (the coordinator now passes verbose as keyword arg)

## TDD Cycle

**RED** (commit `d7e07f4`): 17 tests written — 9 failing, 8 passing (the 8 passing were backward-compat tests that correctly passed before implementation).

**GREEN** (commit `8a09014`): All 4 files updated — 17/17 tests pass, full suite 2765 passed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated fake_sweep signature in existing test**

- **Found during:** Running full probe test suite after implementation
- **Issue:** `test_coordinator_delegates_to_generative_probe` defined `fake_sweep(model_id, loaded, result)` without `**kwargs` — coordinator now calls with `verbose=False` keyword arg, causing `TypeError`
- **Fix:** Updated signature to `fake_sweep(model_id, loaded, result, *, verbose=False)`
- **Files modified:** `backend/tests/test_probe_sweep_capabilities.py`
- **Commit:** `8a09014`

## Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Where to put verbose info | `ProbeStep.details` only | No protocol changes; SSE `to_sse()` already handles details dict |
| Timing measurement | `time.monotonic()` | Immune to system clock changes; accurate relative timing |
| `elapsed_ms` type | `int` (via `round()`) | Clean integer milliseconds; sufficient precision for probe steps |
| Verbose in sweep functions | INFO-level ProbeDiagnostic | Separate channel from WARNING/ACTION_NEEDED; clearly labeled |
| Verbose parameter style | keyword-only (`*, verbose=False`) in `sweep_capabilities` | Prevents accidental positional arg passing; clearly opt-in |

## Next Phase Readiness

No blockers. The verbose flag is now fully threaded. When `verbose=True`:
- Every probe step includes `elapsed_ms` in its details
- Thinking sweep includes raw output sample and parser trial sequence in an INFO diagnostic
- Tool sweep includes parser trials and raw output sample in an INFO diagnostic
- All changes are backward-compatible (`verbose=False` produces identical output to pre-implementation)
