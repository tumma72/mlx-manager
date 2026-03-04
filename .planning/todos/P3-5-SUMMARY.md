---
phase: "adhoc"
plan: "P3-5"
subsystem: "mlx-server-preload"
tags: ["mlx-server", "startup", "preload", "model-pool", "config"]
depends_on: []
provides: ["startup-model-preloading", "preload-config-settings"]
affects: ["mlx-server-lifespan", "model-pool-warmup"]
tech-stack:
  added: []
  patterns: ["try/except non-fatal preload", "lifespan integration for preload"]
key-files:
  created: []
  modified:
    - backend/mlx_manager/mlx_server/config.py
    - backend/mlx_manager/mlx_server/main.py
    - backend/tests/mlx_server/test_mlx_server_main.py
decisions:
  - "Preload in lifespan after pool init, before signal handlers"
  - "try/except wraps apply_preload_list so preload failures never block server startup"
  - "Empty preload_models list is a no-op — fully backward compatible"
  - "warmup_prompt field present for future use but inference call not added (keeps main.py clean)"
metrics:
  duration: "~5 min"
  completed: "2026-03-04"
---

# Phase Adhoc Plan P3-5: Model Preload Warming Summary

**One-liner:** Startup model preloading via `preload_models` config list calling `apply_preload_list()` in lifespan with non-fatal error handling.

## What Was Built

Added two config fields to `MLXServerSettings` and integrated model preloading into the FastAPI lifespan handler.

### Config Fields Added (`config.py`)

```python
preload_models: list[str] = Field(
    default_factory=list,
    description="Model IDs to preload at server startup for reduced cold-start latency",
)
warmup_prompt: str = Field(
    default="Hello",
    description="Prompt to run after preloading to warm up the model",
)
```

Both use the `MLX_SERVER_` env prefix (`MLX_SERVER_PRELOAD_MODELS`, `MLX_SERVER_WARMUP_PROMPT`).

### Lifespan Integration (`main.py`)

After pool initialization, if `preload_models` is non-empty:

1. Calls `pool.model_pool.apply_preload_list(preload_models)` — loads each model and marks it as protected from LRU eviction.
2. Logs progress at INFO level (start, count ready, count failed).
3. Logs partial failures at WARNING level per model.
4. Catches any unexpected exception from the preload step and logs a WARNING — server startup continues regardless.

### Tests Added (`test_mlx_server_main.py`)

13 new tests across two new test classes:

**`TestPreloadConfig` (4 tests):**
- `test_preload_models_default_is_empty_list` — backward compat: empty default
- `test_preload_models_parses_list` — list of model IDs parsed correctly
- `test_warmup_prompt_default` — default is "Hello"
- `test_warmup_prompt_configurable` — can be overridden

**`TestLifespanPreload` (4 tests):**
- `test_empty_preload_list_is_noop` — `apply_preload_list` never called when list is empty
- `test_preload_list_calls_apply_preload` — called with correct model list
- `test_preload_failure_does_not_crash_startup` — RuntimeError is swallowed, server starts
- `test_partial_preload_failure_logs_warning` — partial `failed:` results handled gracefully

Also updated 3 existing lifespan tests to set `mock_settings.preload_models = []` so they continue to pass with the new conditional in lifespan.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| No warmup inference call | Would require importing inference service in main.py — keeps lifespan lightweight; loading weights is the main latency win |
| Empty list = no-op | Full backward compat, zero code change for existing deployments |
| Warning (not error) on failure | Preload is best-effort; a bad model ID should not prevent the server from starting |
| Preload after pool init, before signal handlers | Natural ordering: pool must exist before preloading into it |

## Verification

```
# Config test
cd backend && python -m pytest tests/mlx_server/test_mlx_server_main.py -x -q
# 21 passed

# Full MLX server suite
cd backend && python -m pytest tests/mlx_server/ -x -q
# 1953 passed
```

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written.

**Minor additions for correctness:**
- Updated 3 existing lifespan tests to add `mock_settings.preload_models = []` — these tests were patching `mlx_server_settings` but didn't set the new field, which would cause `AttributeError` on the new `if mlx_server_settings.preload_models:` check in the lifespan. Not a deviation from the plan; normal test maintenance when adding a new settings field.

## Commit

- `5dec5b3`: feat(P3-5): startup model preload warming for MLX Server
