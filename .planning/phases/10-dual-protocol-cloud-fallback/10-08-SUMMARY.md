---
phase: 10
plan: 08
subsystem: cloud-routing
tags: [router, failover, pattern-matching, cloud-backend]
dependency-graph:
  requires: ["10-02", "10-06", "10-07"]
  provides: ["BackendRouter", "get_router", "reset_router"]
  affects: ["10-09"]
tech-stack:
  added: []
  patterns: ["singleton", "fnmatch-pattern-matching", "priority-ordered-lookup"]
key-files:
  created:
    - backend/mlx_manager/mlx_server/services/cloud/router.py
    - backend/tests/mlx_server/services/cloud/test_router.py
  modified:
    - backend/mlx_manager/mlx_server/services/cloud/__init__.py
decisions:
  - id: "fnmatch-pattern-matching"
    choice: "Use fnmatch for model pattern matching"
    why: "Supports both exact match and shell-style wildcards (*, ?) - simple and familiar syntax"
  - id: "priority-desc-ordering"
    choice: "Higher priority number = checked first"
    why: "Matches common priority semantics where higher values are more important"
  - id: "sqlalchemy-asyncsession"
    choice: "Use sqlalchemy.ext.asyncio.AsyncSession"
    why: "Matches rest of codebase (routers use SQLAlchemy AsyncSession)"
metrics:
  duration: "5 min"
  completed: "2026-01-29"
---

# Phase 10 Plan 08: Backend Router with Failover Summary

BackendRouter with fnmatch pattern matching and automatic failover to cloud on local failure.

## What Was Built

### BackendRouter Service (router.py - 274 lines)

Core routing service that:
- Looks up model-to-backend mappings from database
- Uses fnmatch for pattern matching (exact + wildcards)
- Checks mappings in priority order (higher priority first)
- Routes to local MLX inference or cloud backends
- Implements automatic failover on local failure

Key features:
- **Pattern matching**: `gpt-4` (exact), `gpt-*` (wildcard), `claude-3-?-*` (fnmatch)
- **Priority ordering**: Higher priority mappings checked first
- **Cloud backend caching**: Backends created once, reused for all requests
- **Fallback routing**: Local failure triggers fallback to configured cloud backend
- **Singleton pattern**: `get_router()` returns same instance, `reset_router()` clears for testing

### Test Suite (test_router.py - 575 lines)

Comprehensive tests covering:
- Pattern matching (6 tests): exact, wildcard, middle pattern, question mark
- Mapping lookup (5 tests): empty, exact, priority order, enabled filter
- Route request (4 tests): no mapping, LOCAL, OPENAI, ANTHROPIC backends
- Failover (3 tests): with/without fallback, backend_model override
- Cloud backend creation (6 tests): create, cache, custom URL, no credentials
- Singleton (4 tests): get, reset, close backends
- Close (3 tests): close all, clear cache, empty cache

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 67f2d15 | feat | Create BackendRouter with pattern matching and failover |
| 251a741 | chore | Add BackendRouter exports to cloud package |
| f39fb7b | test | Add comprehensive BackendRouter tests (31 tests) |
| bb7495c | fix | Add type annotations for mypy compatibility |

## Key Implementation Details

### Pattern Matching Logic

```python
def _pattern_matches(self, pattern: str, model: str) -> bool:
    # Exact match first
    if pattern == model:
        return True
    # Wildcard pattern via fnmatch
    return fnmatch.fnmatch(model, pattern)
```

### Failover Flow

1. Find mapping for model (priority order)
2. If no mapping -> route to local
3. If LOCAL backend:
   - Try local inference
   - On failure, if `fallback_backend` configured -> route to fallback
   - On failure, if no fallback -> raise exception
4. If cloud backend -> route directly to cloud

### Cloud Backend Caching

```python
async def _get_cloud_backend(self, db, backend_type):
    if backend_type in self._cloud_backends:
        return self._cloud_backends[backend_type]  # Cached

    # Load credentials from database
    credential = await db.execute(...)

    # Create backend (OpenAI or Anthropic)
    backend = OpenAICloudBackend(...) if backend_type == BackendType.OPENAI else ...

    # Cache for reuse
    self._cloud_backends[backend_type] = backend
    return backend
```

## Verification

```
$ pytest tests/mlx_server/services/cloud/test_router.py -v
======================== 31 passed in 0.15s ========================

$ python -c "from mlx_manager.mlx_server.services.cloud import BackendRouter, get_router; print('OK')"
OK

$ mypy mlx_manager/mlx_server/services/cloud/router.py
Success: no issues found in 1 source file
(Note: pool.py error is pre-existing, unrelated to router)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Type annotation fixes for mypy compatibility**

- **Found during:** Final verification
- **Issue:** mypy errors from SQLModel/SQLAlchemy type incompatibility
- **Fix:** Changed to sqlalchemy AsyncSession, added type: ignore comments
- **Files modified:** router.py, test_router.py
- **Commit:** bb7495c

## Dependencies Satisfied

- Uses BackendMapping from 10-02 (models)
- Uses OpenAICloudBackend from 10-06 (OpenAI cloud backend)
- Uses AnthropicCloudBackend from 10-07 (Anthropic cloud backend)

## Next Phase Readiness

Ready for 10-09 (Chat endpoint integration):
- BackendRouter exported from cloud package
- Singleton pattern ready for use in chat router
- All tests passing (31 tests)
- Type checking passes
