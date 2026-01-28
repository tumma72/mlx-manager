---
phase: 08-multi-model-multimodal
plan: 01
subsystem: api
tags: [mlx, memory-management, lru-eviction, psutil, model-pool]

# Dependency graph
requires:
  - phase: 07-foundation
    provides: ModelPoolManager with single-model support
provides:
  - Multi-model hot-swapping with LRU eviction
  - Preload protection for pinned models
  - Configurable memory limits (GB or percentage)
  - Model size estimation from name patterns
  - 503 error handling for insufficient memory
affects: [08-02, 08-03, 08-04, 08-05, 08-06]

# Tech tracking
tech-stack:
  added: [psutil]
  patterns: [LRU eviction, preload protection, memory pressure detection]

key-files:
  created:
    - backend/mlx_manager/mlx_server/models/types.py
    - backend/tests/mlx_server/test_pool.py
  modified:
    - backend/mlx_manager/mlx_server/models/pool.py

key-decisions:
  - "Model size estimation uses parameter count patterns (3B=2GB, 7B=4GB, etc.)"
  - "Default max_models increased from 1 to 4 for multi-model hot-swapping"
  - "Memory limit can be absolute GB or percentage of system memory via psutil"
  - "Preloaded models are never evicted regardless of last_used time"

patterns-established:
  - "LRU eviction: min(evictable, key=lambda m: m.last_used)"
  - "Memory check before load with eviction loop"
  - "503 HTTPException for irrecoverable memory pressure"

# Metrics
duration: 3min
completed: 2026-01-28
---

# Phase 08 Plan 01: ModelPoolManager LRU Eviction Summary

**Multi-model hot-swapping with LRU eviction, preload protection, and configurable memory limits via psutil**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-28T11:39:47Z
- **Completed:** 2026-01-28T11:42:34Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- ModelType enum (TEXT_GEN, VISION, EMBEDDINGS) for future multimodal support
- Extended LoadedModel with model_type and preloaded fields
- LRU eviction removes least-recently-used non-preloaded models when memory pressure detected
- Memory limits configurable as absolute GB or percentage of system memory
- Preloaded models protected from eviction (survive even when oldest)
- 503 HTTPException raised when insufficient memory after all evictions
- 24 comprehensive unit tests covering all new functionality

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ModelType enum** - `ddfc274` (feat)
2. **Task 2: Enhance ModelPoolManager with LRU eviction** - `aec22b2` (feat)
3. **Task 3: Add unit tests for pool** - `9f1514c` (test)

## Files Created/Modified

- `backend/mlx_manager/mlx_server/models/types.py` - ModelType enum for model classification
- `backend/mlx_manager/mlx_server/models/pool.py` - Enhanced ModelPoolManager with LRU eviction
- `backend/tests/mlx_server/test_pool.py` - 24 unit tests for new functionality

## Decisions Made

- **Model size estimation heuristic:** Uses regex pattern matching on model_id (e.g., "3B" -> 2.0GB, "7B" -> 4.0GB) with interpolation for other sizes
- **Default max_models:** Changed from 1 to 4 to support multi-model scenarios out of the box
- **Memory percentage via psutil:** When memory_limit_pct is set, calculates from psutil.virtual_memory().total

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Pool ready for multi-model support (up to 4 hot models by default)
- Vision and embeddings model types defined but not yet used
- Ready for 08-02: Image preprocessing and VLM adapter integration

---
*Phase: 08-multi-model-multimodal*
*Completed: 2026-01-28*
