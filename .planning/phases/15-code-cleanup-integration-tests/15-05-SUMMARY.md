---
phase: 15-code-cleanup-integration-tests
plan: 05
subsystem: api, ui
tags: [memory-metrics, model-unload, servers, fastapi, svelte]

# Dependency graph
requires:
  - phase: 13-mlx-server-integration
    provides: Embedded MLX server with model pool
provides:
  - Per-model memory metrics in server tiles
  - Memory limit percentage gauge for configured limits
  - Working stop button that unloads models from pool
affects: [settings, model-pool, ui-monitoring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Per-model memory tracking via LoadedModel.size_gb
    - Model unload via pool.unload_model() with preload protection

key-files:
  created: []
  modified:
    - backend/mlx_manager/routers/servers.py
    - backend/tests/test_servers.py
    - frontend/src/lib/api/types.ts
    - frontend/src/lib/components/servers/ServerTile.svelte
    - frontend/src/lib/components/servers/ServerTile.test.ts

key-decisions:
  - "Use LoadedModel.size_gb for per-model memory instead of dividing total"
  - "Replace CPU gauge with memory limit gauge (limit % more useful than CPU)"
  - "Protect preloaded models from unload"
  - "Remove unused get_memory_usage import after refactor"

patterns-established:
  - "Per-model metrics pattern: fetch individual model data from pool"
  - "Preload protection: check preloaded flag before unload operations"

# Metrics
duration: 5min
completed: 2026-02-03
---

# Phase 15 Plan 05: Fix Memory Metrics & Stop Button Summary

**Fixed per-model memory display using LoadedModel.size_gb, added memory limit gauge, and implemented actual model unload on stop button**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-03T12:42:18Z
- **Completed:** 2026-02-03T12:47:01Z
- **Tasks:** 6
- **Files modified:** 10

## Accomplishments

- Fixed memory_mb and memory_percent to use per-model size instead of divided total
- Added memory_limit_percent field to track model memory vs configured pool limit
- Replaced non-functional stop endpoint with actual model unload implementation
- Updated frontend ServerTile to show system memory and limit gauges (replaced CPU gauge)

## Task Commits

Each task was committed atomically:

1. **Tasks 1-2: Fix per-model memory display** - `83f2897` (fix)
   - Fix memory_mb to use loaded_model.size_gb * 1024
   - Fix memory_percent to use model size / system total
   - Add memory_limit_percent for model size / configured limit

2. **Tasks 3-4: Implement actual model unload on stop** - `6fb26ef` (feat)
   - Replace no-op stop endpoint with actual unload
   - Use existing pool.unload_model() method
   - Protect preloaded models from unload
   - Add comprehensive stop endpoint tests

3. **Tasks 5-6: Update frontend for memory limit gauge** - `8b96981` (feat)
   - Add memory_limit_percent to RunningServer type
   - Replace CPU gauge with memory limit gauge
   - Rename Memory gauge to System for clarity
   - Update all test mock servers

## Files Created/Modified

- `backend/mlx_manager/routers/servers.py` - Fixed memory calculations, implemented model unload
- `backend/tests/test_servers.py` - Added stop endpoint tests, updated mock data
- `frontend/src/lib/api/types.ts` - Added memory_limit_percent to RunningServer
- `frontend/src/lib/components/servers/ServerTile.svelte` - System and Limit gauges
- `frontend/src/lib/components/servers/ServerTile.test.ts` - Updated gauge tests
- `frontend/src/lib/stores/servers.svelte.test.ts` - Added mock field
- `frontend/src/lib/components/profiles/ProfileCard.test.ts` - Added mock field
- `frontend/src/lib/components/servers/ServerCard.test.ts` - Added mock field

## Decisions Made

- **Use LoadedModel.size_gb instead of dividing total memory** - Each model tracks its own size; using this provides accurate per-model metrics
- **Replace CPU gauge with memory limit gauge** - CPU isn't tracked in embedded mode; memory limit % is more useful for pool management
- **Protect preloaded models from unload** - Preloaded models should remain loaded; return informative error message

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Removed unused get_memory_usage import**
- **Found during:** Task 1 (memory metric fix)
- **Issue:** After refactoring to use LoadedModel.size_gb, the module-level get_memory_usage import became unused
- **Fix:** Removed the import to satisfy ruff linter
- **Files modified:** backend/mlx_manager/routers/servers.py
- **Verification:** ruff check passes
- **Committed in:** 83f2897 (part of task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Import cleanup necessary for linting. No scope creep.

## Issues Encountered

None - plan executed as written.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Memory metrics now accurate per-model
- Stop button functional for model unload
- Memory limit gauge helps users understand pool utilization
- Ready for Gap 5 (Gemma vision model crash) and Gap 6 (download hanging) fixes

---
*Phase: 15-code-cleanup-integration-tests*
*Completed: 2026-02-03*
