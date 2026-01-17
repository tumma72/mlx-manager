---
phase: 02-server-panel-redesign
plan: 01
subsystem: api
tags: [fastapi, psutil, typescript, server-metrics]

# Dependency graph
requires:
  - phase: 01-models-panel-redesign
    provides: Backend API infrastructure and patterns
provides:
  - Extended /api/servers response with cpu_percent and memory_percent
  - RunningServer TypeScript type with extended metrics
affects: [02-02, 02-03, 02-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - psutil.memory_percent() for system memory percentage
    - TypedDict field extension across API layers

key-files:
  created: []
  modified:
    - backend/mlx_manager/types.py
    - backend/mlx_manager/models.py
    - backend/mlx_manager/services/server_manager.py
    - backend/mlx_manager/routers/servers.py
    - frontend/src/lib/api/types.ts

key-decisions:
  - "Use psutil.memory_percent() which returns percentage of total system RAM"
  - "Add fields with default values (0.0) to maintain backward compatibility"

patterns-established:
  - "Extend TypedDicts in types.py before models.py response models"
  - "Propagate API changes through service -> router -> frontend types"

# Metrics
duration: 8min
completed: 2026-01-17
---

# Phase 2 Plan 1: Extended Server Metrics API Summary

**Extended /api/servers response with cpu_percent and memory_percent fields using psutil for dashboard gauges**

## Performance

- **Duration:** 8 min
- **Started:** 2026-01-17T19:45:00Z
- **Completed:** 2026-01-17T19:53:00Z
- **Tasks:** 4
- **Files modified:** 5

## Accomplishments

- Extended backend TypedDicts (ServerStats, RunningServerInfo) with memory_percent field
- Extended RunningServerResponse model with cpu_percent and memory_percent (defaults to 0.0)
- Updated server_manager to calculate memory_percent using psutil.Process.memory_percent()
- Updated servers router to include new fields in API response
- Extended frontend RunningServer TypeScript interface to match API

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend backend types and models** - `99f85b9` (feat)
2. **Task 2: Update server_manager to calculate extended stats** - `eb7b97b` (feat)
3. **Task 3: Update servers router to include new fields in response** - `da8c500` (feat)
4. **Task 4: Update frontend TypeScript types** - `db2b25e` (feat)

## Files Created/Modified

- `backend/mlx_manager/types.py` - Added memory_percent to ServerStats and RunningServerInfo TypedDicts
- `backend/mlx_manager/models.py` - Added cpu_percent and memory_percent to RunningServerResponse
- `backend/mlx_manager/services/server_manager.py` - Added memory_percent calculation via psutil
- `backend/mlx_manager/routers/servers.py` - Included new fields in list_running_servers response
- `frontend/src/lib/api/types.ts` - Extended RunningServer interface with new fields

## Decisions Made

- Used psutil.memory_percent() which returns percentage of total system RAM (not per-process)
- Added fields with default values (0.0) to RunningServerResponse for backward compatibility with existing API consumers

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed without issues.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- API now returns cpu_percent, memory_percent, and uptime_seconds for running servers
- Ready for Plan 02 (ServerTile component redesign) to consume these metrics for dashboard display
- All 22 server tests passing
- Frontend type checking passing

---
*Phase: 02-server-panel-redesign*
*Completed: 2026-01-17*
