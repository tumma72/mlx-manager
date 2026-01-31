---
phase: 13-mlx-server-integration
plan: 01
subsystem: api
tags: [fastapi, mlx-server, embedded-mode, sub-application, model-pool]

# Dependency graph
requires:
  - phase: 07-server-foundation
    provides: MLX Server FastAPI application with model pool
  - phase: 12-production-hardening
    provides: Observability, error handling, audit logging
provides:
  - Embeddable create_app() factory function for MLX Server
  - MLX Server mounted at /v1/* within MLX Manager
  - Model pool initialization in MLX Manager lifespan
  - Embedded mode configuration support
affects: [13-02, 13-03, frontend-chat, settings-ui]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Sub-application mounting with FastAPI app.mount()
    - Lazy module-level app via __getattr__
    - Embedded mode configuration with shared database

key-files:
  created: []
  modified:
    - backend/mlx_manager/main.py
    - backend/mlx_manager/mlx_server/main.py
    - backend/mlx_manager/mlx_server/config.py
    - backend/mlx_manager/mlx_server/api/v1/chat.py
    - backend/mlx_manager/mlx_server/api/v1/completions.py
    - backend/mlx_manager/mlx_server/api/v1/messages.py
    - backend/mlx_manager/mlx_server/api/v1/models.py
    - backend/mlx_manager/models.py

key-decisions:
  - "Mount at /v1 prefix with routers having no prefix to avoid double /v1/v1/*"
  - "Lazy app initialization via __getattr__ to prevent LogFire config on import"
  - "Embedded mode uses MLX Manager database for shared audit logs"
  - "Kept RunningInstance model for backward compatibility with servers router"

patterns-established:
  - "create_app(embedded=True) factory for embedding FastAPI sub-apps"
  - "is_embedded() helper function for runtime mode detection"
  - "get_database_path() method for environment-aware database selection"

# Metrics
duration: 9min
completed: 2026-01-31
---

# Phase 13 Plan 01: Mount MLX Server as Sub-Application Summary

**MLX Server mounted at /v1/* within MLX Manager via create_app(embedded=True) factory with model pool initialization in parent lifespan**

## Performance

- **Duration:** 9 min
- **Started:** 2026-01-31T15:35:17Z
- **Completed:** 2026-01-31T15:44:17Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Created create_app() factory function supporting embedded/standalone modes
- Mounted MLX Server at /v1 prefix in MLX Manager
- Initialized model pool and batching scheduler in MLX Manager lifespan
- Added embedded mode configuration with shared database path

## Task Commits

Each task was committed atomically:

1. **Task 1: Create embeddable app factory** - `ccd1aa2` (feat)
2. **Task 2: Mount MLX Server in main.py** - `6a4cb0d` (feat)
3. **Task 3: Update config for embedded mode** - `d70722b` (feat)

**Fix commit:** `caa0c1a` - Restore RunningInstance model for backward compatibility

## Files Created/Modified

- `backend/mlx_manager/main.py` - Added MLX Server imports, model pool init, and app.mount()
- `backend/mlx_manager/mlx_server/main.py` - Added create_app() factory with embedded mode support
- `backend/mlx_manager/mlx_server/config.py` - Added embedded_mode setting and is_embedded() helper
- `backend/mlx_manager/mlx_server/api/v1/chat.py` - Removed /v1 prefix from router
- `backend/mlx_manager/mlx_server/api/v1/completions.py` - Removed /v1 prefix from router
- `backend/mlx_manager/mlx_server/api/v1/messages.py` - Removed /v1 prefix from router
- `backend/mlx_manager/mlx_server/api/v1/models.py` - Removed /v1 prefix from router
- `backend/mlx_manager/models.py` - Restored RunningInstance model

## Decisions Made

- **Mount at /v1 with no router prefix**: Prevents double prefix `/v1/v1/*` when routers already had `/v1` prefix. Removed prefix from individual routers.
- **Lazy app initialization**: Module-level `app` via `__getattr__` ensures LogFire isn't configured when only importing `create_app`.
- **Embedded database path**: When embedded_mode=True, MLX Server uses MLX Manager's database for audit logs instead of separate file.
- **Restore RunningInstance**: The servers router still uses RunningInstance for subprocess management. Kept model for backward compatibility with TODO for future removal.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Restored RunningInstance model**
- **Found during:** Verification phase
- **Issue:** RunningInstance was removed by previous commit but servers router still imports it
- **Fix:** Restored RunningInstance model with DEPRECATED note
- **Files modified:** backend/mlx_manager/models.py
- **Verification:** App imports and starts successfully
- **Committed in:** caa0c1a

**2. [Rule 3 - Blocking] Removed /v1 prefix from individual routers**
- **Found during:** Task 2 verification
- **Issue:** Routes were at /v1/v1/* due to both mount point and router prefix having /v1
- **Fix:** Removed prefix="/v1" from chat, completions, messages, models routers
- **Files modified:** 4 router files
- **Verification:** /v1/models returns correct response
- **Committed in:** 6a4cb0d (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes necessary for correct operation. No scope creep.

## Issues Encountered

- **Memory limit API change**: `set_memory_limit()` shows warning about incompatible arguments. This is a minor issue from MLX API changes but doesn't block functionality.
- **Deleted files in working tree**: Several files (server_manager.py, parser_options.py, command_builder.py) were deleted by previous commits but still needed. Restored from git history.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- MLX Server routes accessible at /v1/* from MLX Manager
- Model pool initializes on startup
- Ready for Plan 02: Update UI to use embedded server
- **Technical debt**: RunningInstance model and servers router need refactoring for full embedded mode

---
*Phase: 13-mlx-server-integration*
*Completed: 2026-01-31*
