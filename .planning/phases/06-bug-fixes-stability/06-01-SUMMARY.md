---
phase: 06-bug-fixes-stability
plan: 01
subsystem: error-handling
tags: [logging, exception-handling, http-status-codes, fastapi, python]

# Dependency graph
requires:
  - phase: 05-chat-multimodal-support
    provides: Complete backend API surface area
provides:
  - Comprehensive exception logging across all services
  - Proper HTTP error responses in API endpoints
  - Improved debuggability for production failures
affects: [all-phases, debugging, monitoring, production-ops]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "All exception handlers log failures at appropriate level (debug/warning/error)"
    - "API validation uses HTTPException, not assertions"
    - "Service layer uses ValueError/RuntimeError, caught by routers"

key-files:
  created: []
  modified:
    - backend/mlx_manager/services/hf_client.py
    - backend/mlx_manager/services/health_checker.py
    - backend/mlx_manager/services/launchd.py
    - backend/mlx_manager/services/auth_service.py
    - backend/mlx_manager/database.py
    - backend/mlx_manager/routers/system.py
    - backend/mlx_manager/routers/servers.py
    - backend/mlx_manager/menubar.py
    - backend/mlx_manager/services/server_manager.py

key-decisions:
  - "Use debug level for non-critical failures (cache checks, fallbacks, optional deps)"
  - "Use warning level for health check failures"
  - "Use error level for database transaction failures"
  - "Replace assertions with HTTPException(400) for state validation in routers"

patterns-established:
  - "Exception handlers: All except blocks capture exception as 'e' and log with context"
  - "Validation pattern: Router validates → raises HTTPException; Service layer validates → raises ValueError/RuntimeError"
  - "Intentional silent handlers: Only asyncio.CancelledError during shutdown"

# Metrics
duration: 4min
completed: 2026-01-24
---

# Phase 6 Plan 1: Error Handling & Logging Improvements Summary

**All silent exception handlers now log failures; API endpoints return proper 400 Bad Request on validation errors instead of 500 Internal Server Error**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-24T10:52:38Z
- **Completed:** 2026-01-24T10:57:03Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Eliminated invisible failures by adding logging to 10 silent exception handlers
- Fixed API error handling to return proper HTTP 400 status codes on validation failures
- Improved production debuggability across services, routers, and database layer
- All 532 existing tests continue to pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Add logging to silent exception handlers** - `ffbce4a` (refactor)
2. **Task 2: Replace assertions with HTTPException in routers** - `1441ebf` (fix)

## Files Created/Modified
- `backend/mlx_manager/services/hf_client.py` - Added logger, logged 5 exception handlers (cache checks, size estimation)
- `backend/mlx_manager/services/health_checker.py` - Added logger, converted print() to logger.warning()
- `backend/mlx_manager/services/launchd.py` - Added logger, logged launchctl unload failure
- `backend/mlx_manager/services/auth_service.py` - Added logger, logged JWT decode failures
- `backend/mlx_manager/database.py` - Logged session rollback errors
- `backend/mlx_manager/routers/system.py` - Logged optional import failures (mlx, mlx-openai-server)
- `backend/mlx_manager/routers/servers.py` - Replaced 4 assertions with HTTPException(400)
- `backend/mlx_manager/menubar.py` - Added logger, logged server health check failures
- `backend/mlx_manager/services/server_manager.py` - Replaced assertion with explicit ValueError

## Decisions Made

1. **Log level hierarchy:**
   - `debug`: Non-critical failures (cache checks, size estimation fallbacks, optional dependencies)
   - `warning`: Operational issues that don't stop the service (health check errors)
   - `error`: Critical failures requiring investigation (database rollbacks)

2. **Validation error handling:**
   - Routers use HTTPException(400) for client validation errors
   - Service layer uses ValueError/RuntimeError, caught by router Exception handlers
   - Never use assertions for runtime validation (assertions are for invariants)

3. **Intentional silent handlers:**
   - Only `asyncio.CancelledError` remains silent (expected during service shutdown)
   - All other exceptions must be logged with context

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all changes were straightforward logging and validation improvements.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Error handling foundation strengthened for all future phases
- Production debugging significantly improved
- Health monitoring now reports failures instead of silently swallowing them
- API clients receive proper HTTP status codes for invalid requests

---
*Phase: 06-bug-fixes-stability*
*Completed: 2026-01-24*
