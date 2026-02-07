---
phase: 16-mlx-manager-architecture-compliance
plan: 01
subsystem: auth, api
tags: [jwt, websocket, sse, query-param-auth, deprecated-endpoint-removal, router-decoupling]

# Dependency graph
requires:
  - phase: 15-code-cleanup-integration-tests
    provides: Clean codebase with auth infrastructure and embedded MLX server
provides:
  - get_current_user_from_token() dependency for SSE/WS query-param auth
  - JWT secret startup warning in lifespan
  - get_loaded_model() public method on ModelPoolManager
  - Removed deprecated parser endpoints (available-parsers, parser-options)
  - WebSocket audit-logs JWT validation before accept
affects: [16-02-frontend-settings-ui]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Query-param JWT auth for SSE/WebSocket endpoints (browser EventSource limitation)"
    - "WebSocket auth validation before accept() with code 1008 on failure"
    - "Public pool API (get_loaded_model) replaces private _models access"

key-files:
  created: []
  modified:
    - backend/mlx_manager/dependencies.py
    - backend/mlx_manager/main.py
    - backend/mlx_manager/mlx_server/models/pool.py
    - backend/mlx_manager/routers/models.py
    - backend/mlx_manager/routers/system.py
    - backend/mlx_manager/routers/servers.py
    - backend/tests/test_models.py
    - backend/tests/test_system.py
    - backend/tests/test_servers.py

key-decisions:
  - "Query-param JWT for SSE: browser EventSource cannot send custom headers, so token passed as ?token=<jwt>"
  - "WebSocket auth before accept: validate JWT and user status before websocket.accept() to prevent unauthenticated connections"
  - "Code 1008 for WS auth failures: policy violation code per RFC 6455"
  - "Direct mock WebSocket tests: replaced SyncTestClient-based WebSocket tests with direct function calls to avoid lifespan/database initialization issues"

patterns-established:
  - "Query-param auth dependency: get_current_user_from_token() for endpoints where browser cannot send headers"
  - "WebSocket pre-accept validation: validate auth before accepting, close(1008) on failure"
  - "Public pool API: use get_loaded_model() instead of accessing _models directly from routers"

# Metrics
duration: 12min
completed: 2026-02-07
---

# Phase 16 Plan 01: Backend Auth & Housekeeping Summary

**SSE/WebSocket query-param JWT auth, deprecated endpoint removal, pool API encapsulation, and JWT startup warning**

## Performance

- **Duration:** 12 min
- **Started:** 2026-02-07T12:26:29Z
- **Completed:** 2026-02-07T12:38:16Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments
- Added query-param JWT auth for SSE download progress endpoint (browser EventSource cannot send custom headers)
- Added WebSocket JWT validation before accept() for audit-logs endpoint with proper 1008 close on failure
- Removed 2 deprecated parser endpoints (available-parsers, parser-options)
- Added public get_loaded_model() method on ModelPoolManager, eliminating all pool._models access from routers
- Added JWT secret placeholder startup warning
- All 1979 backend tests pass with new WebSocket auth tests (no token, invalid token, unapproved user, pending user)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add query-param auth dependency, JWT warning, and pool public method** - `07e8d3f` (feat)
2. **Task 2: Update SSE/WS endpoints, remove deprecated endpoints, fix router decoupling** - `3c6db30` (feat)
3. **Task 3: Update backend tests for auth changes and removed endpoints** - `6dea2ac` (test)

## Files Created/Modified
- `backend/mlx_manager/dependencies.py` - Added get_current_user_from_token() for query-param JWT auth
- `backend/mlx_manager/main.py` - Added JWT secret placeholder startup warning in lifespan
- `backend/mlx_manager/mlx_server/models/pool.py` - Added get_loaded_model() public method
- `backend/mlx_manager/routers/models.py` - SSE endpoint uses query-param auth; removed /available-parsers
- `backend/mlx_manager/routers/system.py` - WebSocket validates JWT before accept(); removed /parser-options
- `backend/mlx_manager/routers/servers.py` - All pool._models access replaced with get_loaded_model()
- `backend/tests/test_models.py` - Updated SSE tests for query-param auth; deprecated endpoint test updated
- `backend/tests/test_system.py` - Added 4 WebSocket auth tests; rewrote WS tests to use direct mocks
- `backend/tests/test_servers.py` - Updated mock_pool._models to mock_pool.get_loaded_model()

## Decisions Made
- **Query-param JWT for SSE**: Browser EventSource API cannot send custom Authorization headers, so token is passed as `?token=<jwt>` query parameter. This is the standard approach for SSE auth.
- **WebSocket auth before accept**: Validate JWT and user status BEFORE calling `websocket.accept()` to prevent any data exchange with unauthenticated clients. Close with code 1008 (Policy Violation) per RFC 6455.
- **Direct mock WebSocket tests**: Replaced SyncTestClient-based WebSocket tests with direct function calls to `proxy_audit_log_stream()` with mock WebSocket objects. This avoids the lifespan initialization issue where SyncTestClient creates a new database that doesn't have the test user.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_servers.py mock patterns for pool API change**
- **Found during:** Task 3 (Test updates)
- **Issue:** Tests in test_servers.py mocked `pool._models` directly, which no longer works since servers.py now uses `pool.get_loaded_model()`
- **Fix:** Updated 5 test mocks to use `mock_pool.get_loaded_model.return_value` instead of `mock_pool._models`; removed unused `model_path` variables
- **Files modified:** backend/tests/test_servers.py
- **Verification:** All 1979 tests pass
- **Committed in:** 6dea2ac (Task 3 commit)

**2. [Rule 1 - Bug] Rewrote WebSocket tests to avoid lifespan DB issue**
- **Found during:** Task 3 (Test updates)
- **Issue:** SyncTestClient-based WebSocket tests trigger app lifespan which initializes a separate in-memory database without the test user, causing connection failures. This was a pre-existing issue that became more visible with auth validation.
- **Fix:** Rewrote all WebSocket tests to use direct async function calls with mock WebSocket objects and mocked `get_session`, avoiding SyncTestClient entirely.
- **Files modified:** backend/tests/test_system.py
- **Verification:** All WebSocket auth tests pass
- **Committed in:** 6dea2ac (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for test correctness. No scope creep.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Backend auth and housekeeping changes complete
- Ready for plan 16-02 (frontend settings UI and remaining compliance items)
- All success criteria met: SSE auth, WS auth, JWT warning, deprecated endpoints removed, pool API encapsulated

---
*Phase: 16-mlx-manager-architecture-compliance*
*Completed: 2026-02-07*
