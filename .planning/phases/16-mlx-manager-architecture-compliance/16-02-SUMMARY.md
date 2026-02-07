---
phase: 16-mlx-manager-architecture-compliance
plan: 02
subsystem: ui
tags: [jwt, sse, websocket, eventsource, svelte, auth, frontend]

# Dependency graph
requires:
  - phase: 16-01
    provides: Backend SSE/WS query-param JWT auth, deprecated endpoint removal
provides:
  - JWT token injection in EventSource URL for download progress SSE
  - JWT token injection in WebSocket URL for audit log streaming
  - Removal of deprecated parser API functions and types from frontend
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Query-param JWT for browser SSE/WS: ?token=<jwt> since EventSource cannot send custom headers"

key-files:
  created: []
  modified:
    - frontend/src/lib/stores/downloads.svelte.ts
    - frontend/src/lib/api/client.ts
    - frontend/src/lib/api/types.ts
    - frontend/src/lib/stores/downloads.svelte.test.ts
    - frontend/src/lib/api/client.test.ts
    - frontend/e2e/app.spec.ts

key-decisions:
  - "Conditional token inclusion: token ? `?token=${token}` : '' for defensive handling when token is null"
  - "Added unauthenticated WebSocket test to verify graceful behavior without token"

patterns-established:
  - "Token injection pattern: authStore.token appended as query param for SSE/WS connections"

# Metrics
duration: 3min
completed: 2026-02-07
---

# Phase 16 Plan 02: Frontend SSE/WS Auth & Parser Cleanup Summary

**JWT token injection in EventSource and WebSocket URLs via authStore, deprecated parser API functions and ParserOptions type removed**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-07T12:41:22Z
- **Completed:** 2026-02-07T12:44:53Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- EventSource URL for download progress now includes ?token=<jwt> from authStore for SSE authentication
- WebSocket URL for audit log streaming now includes ?token=<jwt> from authStore for WS authentication
- Removed getAvailableParsers() and parserOptions() deprecated API functions from client.ts
- Removed ParserOptions interface from types.ts
- Updated all test expectations for token-injected URLs
- All 984 frontend unit tests pass, type checking and linting clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Add token to SSE and WebSocket URLs in frontend** - `c159549` (feat)
2. **Task 2: Update frontend tests and types, remove deprecated test code** - `3d7d298` (test)

## Files Created/Modified
- `frontend/src/lib/stores/downloads.svelte.ts` - Added authStore import and token injection in connectSSE()
- `frontend/src/lib/api/client.ts` - Added token to WebSocket URL, removed getAvailableParsers(), removed parserOptions(), removed ParserOptions import
- `frontend/src/lib/api/types.ts` - Removed ParserOptions interface
- `frontend/src/lib/stores/downloads.svelte.test.ts` - Added authStore mock, updated 4 EventSource URL expectations with token
- `frontend/src/lib/api/client.test.ts` - Updated 2 WebSocket URL expectations with token, added unauthenticated test, removed getAvailableParsers and parserOptions tests
- `frontend/e2e/app.spec.ts` - Removed parser-options API mock

## Decisions Made
- Used conditional token inclusion (`token ? \`?token=${token}\` : ""`) for defensive handling when authStore.token is null
- Added an explicit test for WebSocket creation without authentication to verify the no-token path works correctly

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 16 is now complete (both plans 16-01 and 16-02 done)
- Full ARCH-01 compliance: SSE and WebSocket connections authenticated via query-param JWT on both backend and frontend
- Full ARCH-03 compliance: Deprecated parser-options and available-parsers endpoints removed from backend (16-01) and frontend (16-02)
- No blockers or concerns

---
*Phase: 16-mlx-manager-architecture-compliance*
*Completed: 2026-02-07*
