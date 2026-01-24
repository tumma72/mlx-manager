---
phase: 06-bug-fixes-stability
plan: 14
subsystem: frontend-ui
tags: [startup-polling, health-check, error-elimination]
requires: [06-13]
provides: [console-error-free-startup, backend-health-polling]
affects: []
tech-stack:
  added: []
  patterns: [backend-mediated-health-checks]
key-files:
  created: []
  modified:
    - frontend/src/lib/components/servers/StartingTile.svelte
decisions:
  - what: Use backend health API instead of direct fetch
    why: Browser fetch() to not-yet-ready server logs native errors before JS can catch them
    impact: Eliminates console noise during server startup
metrics:
  duration: 88s
  completed: 2026-01-24
---

# Phase 06 Plan 14: Health Check Console Errors Summary

**One-liner:** Backend-mediated health polling eliminates browser console errors during server startup.

## What Was Built

Replaced direct browser fetch() calls to the mlx-server `/v1/models` endpoint with backend-mediated health checks via the existing `GET /api/servers/{id}/health` API.

**The Problem:**
Browser fetch() to a not-yet-ready server logs network connection errors at the native browser level (before JavaScript catch blocks can suppress them). This created console noise during normal startup flow.

**The Solution:**
Backend already has a health endpoint that checks the server internally via httpx. The backend handles connection errors gracefully and returns structured responses without browser console errors.

## Implementation Details

### Changes to StartingTile.svelte

1. **Added servers import:**
   - Added `servers` to imports from `$api` (alongside existing `serversApi`)
   - Pattern: `import { servers as serversApi, servers } from '$api';`

2. **Removed unused state and constants:**
   - Removed `healthCheckReady` state variable (no longer needed)
   - Removed `INITIAL_HEALTH_DELAY_MS` constant (no delay required)
   - Backend httpx errors don't show in browser console, so no need to delay

3. **Replaced direct fetch with health API:**
   ```typescript
   // OLD: Direct fetch (caused console errors)
   const response = await fetch(`http://${profile.host}:${profile.port}/v1/models`);

   // NEW: Backend health API (no console errors)
   const healthStatus = await servers.health(profile.id);
   if (healthStatus.status === 'healthy' && healthStatus.model_loaded) {
       // Model ready
   }
   ```

4. **Model detection logic:**
   - Old: Checked `data.data.length > 0` from /v1/models response
   - New: Checks `healthStatus.model_loaded === true` from health API
   - Functionally equivalent, cleaner implementation

### Flow After Changes

1. User starts server → PID confirmed via `servers.status()`
2. Poll begins immediately with `servers.health()` (no delay)
3. Backend checks server health via httpx internally
4. Backend returns structured response: `{ status, model_loaded, response_time_ms, error }`
5. If `model_loaded === true`, tile transitions to running state
6. If timeout (2 minutes), error tile displayed
7. No browser console errors at any step

## Verification

### Type Safety
- ✅ `npm run check` passes (0 errors, 0 warnings)
- ✅ `servers.health()` correctly typed as `Promise<HealthStatus>`

### Code Cleanup
- ✅ No `/v1/models` references remain in file
- ✅ No `fetch(` calls to dynamic server URLs
- ✅ `INITIAL_HEALTH_DELAY_MS` constant removed
- ✅ `healthCheckReady` state variable removed

### Functional Equivalence
- ✅ Model-loaded detection: `health.model_loaded === true` matches previous `data.data.length > 0`
- ✅ Timeout still works: 2-minute timeout unchanged
- ✅ PID check still happens first via `servers.status()`
- ✅ Poll interval unchanged: 3 seconds

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

**Blockers:** None

**Dependencies satisfied:**
- Health API already existed in `client.ts` (line 350)
- `HealthStatus` type already defined
- No new infrastructure needed

**Impact on other features:**
- StartingTile now consistent with other API usage patterns
- No more direct browser fetch to server ports
- All server communication goes through backend API

## Files Changed

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `frontend/src/lib/components/servers/StartingTile.svelte` | -17, +5 | Replaced fetch with health API, removed unused constants |

## Commit

- **Hash:** `80f8179`
- **Message:** `fix(06-14): use backend health API for startup polling`
- **Files:** StartingTile.svelte

## Testing Notes

**Manual verification needed:**
1. Start a server profile from the UI
2. Watch browser console during startup
3. Expected: No connection errors visible
4. Expected: Tile transitions to "Running" when model loads
5. Expected: Timeout after 2 minutes shows error tile

**Edge cases:**
- Server crash during startup → Should show error tile with details
- Slow model load → Should show "Starting" badge throughout
- User cancels during startup → Should stop polling cleanly
