---
phase: 02-server-panel-redesign
plan: 05
subsystem: ui
tags: [svelte, reactivity, server-management, restart]

# Dependency graph
requires:
  - phase: 02-server-panel-redesign
    provides: ServerTile component, server store, server page filter logic
provides:
  - restartingProfiles SvelteSet in server store
  - isRestarting() method for checking restart state
  - Restarting badge display in ServerTile
  - Filter logic that keeps restarting servers mounted
affects: [server-management, restart-flow]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Separate state tracking for restart vs starting operations"
    - "Filter logic that accounts for transitional states"

key-files:
  created: []
  modified:
    - frontend/src/lib/stores/servers.svelte.ts
    - frontend/src/lib/components/servers/ServerTile.svelte
    - frontend/src/routes/servers/+page.svelte
    - frontend/src/lib/components/servers/ServerTile.test.ts

key-decisions:
  - "Track restarting state separately from starting state"
  - "Use restartingProfiles SvelteSet for proper reactivity"
  - "Transition from restarting to starting after backend confirms restart"

patterns-established:
  - "Transitional UI state tracking for multi-phase operations"

# Metrics
duration: 3min
completed: 2026-01-19
---

# Phase 2 Plan 5: Gap Closure - Restart Tile Disappearing Summary

**Fix server tile disappearing during restart by tracking restarting state separately and keeping ServerTile mounted throughout the restart operation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-19T12:37:14Z
- **Completed:** 2026-01-19T12:40:19Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Added restartingProfiles SvelteSet to track profiles in restart operation
- Added isRestarting() method to check restart state
- ServerTile now displays "Restarting..." badge with warning variant
- Filter logic updated to keep restarting servers in runningServers list
- Server tile remains visible throughout entire restart cycle

## Task Commits

Each task was committed atomically:

1. **Task 1: Add restartingProfiles state to server store** - `dbf12c3` (feat)
2. **Task 2: Update ServerTile to show Restarting badge** - `0356f20` (feat)
3. **Task 3: Update server page filters to keep restarting servers in runningServers** - `4e7d63b` (feat)

## Files Created/Modified
- `frontend/src/lib/stores/servers.svelte.ts` - Added restartingProfiles SvelteSet, isRestarting() method, updated restart/markStartupSuccess/markStartupFailed methods
- `frontend/src/lib/components/servers/ServerTile.svelte` - Added isRestarting derived state, conditional Restarting badge, disabled buttons during restart
- `frontend/src/routes/servers/+page.svelte` - Updated filter logic to exclude restarting from startingOrFailedProfiles and include in runningServers
- `frontend/src/lib/components/servers/ServerTile.test.ts` - Added isRestarting mock to fix test compatibility

## Decisions Made
- Track restarting state separately from starting state to enable smooth UI transitions
- Transition from restartingProfiles to startingProfiles after serversApi.restart() completes (backend confirms stop)
- Keep restarting servers in runningServers filter to prevent tile unmounting

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed ServerTile test mock**
- **Found during:** Task 3
- **Issue:** Test mock didn't include isRestarting method, causing all ServerTile tests to fail
- **Fix:** Added `isRestarting: vi.fn().mockReturnValue(false)` to the serverStore mock
- **Files modified:** `frontend/src/lib/components/servers/ServerTile.test.ts`
- **Verification:** All 308 tests now pass
- **Committed in:** `4e7d63b` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Auto-fix necessary for test compatibility. No scope creep.

## Issues Encountered
None - plan executed smoothly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Gap closure complete for restart tile disappearing issue
- UAT issue resolved - server tile now stays mounted during restart
- Ready for Phase 3 planning

---
*Phase: 02-server-panel-redesign*
*Completed: 2026-01-19*
