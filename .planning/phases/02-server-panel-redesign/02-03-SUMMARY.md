---
phase: 02-server-panel-redesign
plan: 03
subsystem: ui
tags: [svelte, tailwind, svg-gauge, server-metrics, reactive-state]

# Dependency graph
requires:
  - phase: 02-01
    provides: Extended server metrics API (memory_percent, cpu_percent)
  - phase: 02-02
    provides: ProfileSelector component for starting servers
provides:
  - MetricGauge SVG circular progress component
  - ServerTile component with metrics display and controls
  - StartingTile component for starting/failed server status
  - Full status lifecycle UI (stopped -> starting -> running/error)
affects: [02-04-PLAN.md, chat-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - SVG stroke-dasharray for circular progress gauges
    - Stable array references to prevent Combobox state reset
    - Server list-based running state detection

key-files:
  created:
    - frontend/src/lib/components/servers/MetricGauge.svelte
    - frontend/src/lib/components/servers/ServerTile.svelte
    - frontend/src/lib/components/servers/StartingTile.svelte
  modified:
    - frontend/src/lib/components/servers/index.ts
    - frontend/src/lib/components/servers/ProfileSelector.svelte
    - frontend/src/routes/servers/+page.svelte
    - frontend/src/routes/chat/+page.svelte

key-decisions:
  - "Stabilize ProfileSelector profiles via ID comparison to prevent polling flicker"
  - "Create separate StartingTile component for starting/failed states"
  - "Use servers list directly for running state detection in chat page"

patterns-established:
  - "MetricGauge: Reusable SVG circular progress with color thresholds"
  - "Array stabilization: Compare IDs to detect actual changes vs polling updates"

# Metrics
duration: 15min
completed: 2026-01-17
---

# Phase 2 Plan 3: ServerTile with Metrics Gauges Summary

**Rich server tiles with SVG circular gauges for memory/CPU, plus StartingTile for full status lifecycle with error handling and copy-to-clipboard**

## Performance

- **Duration:** ~15 min (including continuation after user feedback)
- **Tasks:** 5 (4 initial + 1 bug fix continuation)
- **Files created:** 3
- **Files modified:** 4

## Accomplishments

- MetricGauge component with SVG circular progress, color thresholds (green/yellow/red)
- ServerTile component displaying running servers with memory, CPU, uptime, and controls
- StartingTile component for starting/failed status with full error details and copy-to-clipboard
- Fixed ProfileSelector polling flicker that reset dropdown state every 5 seconds
- Fixed chat page integration to correctly navigate from server dashboard

## Task Commits

1. **Task 1: Create MetricGauge component** - `daff165` (feat)
2. **Task 2: Create ServerTile component** - `c3ebb5c` (feat)
3. **Task 3: Export components from servers index** - `7a67e84` (feat)
4. **Task 4: Integrate ServerTile into server page** - `5d25104` (feat)
5. **Task 5 (continuation): Fix regressions from user feedback** - `2725b2b` (fix)

## Files Created/Modified

- `frontend/src/lib/components/servers/MetricGauge.svelte` - SVG circular gauge with color thresholds
- `frontend/src/lib/components/servers/ServerTile.svelte` - Running server tile with metrics and controls
- `frontend/src/lib/components/servers/StartingTile.svelte` - Starting/failed status with error handling
- `frontend/src/lib/components/servers/index.ts` - Barrel exports for server components
- `frontend/src/lib/components/servers/ProfileSelector.svelte` - Stabilized array reference to prevent flicker
- `frontend/src/routes/servers/+page.svelte` - Integrated ServerTile and StartingTile sections
- `frontend/src/routes/chat/+page.svelte` - Fixed running profile detection and URL param reactivity

## Decisions Made

1. **Array stabilization for ProfileSelector**: Rather than letting the profiles array reference change on every poll cycle (which reset Combobox state), we now compare profile IDs to detect actual changes. Only update the stable array when the set of profiles actually changes.

2. **Separate StartingTile component**: Created a new component specifically for starting/failed states rather than overloading ServerTile. This keeps ServerTile focused on healthy running servers with metrics, while StartingTile handles the status lifecycle (polling, timeout, error display).

3. **Direct server list for running detection**: In the chat page, changed from `serverStore.isRunning(p.id)` (which considers `startingProfiles`) to directly checking if profile ID is in `serverStore.servers`. This ensures servers that appear in the list are considered running for chat purposes.

## Deviations from Plan

### User-Reported Issues Fixed

**1. [Bug] ProfileSelector polling flicker**
- **Found during:** User verification checkpoint
- **Issue:** 5-second polling reset dropdown state, causing selection loss if user took >5s
- **Fix:** Stabilize profiles array by comparing IDs, only update reference on actual changes
- **Files modified:** ProfileSelector.svelte
- **Committed in:** 2725b2b

**2. [Regression] Lost start statuses and error handling**
- **Found during:** User verification checkpoint
- **Issue:** Previous UI had status badges, error display, and copy-to-clipboard that were lost in redesign
- **Fix:** Created StartingTile component with full status lifecycle, error details, and clipboard functionality
- **Files modified:** StartingTile.svelte (new), index.ts, +page.svelte
- **Committed in:** 2725b2b

**3. [Bug] Chat button wiring broken**
- **Found during:** User verification checkpoint
- **Issue:** Clicking chat navigated to `/chat?profile={id}` but page showed "no running server"
- **Fix:** Changed running detection to use server list directly, made URL param handling reactive
- **Files modified:** chat/+page.svelte
- **Committed in:** 2725b2b

---

**Total deviations:** 3 bugs fixed during user verification
**Impact on plan:** Essential fixes for correct operation. All issues were regressions from the redesign.

## Known Gaps

**Throughput metrics not available**: User requested tokens/s and message count per server. Investigation confirmed the backend API does not provide this data - `RunningServer` only includes memory, CPU, and uptime metrics. The mlx-openai-server process would need to expose additional metrics for throughput tracking.

**Recommendation:** Create a future task to investigate exposing throughput metrics from mlx-openai-server, potentially via a /v1/stats endpoint or log parsing.

## Issues Encountered

None beyond the user-reported regressions addressed above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Server tiles complete with metrics and status lifecycle
- Ready for 02-04 (final polish/integration) if planned
- Throughput metrics documented as gap for future planning

---
*Phase: 02-server-panel-redesign*
*Completed: 2026-01-17*
