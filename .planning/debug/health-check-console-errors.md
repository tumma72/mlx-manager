---
status: resolved
trigger: "Investigate why health check polling still generates 'Failed to load resource: Could not connect to the server' browser console errors during server startup, despite a 5s delay fix applied in plan 06-08."
created: 2026-01-24T10:30:00Z
updated: 2026-01-24T10:30:00Z
---

## Current Focus

hypothesis: Browser fetch() API logs all network errors to console regardless of catch blocks
test: Code review of StartingTile.svelte health check logic
expecting: Confirm that try-catch cannot suppress browser's native console logging
next_action: Document root cause and affected code

## Symptoms

expected: No browser console errors during server startup when health checks fail
actual: Browser console shows "Failed to load resource: Could not connect to the server" errors
errors: Network errors logged by browser's fetch() implementation
reproduction: Start any server profile and watch browser console during startup
started: Always present - browser behavior, not a regression

## Eliminated

- hypothesis: 5s delay wasn't long enough
  evidence: Delay reduces frequency but doesn't eliminate errors because model loading time varies (can be >10s)
  timestamp: 2026-01-24T10:30:00Z

- hypothesis: Health check polling logic has a bug
  evidence: Code is correct - properly catches errors at lines 140-154 and 158-160
  timestamp: 2026-01-24T10:30:00Z

## Evidence

- timestamp: 2026-01-24T10:30:00Z
  checked: StartingTile.svelte lines 139-155
  found: Health check uses fetch() in try-catch block, errors silently caught
  implication: JavaScript code is handling errors correctly

- timestamp: 2026-01-24T10:30:00Z
  checked: Browser console behavior documentation
  found: Browser's fetch() implementation logs network errors to console BEFORE promise rejection
  implication: No way to prevent console errors using JavaScript alone

- timestamp: 2026-01-24T10:30:00Z
  checked: Lines 131-136
  found: 5s initial delay (INITIAL_HEALTH_DELAY_MS) before first health check
  implication: Reduces but cannot eliminate console errors for slow-loading models

- timestamp: 2026-01-24T10:30:00Z
  checked: Lines 10-11
  found: POLL_INTERVAL_MS = 3000 (3s between checks)
  implication: Multiple checks happen during typical model load time (15-30s)

## Resolution

root_cause: Browser's fetch() API logs network errors (connection refused) to console at the native level, before JavaScript error handling. Try-catch blocks cannot suppress this browser behavior.

fix: Not fixable via JavaScript - browser behavior is outside application control. Solutions require:
  1. Backend polling instead of frontend polling (backend checks health, frontend polls backend)
  2. Server-sent events (SSE) to notify frontend when ready
  3. Accept console errors as unavoidable during startup
  4. Increase delays further (not viable - increases perceived latency)

verification: Root cause confirmed through code analysis and browser API behavior

files_changed: []
