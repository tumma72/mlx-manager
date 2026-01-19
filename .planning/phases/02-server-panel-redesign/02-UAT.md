---
status: diagnosed
phase: 02-server-panel-redesign
source: 02-01-SUMMARY.md, 02-02-SUMMARY.md, 02-03-SUMMARY.md, 02-04-SUMMARY.md
started: 2026-01-19T10:00:00Z
updated: 2026-01-19T10:01:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Server Metrics in API Response
expected: API response from /api/servers includes memory_percent and cpu_percent fields for each running server.
result: pass

### 2. Profile Dropdown Search
expected: On the servers page, a searchable dropdown lets you type to filter profiles by name or model path. Selecting a profile populates the dropdown.
result: pass

### 3. Start Server from Dropdown
expected: After selecting a profile, clicking "Start" launches the server. The selection clears after successful start.
result: pass

### 4. Running Server Tile Display
expected: Running servers appear as tiles showing model name, port, memory gauge (circular), CPU gauge (circular), and uptime.
result: pass

### 5. Server Control Buttons
expected: Running server tiles have Stop, Restart, and Chat buttons. Stop terminates the server. Restart stops and restarts. Chat navigates to chat page with that server.
result: issue
reported: "pass but when restarting the tile of the server completely disappears for a couple of seconds, and then reappears, it would be better to show status changes 'restarting' or 'shutting down...' then 'starting...' which we already have, and then 'running' again, without destroying the tile and recreating it"
severity: minor

### 6. Starting Server Status
expected: After clicking Start, a starting tile shows the profile with a "Starting..." status. If startup fails, error details display with copy-to-clipboard functionality.
result: pass

### 7. Profile Dropdown Stability
expected: The profile dropdown does not reset or flicker every 5 seconds when polling updates occur. Selected profile remains stable.
result: pass

### 8. Scroll Preservation
expected: When viewing multiple running servers and scrolling down, the scroll position is maintained during 5-second polling updates. No jumping back to top.
result: pass

## Summary

total: 8
passed: 7
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "Restart shows smooth status transitions without tile disappearing"
  status: failed
  reason: "User reported: pass but when restarting the tile of the server completely disappears for a couple of seconds, and then reappears, it would be better to show status changes 'restarting' or 'shutting down...' then 'starting...' which we already have, and then 'running' again, without destroying the tile and recreating it"
  severity: minor
  test: 5
  root_cause: "Server page uses two separate tile lists (ServerTile for running, StartingTile for starting) with mutually exclusive filters. During restart, startingProfiles.add() immediately removes server from runningServers (unmounting ServerTile), but StartingTile may not appear immediately due to reactive update batching, causing visible gap."
  artifacts:
    - path: "frontend/src/routes/servers/+page.svelte"
      issue: "Lines 49-60: mutually exclusive filter conditions between runningServers and startingOrFailedProfiles"
    - path: "frontend/src/lib/stores/servers.svelte.ts"
      issue: "Lines 193-199: restart() adds to startingProfiles before API completes, causing immediate state transition"
  missing:
    - "Add restartingProfiles SvelteSet to track profiles currently restarting"
    - "Keep ServerTile mounted with 'Restarting...' badge while in restartingProfiles set"
    - "Only transition to StartingTile after backend confirms server stopped"
  debug_session: ".planning/debug/server-tile-disappears-restart.md"
