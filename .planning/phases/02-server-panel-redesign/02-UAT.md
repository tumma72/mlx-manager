---
status: complete
phase: 02-server-panel-redesign
source: 02-01-SUMMARY.md, 02-02-SUMMARY.md, 02-03-SUMMARY.md, 02-04-SUMMARY.md, 02-05-SUMMARY.md
started: 2026-01-19T10:00:00Z
updated: 2026-01-20T10:00:00Z
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
expected: Running server tiles have Stop, Restart, and Chat buttons. Stop terminates the server. Restart stops and restarts with smooth status transitions (Restarting → Starting → Running). Chat navigates to chat page with that server.
result: pass
note: Re-verified after 02-05 gap closure fix

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
passed: 8
issues: 0
pending: 0
skipped: 0

## Gaps

[none - all issues resolved]
