---
status: testing
phase: 12-production-hardening
source: [12-01-SUMMARY.md, 12-02-SUMMARY.md, 12-03-SUMMARY.md, 12-04-SUMMARY.md, 12-05-SUMMARY.md, 12-06-SUMMARY.md, 12-07-SUMMARY.md]
started: 2026-01-31T12:00:00Z
updated: 2026-01-31T12:00:00Z
---

## Current Test

number: 6
name: Audit Log Table
expected: |
  Audit log table shows columns: Timestamp, Model, Backend, Endpoint, Duration, Status, Tokens.
  If no requests yet, shows empty state.
awaiting: user response

## Tests

### 1. LogFire Observability Configuration
expected: Both apps import successfully with LogFire initialized (no LOGFIRE_TOKEN needed for offline mode)
result: pass

### 2. RFC 7807 Error Response Format
expected: API errors return JSON with type, title, status, detail, instance, request_id fields. Test: call an invalid endpoint or trigger a validation error.
result: issue
reported: "404 for nonexistent route returns default FastAPI format {\"detail\": \"Not Found\"} instead of RFC 7807 Problem Details format"
severity: major

### 3. Request Timeout Configuration
expected: Settings page shows "Request Timeouts" section with sliders for Chat (15min default), Completions (10min), and Embeddings (2min)
result: pass

### 4. Timeout Slider Controls
expected: Moving a slider updates the number input and vice versa. Clicking Save persists values.
result: pass

### 5. Audit Log Panel Visible
expected: Settings page shows "Audit Logs" section at the bottom with stats grid (Total, Successful, Errors, Unique Models)
result: pass

### 6. Audit Log Table
expected: Audit log table shows columns: Timestamp, Model, Backend, Endpoint, Duration, Status, Tokens. If no requests yet, shows empty state.
result: [pending]

### 7. Audit Log Filters
expected: Filter dropdowns for Model, Backend (LOCAL/OPENAI/ANTHROPIC), and Status (success/error) filter the log entries
result: [pending]

### 8. Audit Log Export
expected: Export buttons (JSONL, CSV) download audit log data in respective formats
result: [pending]

### 9. CLI Benchmark Tool Available
expected: Running `mlx-benchmark --help` shows CLI help with run and suite commands
result: [pending]

### 10. Performance Documentation
expected: docs/PERFORMANCE.md exists with benchmark methodology, sample results, and optimization recommendations
result: [pending]

## Summary

total: 10
passed: 4
issues: 1
pending: 5
skipped: 0

## Gaps

- truth: "API errors return JSON with type, title, status, detail, instance, request_id fields (RFC 7807)"
  status: failed
  reason: "User reported: 404 for nonexistent route returns default FastAPI format {\"detail\": \"Not Found\"} instead of RFC 7807 Problem Details format"
  severity: major
  test: 2
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
