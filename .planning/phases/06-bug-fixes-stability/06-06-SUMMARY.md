---
phase: 06-bug-fixes-stability
plan: 06
subsystem: ui
tags: [svelte, chat, retry, error-handling, ux]

# Dependency graph
requires:
  - phase: 06-05
    provides: System prompt functionality with pinned message display
provides:
  - Retry-with-backoff logic for chat message sending
  - Progress indicator for connection attempts
  - Manual retry button after failures
  - User-friendly handling of model loading timing
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Linear backoff retry pattern (2s, 4s, 6s)
    - Failed message state for manual retry
    - Retry discrimination (server/network errors only)

key-files:
  created: []
  modified:
    - frontend/src/routes/(protected)/chat/+page.svelte

key-decisions:
  - "Linear backoff timing: 2s, 4s, 6s for 3 attempts"
  - "Only retry on server errors (5xx), gateway errors (502/503/504), and network errors"
  - "No retry on client errors (4xx) to avoid auth token exhaustion"
  - "Display retry progress during attempts: 'Connecting to model... (attempt X/3)'"
  - "Keep chat input functional during retries"

patterns-established:
  - "Retry pattern: sendWithRetry(content, attachments, attempt) recursive function with backoff"
  - "Failed message state pattern: store last failed message for manual retry"
  - "Retry progress UI: separate from loading state, shown only during retry attempts"

# Metrics
duration: 2min
completed: 2026-01-24
---

# Phase 6 Plan 06: Chat Retry-with-Backoff Summary

**Chat gracefully handles model loading with automatic retry-with-backoff (3 attempts, linear backoff) and manual retry button**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-24T11:09:49Z
- **Completed:** 2026-01-24T11:11:56Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Automatic retry with linear backoff (2s, 4s, 6s) when model server returns connection/server errors
- Visual progress indicator during retry attempts showing current attempt count
- Manual Retry button after all automatic attempts fail
- Chat input remains functional throughout retry process
- Smart retry discrimination (only on server/network errors, not client errors)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add retry-with-backoff to chat send function** - `43f696e` (feat)

## Files Created/Modified
- `frontend/src/routes/(protected)/chat/+page.svelte` - Added retry state, sendWithRetry() function, retry progress UI, and manual retry button

## Decisions Made

**1. Linear backoff timing: 2s, 4s, 6s**
- Simple, predictable timing
- Total wait time: 12 seconds across 3 attempts
- Enough time for models to finish loading into memory

**2. Retry only on server/network errors**
- Server errors (5xx): indicates model may still be loading
- Gateway errors (502, 503, 504): backend/proxy issues
- Network errors (TypeError, connection refused): connectivity issues
- NOT on client errors (4xx): auth, bad request, etc. (no point retrying)

**3. Keep chat input functional during retries**
- User can type and send new messages during retry
- New messages trigger their own retry logic if model isn't ready
- Better UX than locking input during wait period

**4. Separate retry progress from loading state**
- Retry progress shown only during isRetrying (attempt > 1)
- Normal loading spinner shown during first attempt and streaming
- Clear distinction between "first attempt" and "retrying"

**5. Manual retry after automatic failures**
- Store failed message content and attachments in lastFailedMessage
- Display Retry button when lastFailedMessage is set
- Clicking Retry re-adds user message and calls sendWithRetry(content, attachments, 1)
- User doesn't have to retype message or re-attach files

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - implementation straightforward, all quality checks passed.

## Next Phase Readiness

- Chat now gracefully handles model loading timing
- Final plan in Phase 6 (06-07) already complete
- Phase 6 complete after this plan
- Ready for project wrap-up

---
*Phase: 06-bug-fixes-stability*
*Completed: 2026-01-24*
