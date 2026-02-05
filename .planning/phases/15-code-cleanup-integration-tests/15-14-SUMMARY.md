---
phase: 15
plan: 14
subsystem: downloads
tags: [download-management, pause-resume, hf-cache, sse, threading]
dependency-graph:
  requires: []
  provides:
    - Download pause/resume/cancel API endpoints
    - Threading-based download cancellation infrastructure
    - HF cache cleanup on cancel
    - Frontend download management UI buttons
  affects: []
tech-stack:
  added: []
  patterns:
    - threading.Event for cross-task cancellation signaling
    - Inline cancel confirmation pattern in Svelte components
key-files:
  created:
    - backend/tests/test_download_management.py
  modified:
    - backend/mlx_manager/models.py
    - backend/mlx_manager/services/hf_client.py
    - backend/mlx_manager/routers/models.py
    - backend/mlx_manager/database.py
    - backend/mlx_manager/main.py
    - backend/tests/test_models.py
    - frontend/src/lib/api/client.ts
    - frontend/src/lib/api/types.ts
    - frontend/src/lib/stores/downloads.svelte.ts
    - frontend/src/lib/components/models/DownloadProgressTile.svelte
decisions:
  - id: threading-event-cancellation
    description: "Use threading.Event for download cancellation signaling between async endpoints and executor-based downloads"
  - id: paused-not-auto-resumed
    description: "Paused downloads stay paused on server restart - user must explicitly click Resume"
  - id: inline-cancel-confirm
    description: "Cancel confirmation is inline in DownloadProgressTile (Confirm/Keep) rather than a modal dialog"
  - id: amber-color-paused
    description: "Paused state uses amber/yellow color for progress bar and status to distinguish from active downloads"
metrics:
  duration: "10 min"
  completed: "2026-02-05"
  tasks: 6/6
  tests-added: 27
  tests-total-backend: 1326
  tests-total-frontend: 970
---

# Phase 15 Plan 14: Download Management Summary

**One-liner:** Pause, resume, and cancel controls for model downloads with threading-based cancellation, HF cache cleanup, and persistent state across restarts.

## What Was Built

### Backend Infrastructure
1. **Cancellation mechanism** (`hf_client.py`): Module-level `_cancel_events` dict with `threading.Event` instances per download. Functions: `register_cancel_event()`, `request_cancel()`, `cleanup_cancel_event()`.

2. **download_model() cancel support**: The polling loop checks `cancel_event.is_set()` each iteration. When set, cancels the executor Future and yields a "cancelled" status.

3. **HF cache cleanup** (`cleanup_partial_download()`): Uses `huggingface_hub.scan_cache_dir()` to find and delete all revisions for a model. Falls back to `shutil.rmtree` on the cache directory if the API fails.

4. **Three new API endpoints**:
   - `POST /api/models/download/{id}/pause` - Signals cancel event, sets status to "paused"
   - `POST /api/models/download/{id}/resume` - Sets status to "downloading", returns new task_id for SSE reconnection
   - `POST /api/models/download/{id}/cancel` - Signals cancel, sets "cancelled", runs cache cleanup

5. **Recovery behavior**: Downloads in "downloading" state auto-resume on restart (existing behavior). Paused downloads stay paused. Cancelled downloads stay cancelled.

6. **Active downloads listing**: Now includes paused downloads with `download_id` for frontend pause/resume/cancel actions.

### Frontend
1. **API client methods**: `pauseDownload()`, `resumeDownload()`, `cancelDownload()` in `client.ts`
2. **Store methods**: `pauseDownload()`, `resumeDownload()`, `cancelDownload()`, `isPaused()` in `downloads.svelte.ts` with SSE lifecycle management
3. **DownloadProgressTile UI**: Contextual action buttons based on state:
   - Downloading: [Pause] + [Cancel]
   - Paused: [Resume] + [Cancel]
   - Cancel shows inline confirmation (Confirm/Keep)
   - Loading spinners on buttons during API calls
   - Amber/yellow color theme for paused state

### Test Coverage
27 new tests in `test_download_management.py`:
- Cancel event mechanism (6 tests)
- HF cache cleanup (3 tests)
- Pause endpoint (4 tests)
- Resume endpoint (3 tests)
- Cancel endpoint (6 tests)
- Active downloads listing (2 tests)
- Recovery on restart (3 tests)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fix existing test mocks for cancel_event parameter**
- **Found during:** Task 6
- **Issue:** Adding `cancel_event` parameter to `download_model()` broke 5 existing test mocks in `test_models.py` that didn't accept keyword arguments
- **Fix:** Added `cancel_event=None` parameter to all mock `download_model` functions
- **Files modified:** `backend/tests/test_models.py`
- **Commit:** 9536d5c

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 66e3a7d | feat | Add download cancellation infrastructure and state fields |
| 46a4e5d | feat | Add pause, resume, and cancel API endpoints |
| 0fc735d | feat | Handle paused downloads in recovery and active listing |
| d8562bb | feat | Add download management to frontend store and API client |
| bd33d3b | feat | Add pause, resume, and cancel buttons to DownloadProgressTile |
| 9536d5c | test | Add unit tests for download pause/resume/cancel management |

## Verification Results

- Backend endpoints registered: pause, resume, cancel confirmed via route inspection
- Backend tests: 1326 passed, 0 failed
- Frontend type check: 0 errors, 0 warnings
- Frontend tests: 970 passed
- Frontend lint: 0 errors (2 pre-existing warnings in coverage/)
