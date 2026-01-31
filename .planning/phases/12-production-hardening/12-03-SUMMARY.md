---
phase: 12-production-hardening
plan: 03
subsystem: timeout-handling
tags: [asyncio, timeout, request-timeout, error-handling]
requires: [12-02]
provides: [endpoint-timeouts, timeout-decorator, streaming-timeout]
affects: []
tech-stack:
  added: []
  patterns: [asyncio-wait-for, sse-error-events, per-endpoint-timeout]
key-files:
  created:
    - backend/mlx_manager/mlx_server/middleware/__init__.py
    - backend/mlx_manager/mlx_server/middleware/timeout.py
    - backend/tests/mlx_server/test_timeout.py
  modified:
    - backend/mlx_manager/mlx_server/config.py
    - backend/mlx_manager/mlx_server/api/v1/chat.py
    - backend/mlx_manager/mlx_server/api/v1/completions.py
    - backend/mlx_manager/mlx_server/api/v1/embeddings.py
    - backend/tests/mlx_server/api/v1/test_chat_routing.py
decisions:
  - id: timeout-tiers
    choice: "Per-endpoint timeouts (Chat: 15min, Completions: 10min, Embeddings: 2min)"
    rationale: "Different endpoint types have different expected durations"
  - id: timeout-error-format
    choice: "TimeoutHTTPException with RFC 7807 Problem Details"
    rationale: "Consistent with 12-02 error handling pattern"
  - id: streaming-timeout-error
    choice: "SSE error event with type and message"
    rationale: "Graceful degradation for streaming clients"
metrics:
  duration: 5m23s
  completed: 2026-01-31
---

# Phase 12 Plan 03: Request Timeouts Summary

Per-endpoint configurable timeouts using asyncio.wait_for with RFC 7807 error responses

## Key Deliverables

### Timeout Settings (config.py)
```python
timeout_chat_seconds: float = 900.0      # 15 minutes
timeout_completions_seconds: float = 600.0  # 10 minutes
timeout_embeddings_seconds: float = 120.0   # 2 minutes
```

All configurable via environment variables (`MLX_SERVER_TIMEOUT_*_SECONDS`).

### Timeout Decorator (middleware/timeout.py)
```python
def with_timeout(seconds: float) -> Callable[...]:
    """Decorator using asyncio.wait_for to enforce timeouts."""
```

Raises `TimeoutHTTPException` on timeout, which triggers RFC 7807 Problem Details response.

### Endpoint Integration

1. **Chat completions** (`/v1/chat/completions`):
   - Non-streaming: asyncio.wait_for around generate_chat_completion
   - Streaming: Timeout within async generator
   - Vision: Timeout on generate_vision_completion
   - Routed (cloud): Timeout on route_request
   - Batched: Timeout on token collection

2. **Completions** (`/v1/completions`):
   - Non-streaming: asyncio.wait_for around generate_completion
   - Streaming: Timeout within async generator

3. **Embeddings** (`/v1/embeddings`):
   - asyncio.wait_for around generate_embeddings

### Streaming Error Format
```json
{
  "error": {
    "type": "https://mlx-manager.dev/errors/timeout",
    "message": "Request timed out after 900 seconds"
  }
}
```

Sent as SSE event before closing connection.

## Test Coverage

15 new tests in `test_timeout.py`:
- Decorator behavior (fast/slow functions, metadata preservation, argument handling)
- Settings defaults verification
- TimeoutHTTPException properties
- get_timeout_for_endpoint helper

All 491 MLX server tests pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed mock settings in test_chat_routing.py**
- **Found during:** Task 3 verification
- **Issue:** Mock settings didn't include timeout_chat_seconds, causing TypeError
- **Fix:** Added `settings.timeout_chat_seconds = 900.0` to mock
- **Files modified:** backend/tests/mlx_server/api/v1/test_chat_routing.py
- **Commit:** 047ebc5

**2. [Rule 1 - Bug] Applied linting fixes**
- **Found during:** Final verification
- **Issue:** UP041 (asyncio.TimeoutError should be TimeoutError), line length
- **Fix:** Ran `ruff check --fix`, split long line
- **Files modified:** middleware/timeout.py, chat.py, completions.py, embeddings.py
- **Commit:** 848de41

## Verification Checklist

- [x] Timeout settings exist with correct defaults
- [x] with_timeout decorator raises TimeoutHTTPException on timeout
- [x] All inference endpoints have timeout handling
- [x] Streaming endpoints send error event on timeout
- [x] Timeouts configurable via environment variables
- [x] All timeout tests pass (15/15)
- [x] Same timeouts apply to local and cloud backends

## Technical Notes

1. **Timeout application strategy:**
   - For non-streaming: Wrap the generation call in asyncio.wait_for
   - For streaming: Apply timeout within the async generator using try/except
   - For batched streaming: Track elapsed time in loop (simpler than nested asyncio)

2. **TimeoutError vs asyncio.TimeoutError:**
   - Python 3.11+ prefers builtin TimeoutError
   - Ruff UP041 enforces this for cleaner code

## Next Phase Readiness

Phase 12 Plan 04 (Comprehensive Testing) can proceed. All timeout infrastructure is in place with:
- Configurable per-endpoint timeouts
- Proper error responses via RFC 7807
- SSE error events for streaming
- 15 new tests covering timeout functionality
