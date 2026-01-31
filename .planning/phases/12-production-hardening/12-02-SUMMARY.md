---
phase: 12-production-hardening
plan: 02
subsystem: api
tags: [rfc-7807, problem-details, error-handling, fastapi, request-id]

# Dependency graph
requires:
  - phase: 12-production-hardening
    provides: Production hardening foundation
provides:
  - RFC 7807 Problem Details error format
  - request_id generation for log correlation
  - X-Request-ID response header
  - TimeoutHTTPException for timeout errors
  - Generic exception handler that hides internals
affects: [12-03, 12-04, all-api-consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - RFC 7807 Problem Details for all API errors
    - request_id generation with uuid4 prefix
    - X-Request-ID header for correlation

key-files:
  created:
    - backend/mlx_manager/mlx_server/errors/__init__.py
    - backend/mlx_manager/mlx_server/errors/problem_details.py
    - backend/mlx_manager/mlx_server/errors/handlers.py
    - backend/tests/mlx_server/test_error_handlers.py
  modified:
    - backend/mlx_manager/mlx_server/main.py

key-decisions:
  - "Type ignore for FastAPI exception handlers - Starlette type stubs expect generic Exception but specific types work at runtime"
  - "request_id format: req_{12-char-hex} - prefix makes IDs identifiable in logs"
  - "Problem type URIs use mlx-manager.dev domain for namespacing"

patterns-established:
  - "RFC 7807 Problem Details: All API errors return type/title/status/detail/instance/request_id"
  - "Error handler registration: Call register_error_handlers(app) after FastAPI creation"
  - "Internal error hiding: 500 errors return generic message, full exception logged server-side"

# Metrics
duration: 3min
completed: 2026-01-31
---

# Phase 12 Plan 02: RFC 7807 Error Responses Summary

**RFC 7807 Problem Details error responses with request_id for log correlation across all MLX Server API errors**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-31T11:26:52Z
- **Completed:** 2026-01-31T11:29:24Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- ProblemDetail Pydantic model implementing RFC 7807 standard
- TimeoutHTTPException with specialized TimeoutProblem response
- Exception handlers for HTTPException, RequestValidationError, and generic Exception
- request_id generation with X-Request-ID header for log correlation
- Internal error details (stack traces) never exposed to clients
- Comprehensive test suite with 7 tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Create RFC 7807 Problem Details models and exception classes** - `d9ad4d2` (feat)
2. **Task 2: Create exception handlers with request_id generation** - `7701b29` (feat)
3. **Task 3: Integrate error handlers into MLX Server and add tests** - `3aa055d` (feat)

## Files Created/Modified
- `backend/mlx_manager/mlx_server/errors/__init__.py` - Module exports
- `backend/mlx_manager/mlx_server/errors/problem_details.py` - RFC 7807 models
- `backend/mlx_manager/mlx_server/errors/handlers.py` - FastAPI exception handlers
- `backend/mlx_manager/mlx_server/main.py` - Registered error handlers
- `backend/tests/mlx_server/test_error_handlers.py` - Test suite

## Decisions Made
- Used `type: ignore[arg-type]` for FastAPI exception handler registration - Starlette's type stubs expect generic Exception handlers but specific exception types work correctly at runtime
- request_id format `req_{uuid4.hex[:12]}` - prefix makes IDs identifiable in logs
- Problem type URIs use `https://mlx-manager.dev/errors/` namespace
- Status code to problem type mapping for common HTTP errors (400, 401, 403, 404, 422, 429, 500, 503)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- mypy type errors for exception handler registration - resolved with documented type ignore comments

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Error handling foundation complete
- Ready for PROD-03 (Logging/Metrics) and PROD-04 (Health/Readiness)
- request_id available for log correlation in future observability work

---
*Phase: 12-production-hardening*
*Completed: 2026-01-31*
