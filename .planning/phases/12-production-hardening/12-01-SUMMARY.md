---
phase: 12-production-hardening
plan: 01
subsystem: observability
tags: [logfire, opentelemetry, fastapi, httpx, sqlalchemy, tracing, instrumentation]

# Dependency graph
requires:
  - phase: 07-server-foundation
    provides: MLX Server FastAPI app structure
  - phase: 11-configuration-ui
    provides: MLX Manager FastAPI app structure
provides:
  - LogFire observability modules for both apps
  - FastAPI request lifecycle tracing
  - HTTPX outbound HTTP tracing
  - SQLAlchemy database query tracing
  - OpenAI/Anthropic LLM token tracking (mlx_server only)
  - Offline development mode (send_to_logfire='if-token-present')
affects: [12-02, 12-03, 12-04]

# Tech tracking
tech-stack:
  added: [logfire[fastapi,httpx,sqlalchemy]]
  patterns: [centralized observability configuration, early module initialization, graceful degradation for optional dependencies]

key-files:
  created:
    - backend/mlx_manager/mlx_server/observability/__init__.py
    - backend/mlx_manager/mlx_server/observability/logfire_config.py
    - backend/mlx_manager/observability/__init__.py
    - backend/mlx_manager/observability/logfire_config.py
  modified:
    - backend/pyproject.toml
    - backend/mlx_manager/mlx_server/main.py
    - backend/mlx_manager/mlx_server/config.py
    - backend/mlx_manager/main.py

key-decisions:
  - "LogFire configured BEFORE any instrumented imports for proper tracing"
  - "send_to_logfire='if-token-present' enables offline development without token"
  - "LLM client instrumentation handles missing openai/anthropic packages gracefully"
  - "E402 exception added for both main.py files due to early module-level setup"

patterns-established:
  - "Observability module pattern: centralized configure_logfire() + instrument_* functions"
  - "Early initialization pattern: observability config before other instrumented imports"
  - "Graceful degradation: try/except for optional instrumentation dependencies"

# Metrics
duration: 5min
completed: 2026-01-31
---

# Phase 12 Plan 01: LogFire Integration Summary

**Full observability instrumentation for both apps with FastAPI, HTTPX, SQLAlchemy, and LLM client tracing using Pydantic LogFire**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-31T11:25:53Z
- **Completed:** 2026-01-31T11:31:00Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- LogFire observability modules created for both mlx_server and mlx_manager apps
- Full instrumentation stack: FastAPI requests, HTTPX outbound calls, SQLAlchemy queries
- MLX Server also instruments OpenAI/Anthropic clients for LLM token tracking
- Offline development mode with send_to_logfire='if-token-present'
- All 1190 tests passing (476 mlx_server + 714 manager)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add LogFire extras and create observability modules** - `d9ad4d2` (feat - previously committed as part of 12-02)
2. **Task 2: Integrate LogFire into MLX Server startup** - `a056361` (feat)
3. **Task 3: Integrate LogFire into MLX Manager startup** - `635296f` (feat)

## Files Created/Modified

- `backend/pyproject.toml` - Added logfire[fastapi,httpx,sqlalchemy]>=3.0.0, E402 exception for mlx_server/main.py
- `backend/mlx_manager/mlx_server/observability/__init__.py` - Module docstring
- `backend/mlx_manager/mlx_server/observability/logfire_config.py` - configure_logfire, instrument_fastapi, instrument_httpx, instrument_sqlalchemy, instrument_llm_clients
- `backend/mlx_manager/observability/__init__.py` - Module docstring
- `backend/mlx_manager/observability/logfire_config.py` - configure_logfire, instrument_fastapi, instrument_httpx, instrument_sqlalchemy
- `backend/mlx_manager/mlx_server/config.py` - Added environment field (development/production)
- `backend/mlx_manager/mlx_server/main.py` - Centralized LogFire config, removed inline configuration
- `backend/mlx_manager/main.py` - LogFire integration with FastAPI, HTTPX, SQLAlchemy instrumentation

## Decisions Made

- **LogFire configured BEFORE instrumented imports:** Per LogFire documentation, configure() must be called before importing instrumented libraries (httpx, sqlalchemy) for proper tracing. Both main.py files now do this at module level.
- **send_to_logfire='if-token-present':** Enables development without a LogFire token. In production, set LOGFIRE_TOKEN env var to enable cloud tracing.
- **Graceful LLM client instrumentation:** instrument_llm_clients() catches exceptions for missing openai/anthropic packages since they're optional dependencies.
- **E402 exception for both main.py:** Ruff config updated to allow module-level imports after observability setup.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] LLM client instrumentation fails without openai/anthropic packages**
- **Found during:** Task 2 (MLX Server integration)
- **Issue:** logfire.instrument_openai() raises ModuleNotFoundError if openai package not installed
- **Fix:** Wrapped instrument_openai() and instrument_anthropic() in try/except blocks with debug logging
- **Files modified:** backend/mlx_manager/mlx_server/observability/logfire_config.py
- **Verification:** Import succeeds without openai/anthropic packages
- **Committed in:** a056361 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Bug fix necessary for proper operation. No scope creep.

## Issues Encountered

- Task 1 files were already committed in d9ad4d2 (12-02 plan) - the error handling plan needed the observability infrastructure. No re-work needed, just proceeded with Task 2 and Task 3.
- The opentelemetry warning "Attempting to instrument while already instrumented" appears when importing both apps in the same process. This is expected and harmless - our _configured guard prevents double LogFire configuration.

## User Setup Required

**External services require optional configuration.** LogFire works in offline mode by default.

For cloud tracing (optional):
1. Create a LogFire account at https://logfire.pydantic.dev/
2. Generate an API token: Dashboard -> Settings -> API Tokens -> Create Token
3. Set environment variable: `export LOGFIRE_TOKEN=your-token`

Verification:
```bash
# Both apps should import successfully
cd backend && source .venv/bin/activate
python -c "from mlx_manager.mlx_server.main import app; from mlx_manager.main import app as manager_app; print('OK')"
```

## Next Phase Readiness

- LogFire observability infrastructure is in place
- Ready for Plan 02 (RFC 7807 Error Responses) - already complete
- Ready for Plan 03 (Request Timeouts) and Plan 04 (Comprehensive Testing)
- All instrumentation functions available for future observability needs

---
*Phase: 12-production-hardening*
*Completed: 2026-01-31*
