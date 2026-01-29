---
phase: 11-configuration-ui
plan: 01
subsystem: api
tags: [fernet, encryption, pbkdf2, settings, crud, routing-rules, model-pool]

# Dependency graph
requires:
  - phase: 10-dual-protocol-cloud-fallback
    provides: CloudCredential and BackendMapping models
provides:
  - Fernet-based API key encryption service
  - Settings CRUD endpoints for providers, rules, pool config
  - Pattern matching (exact, prefix, regex) for routing rules
  - Model pool configuration (memory limit, eviction policy, preload)
affects: [11-02-frontend-settings, phase-12-production]

# Tech tracking
tech-stack:
  added: [cryptography]
  patterns: [fernet-encryption, singleton-config, static-routes-before-dynamic]

key-files:
  created:
    - backend/mlx_manager/services/encryption_service.py
    - backend/mlx_manager/routers/settings.py
    - backend/tests/test_encryption_service.py
    - backend/tests/test_settings_router.py
  modified:
    - backend/mlx_manager/models.py
    - backend/mlx_manager/routers/__init__.py
    - backend/mlx_manager/main.py

key-decisions:
  - "PBKDF2HMAC with 1.2M iterations for key derivation from jwt_secret"
  - "Salt persists in ~/.mlx-manager/.encryption_salt alongside database"
  - "Static routes (/rules/priorities, /rules/test) before dynamic routes (/rules/{rule_id})"
  - "ServerConfig as singleton (id=1) for global pool settings"

patterns-established:
  - "Encryption service pattern: lru_cache Fernet instance for performance"
  - "Upsert pattern: POST creates or updates by unique backend_type"
  - "Route ordering: specific paths before path parameters"

# Metrics
duration: 12min
completed: 2026-01-29
---

# Phase 11 Plan 01: Backend Encryption & Settings API Summary

**Fernet-based API key encryption with PBKDF2 key derivation, and full CRUD settings router for providers, routing rules, and model pool configuration**

## Performance

- **Duration:** 12 min
- **Started:** 2026-01-29T17:21:35Z
- **Completed:** 2026-01-29T17:33:00Z
- **Tasks:** 3
- **Files created/modified:** 7
- **Tests added:** 66 (17 encryption + 49 settings router)

## Accomplishments

- Fernet encryption service for API keys with PBKDF2 key derivation (1.2M iterations)
- Salt file persistence in ~/.mlx-manager/ for encryption key stability
- Settings router with 12 endpoints for providers, routing rules, and pool config
- Pattern matching supporting exact, prefix, and regex types for routing rules
- API keys never returned in responses (security by design)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create encryption service for API keys** - `4d54a57` (feat)
2. **Task 2: Add model pool configuration model** - `c4a28e8` (feat)
3. **Task 3: Create settings router with full CRUD** - `bd1d02f` (feat)

## Files Created/Modified

**Created:**
- `backend/mlx_manager/services/encryption_service.py` - Fernet encryption with PBKDF2 key derivation
- `backend/mlx_manager/routers/settings.py` - Settings CRUD for providers, rules, pool config
- `backend/tests/test_encryption_service.py` - 17 tests for encryption roundtrip, failures, persistence
- `backend/tests/test_settings_router.py` - 49 tests covering all endpoints

**Modified:**
- `backend/mlx_manager/models.py` - Added ServerConfig, pattern_type to BackendMapping, helper schemas
- `backend/mlx_manager/routers/__init__.py` - Export settings_router
- `backend/mlx_manager/main.py` - Include settings_router

## API Endpoints Added

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | /api/settings/providers | List configured providers (no keys) |
| POST | /api/settings/providers | Create/update provider (upsert) |
| DELETE | /api/settings/providers/{backend_type} | Remove provider |
| POST | /api/settings/providers/{backend_type}/test | Test connection |
| GET | /api/settings/rules | List rules by priority |
| POST | /api/settings/rules | Create rule |
| PUT | /api/settings/rules/priorities | Batch update priorities |
| POST | /api/settings/rules/test | Test rule matching |
| PUT | /api/settings/rules/{rule_id} | Update rule |
| DELETE | /api/settings/rules/{rule_id} | Delete rule |
| GET | /api/settings/pool | Get pool config |
| PUT | /api/settings/pool | Update pool config |

## Decisions Made

1. **PBKDF2HMAC with 1.2M iterations** - Industry standard for key derivation, resistant to brute force
2. **Salt file persistence** - Stored alongside database in ~/.mlx-manager/ for consistency
3. **lru_cache for Fernet instance** - Avoid repeated key derivation overhead
4. **Static routes before dynamic** - /rules/priorities and /rules/test must come before /rules/{rule_id}
5. **ServerConfig singleton** - Use id=1 for global pool settings, created on first access

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Install cryptography package**
- **Found during:** Task 1 (Encryption service creation)
- **Issue:** cryptography package not installed despite pwdlib[argon2] dependency
- **Fix:** `uv pip install cryptography>=43.0.0`
- **Verification:** Import succeeds, tests pass
- **Committed in:** Part of development, not tracked in commit

**2. [Rule 1 - Bug] Route ordering for static vs dynamic paths**
- **Found during:** Task 3 (Settings router testing)
- **Issue:** /rules/priorities matched by /rules/{rule_id} with rule_id="priorities"
- **Fix:** Moved static routes (/rules/priorities, /rules/test) before dynamic routes
- **Files modified:** backend/mlx_manager/routers/settings.py
- **Verification:** All 49 router tests pass
- **Committed in:** bd1d02f (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered

None - plan executed smoothly after auto-fixes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Backend encryption and settings API complete
- Ready for Phase 11 Plan 02: Frontend settings UI components
- Endpoints return proper response schemas for frontend consumption
- API key security verified (never returned in responses)

---
*Phase: 11-configuration-ui*
*Plan: 01*
*Completed: 2026-01-29*
