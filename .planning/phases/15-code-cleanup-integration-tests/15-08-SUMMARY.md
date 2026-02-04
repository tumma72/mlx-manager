---
phase: 15-code-cleanup-integration-tests
plan: 08
subsystem: database, api
tags: [sqlmodel, pydantic, fastapi, svelte, migration]

# Dependency graph
requires:
  - phase: 13-mlx-server-integration
    provides: embedded MLX server replacing external mlx-openai-server
provides:
  - ServerProfile with generation parameters (temperature, max_tokens, top_p)
  - Simplified profile model without obsolete server fields
  - Chat endpoint using profile generation defaults with request overrides
affects: [ui-profiles, chat-api, future-model-config]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Profile-level generation defaults with request-level overrides

key-files:
  created: []
  modified:
    - backend/mlx_manager/models.py
    - backend/mlx_manager/database.py
    - backend/mlx_manager/routers/chat.py
    - backend/mlx_manager/routers/profiles.py
    - frontend/src/lib/api/types.ts
    - frontend/src/lib/components/profiles/ProfileForm.svelte

key-decisions:
  - "Profile stores generation defaults, request can override"
  - "Removed 14 obsolete fields from ServerProfile model"
  - "ServerTile shows PID instead of port number"

patterns-established:
  - "Generation parameter cascade: request > profile > system default"

# Metrics
duration: ~15min
completed: 2026-02-04
---

# Phase 15 Plan 08: Profile Model Cleanup Summary

**Removed 14 obsolete ServerProfile fields (port, host, parsers, queue settings) and added generation parameters (temperature, max_tokens, top_p) with profile-level defaults that can be overridden per request**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-02-04T10:00:00Z
- **Completed:** 2026-02-04T13:12:45Z
- **Tasks:** 1 (monolithic cleanup task)
- **Files modified:** 22

## Accomplishments

- Removed 14 obsolete fields from ServerProfile: port, host, max_concurrency, queue_timeout, queue_size, tool_call_parser, reasoning_parser, message_converter, enable_auto_tool_choice, trust_remote_code, chat_template_file, log_level, log_file, no_log_file
- Added generation parameters with validation: temperature (0.0-2.0, default 0.7), max_tokens (1-128000, default 4096), top_p (0.0-1.0, default 1.0)
- Updated chat endpoint to use profile generation settings with request-level overrides
- Simplified ProfileForm.svelte with new Generation Settings section
- Updated all frontend types, stores, and 12 test files

## Task Commits

Each task was committed atomically:

1. **Task 1: Profile model cleanup** - `fdca684` (refactor)

## Files Created/Modified

### Backend
- `backend/mlx_manager/models.py` - Removed 14 fields, added 3 generation parameters
- `backend/mlx_manager/database.py` - Migration entries for new columns
- `backend/mlx_manager/routers/chat.py` - Generation parameters from profile with request override
- `backend/mlx_manager/routers/profiles.py` - Removed port-related logic
- `backend/tests/conftest.py` - Updated fixtures with new fields
- `backend/tests/test_profiles.py` - Rewrote all 21 profile tests

### Frontend
- `frontend/src/lib/api/types.ts` - Updated interfaces
- `frontend/src/lib/api/client.ts` - Removed getNextPort, updated start() return type
- `frontend/src/lib/components/profiles/ProfileForm.svelte` - Complete rewrite with generation settings
- `frontend/src/lib/components/servers/ServerTile.svelte` - Shows PID instead of port
- `frontend/src/lib/stores/profiles.svelte.ts` - Updated profilesEqual, removed port logic
- `frontend/src/lib/stores/servers.svelte.ts` - Updated serversEqual
- `frontend/src/routes/(protected)/profiles/new/+page.svelte` - Removed nextPort logic

### Test Files Updated (12)
- `frontend/src/lib/api/client.test.ts`
- `frontend/src/lib/components/profiles/ProfileCard.test.ts`
- `frontend/src/lib/components/profiles/ProfileForm.test.ts`
- `frontend/src/lib/components/servers/ProfileSelector.test.ts`
- `frontend/src/lib/components/servers/ServerCard.test.ts`
- `frontend/src/lib/components/servers/ServerTile.test.ts`
- `frontend/src/lib/components/servers/StartingTile.test.ts`
- `frontend/src/lib/stores/profiles.svelte.test.ts`
- `frontend/src/lib/stores/servers.svelte.test.ts`

## Decisions Made

1. **Generation parameter cascade pattern**: Request parameters override profile defaults, which override system defaults (0.7/4096/1.0). This allows per-profile customization while still supporting request-level flexibility.

2. **ServerTile shows PID instead of port**: With embedded server, there's no per-profile port. PID is more useful for debugging.

3. **Database migration adds columns with defaults**: New columns get default values so existing profiles work without modification.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

1. **Frontend Input component doesn't support min/max props**: The custom Input component from bits-ui didn't support min/max HTML attributes needed for max_tokens validation. Fixed by using native `<input>` element for that field.

2. **Extensive test file updates**: 12 test files needed mock data updates to include the new generation parameters. All mock profile helpers updated with temperature, max_tokens, top_p fields.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Profile model is now clean and focused on what matters for the embedded MLX server
- Generation parameters are ready to use in all chat requests
- All 21 backend profile tests pass
- All frontend type checks pass (svelte-check: 0 errors)

---
*Phase: 15-code-cleanup-integration-tests*
*Completed: 2026-02-04*
