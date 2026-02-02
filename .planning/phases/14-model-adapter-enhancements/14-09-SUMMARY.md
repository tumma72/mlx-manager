---
phase: 14-model-adapter-enhancements
plan: 09
subsystem: api
tags: [providers, openai, anthropic, together, groq, fireworks, mistral, deepseek, cloud-routing]

# Dependency graph
requires:
  - phase: 10-cloud-fallback
    provides: BackendType enum, CloudCredential model, BackendRouter
  - phase: 11-configuration-ui
    provides: ProviderForm component, settings API
provides:
  - ApiType enum for protocol selection (openai, anthropic)
  - Extended BackendType with common providers (together, groq, fireworks, mistral, deepseek)
  - Generic openai_compatible and anthropic_compatible backend types
  - DEFAULT_BASE_URLS mapping for known providers
  - API_TYPE_FOR_BACKEND mapping for automatic protocol detection
  - Updated CloudCredential with api_type and name fields
  - Multiple credentials per backend type support
  - Provider defaults API endpoint
affects: [cloud-routing, settings-ui, any-provider-configuration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "API type selection based on credential.api_type field"
    - "Cache by credential ID (not backend_type) for multiple providers"
    - "Backwards compatibility via API_TYPE_FOR_BACKEND mapping"

key-files:
  created: []
  modified:
    - "backend/mlx_manager/models.py"
    - "backend/mlx_manager/mlx_server/services/cloud/router.py"
    - "backend/mlx_manager/routers/settings.py"
    - "frontend/src/lib/api/types.ts"
    - "frontend/src/lib/components/settings/ProviderForm.svelte"
    - "frontend/src/lib/components/settings/ProviderSection.svelte"
    - "backend/tests/mlx_server/services/cloud/test_router.py"

key-decisions:
  - "ApiType enum with openai/anthropic values for explicit protocol selection"
  - "Cache by credential ID instead of backend_type to support multiple providers"
  - "Backwards compatibility: credentials without api_type use API_TYPE_FOR_BACKEND mapping"
  - "Don't pre-populate base_url in UI - let placeholder show default, send undefined to use server default"

patterns-established:
  - "API type selection: credential.api_type determines OpenAI vs Anthropic client"
  - "Provider defaults: backend has DEFAULT_BASE_URLS and API_TYPE_FOR_BACKEND mappings"

# Metrics
duration: 13min
completed: 2026-02-02
---

# Phase 14 Plan 09: Generic Provider Support Summary

**ApiType enum and extended BackendType for generic OpenAI/Anthropic-compatible providers with auto-fill defaults for Together, Groq, Fireworks, Mistral, and DeepSeek**

## Performance

- **Duration:** 13 min
- **Started:** 2026-02-02T13:48:09Z
- **Completed:** 2026-02-02T14:02:01Z
- **Tasks:** 5 (+ 1 bug fix)
- **Files modified:** 9

## Accomplishments
- Added ApiType enum for explicit protocol selection (openai vs anthropic)
- Extended BackendType with 7 new values (together, groq, fireworks, mistral, deepseek, openai_compatible, anthropic_compatible)
- Updated CloudCredential with api_type and name fields, removed unique constraint on backend_type
- Added DEFAULT_BASE_URLS and API_TYPE_FOR_BACKEND mappings for automatic configuration
- Updated backend router to select client based on api_type, cache by credential ID
- Added /providers/defaults endpoint for UI auto-fill
- Updated frontend with all new providers and auto-fill functionality
- Added 17 new tests for generic provider functionality

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend Data Models** - `3ee37ea` (feat)
2. **Task 2: Update Backend Router** - `4195e41` (feat)
3. **Task 3: Update Settings API** - `8ac637e` (feat)
4. **Task 4: Update Frontend Provider Form** - `76d3226` (feat)
5. **Task 5: Add Tests** - `ccd5166` (test)
6. **Bug Fix: ProviderForm base_url handling** - `8a47273` (fix)

## Files Created/Modified
- `backend/mlx_manager/models.py` - Added ApiType enum, extended BackendType, added DEFAULT_BASE_URLS and API_TYPE_FOR_BACKEND mappings
- `backend/mlx_manager/mlx_server/services/cloud/router.py` - Updated to use api_type for client selection, cache by credential ID
- `backend/mlx_manager/routers/settings.py` - Added /providers/defaults endpoint, updated create/test endpoints
- `frontend/src/lib/api/types.ts` - Added ApiType, extended BackendType, updated CloudCredential interface
- `frontend/src/lib/components/settings/ProviderForm.svelte` - Added provider configs, auto-fill functionality
- `frontend/src/lib/components/settings/ProviderSection.svelte` - Added all new providers to UI
- `frontend/src/lib/components/settings/ProviderForm.test.ts` - Updated test expectations for new fields
- `frontend/src/lib/stores/settings.svelte.test.ts` - Updated mock credential helper
- `backend/tests/mlx_server/services/cloud/test_router.py` - Added 17 new tests for generic provider support

## Decisions Made
- **ApiType enum**: Created separate enum for protocol type (openai/anthropic) rather than inferring from BackendType, making the relationship explicit
- **Cache by credential ID**: Changed cloud backend cache key from BackendType to credential ID to support multiple providers of the same API type
- **Backwards compatibility**: Credentials without api_type fall back to API_TYPE_FOR_BACKEND mapping for seamless migration
- **Don't pre-populate base_url**: Let placeholder show default URL, send undefined to use server-side default, avoiding URL concatenation issues

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ProviderForm base_url concatenation**
- **Found during:** Task 6 (Verification - frontend tests)
- **Issue:** Form pre-populated base_url with provider default, causing user input to be concatenated
- **Fix:** Don't pre-populate base_url - let placeholder show default, send undefined to use server default
- **Files modified:** frontend/src/lib/components/settings/ProviderForm.svelte
- **Verification:** All 971 frontend tests pass
- **Committed in:** 8a47273

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix essential for correct URL handling. No scope creep.

## Issues Encountered
- Pre-existing mypy errors in other files (not from this plan) - 4 type errors in chat.py, system.py, settings.py unrelated to this plan's changes
- Pre-existing formatting issues in 30 files from earlier work (not this plan)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Generic provider support complete - users can now configure Together, Groq, Fireworks, Mistral, DeepSeek, or any custom OpenAI/Anthropic-compatible API
- Routing rules can target any configured provider
- Cloud fallback system works with new providers
- Database migration may be needed to add api_type and name columns to existing cloud_credentials table

---
*Phase: 14-model-adapter-enhancements*
*Completed: 2026-02-02*
