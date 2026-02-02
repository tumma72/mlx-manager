---
phase: 15-code-cleanup-integration-tests
plan: 02
subsystem: database, api, models
tags: [sqlite, migration, adapter, logging, processor, tokenizer]

# Dependency graph
requires:
  - phase: 14-model-adapter-enhancements
    provides: CloudCredential model with api_type and name fields
provides:
  - Database migration for api_type and name columns in cloud_credentials
  - Robust exception handling in Qwen adapter for enable_thinking
  - DEBUG-level streaming logs instead of INFO
  - Consistent getattr pattern for processor/tokenizer in all adapters
affects: [phase-15-03, uat-resumption]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "getattr(tokenizer, 'tokenizer', tokenizer) for processor compatibility"
    - "Database migration skip for non-existent tables"

key-files:
  created: []
  modified:
    - backend/mlx_manager/database.py
    - backend/mlx_manager/mlx_server/models/adapters/base.py
    - backend/mlx_manager/mlx_server/models/adapters/gemma.py
    - backend/mlx_manager/mlx_server/models/adapters/glm4.py
    - backend/mlx_manager/mlx_server/models/adapters/llama.py
    - backend/mlx_manager/mlx_server/models/adapters/mistral.py
    - backend/mlx_manager/mlx_server/models/adapters/qwen.py
    - backend/mlx_manager/routers/chat.py
    - backend/tests/mlx_server/test_adapters.py

key-decisions:
  - "api_type default 'openai' for backward compatibility with existing credentials"
  - "Skip migration for non-existent tables to avoid errors on fresh database"
  - "Catch TypeError, ValueError, KeyError, AttributeError for enable_thinking fallback"
  - "All adapters use getattr pattern for processor/tokenizer compatibility"

patterns-established:
  - "Database migration skips tables that don't exist yet (fresh DBs)"
  - "Adapter apply_chat_template uses getattr(tokenizer, 'tokenizer', tokenizer)"
  - "Adapter get_stop_tokens uses getattr(tokenizer, 'tokenizer', tokenizer)"
  - "MagicMock spec=[] in tests to prevent auto-creation of tokenizer attribute"

# Metrics
duration: 5min
completed: 2026-02-02
---

# Phase 15 Plan 02: Fix Blocker Bugs Summary

**Database migration for cloud_credentials columns, robust Qwen enable_thinking handling, DEBUG-level streaming logs, and consistent processor/tokenizer getattr pattern across all adapters**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-02T17:22:26Z
- **Completed:** 2026-02-02T17:27:35Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments
- Database migration now adds api_type and name columns to cloud_credentials table with backward-compatible defaults
- Qwen adapter catches all exception types for enable_thinking parameter rejection (TypeError, ValueError, KeyError, AttributeError)
- Streaming content preview logging changed from INFO to DEBUG level
- All model adapters (base, gemma, glm4, llama, mistral, qwen) use consistent getattr pattern for processor/tokenizer compatibility

## Task Commits

Each task was committed atomically:

1. **Task 1: Add database migration for CloudCredential columns** - `3c29cf6` (fix)
2. **Task 2: Fix Qwen adapter exception handling and logging levels** - `6e7f07b` (fix)
3. **Task 3: Fix vision processor attribute access** - `3644d6f` (test)

Note: Task 3's adapter changes were merged with 15-01's refactor commit `3f7dbd8`. The test updates were committed separately.

## Files Created/Modified
- `backend/mlx_manager/database.py` - Added cloud_credentials migrations with default values
- `backend/mlx_manager/mlx_server/models/adapters/base.py` - Added getattr pattern to DefaultAdapter
- `backend/mlx_manager/mlx_server/models/adapters/gemma.py` - Added getattr pattern to apply_chat_template
- `backend/mlx_manager/mlx_server/models/adapters/glm4.py` - Added getattr pattern to apply_chat_template
- `backend/mlx_manager/mlx_server/models/adapters/llama.py` - Added getattr pattern to apply_chat_template
- `backend/mlx_manager/mlx_server/models/adapters/mistral.py` - Added getattr pattern to apply_chat_template
- `backend/mlx_manager/mlx_server/models/adapters/qwen.py` - Expanded exception handling, changed log level to debug
- `backend/mlx_manager/routers/chat.py` - Changed streaming content log from INFO to DEBUG
- `backend/tests/mlx_server/test_adapters.py` - Fixed tests to use spec=[] for MagicMock

## Decisions Made
- **api_type default 'openai':** Ensures existing credentials continue to work without manual migration
- **name default '':** Empty string allows credentials to function without requiring a name
- **Migration skips non-existent tables:** Prevents errors on fresh databases where CREATE TABLE handles columns
- **Catch multiple exception types:** enable_thinking can fail with TypeError (unexpected kwarg), ValueError (invalid value), KeyError (template lookup), or AttributeError (missing method)
- **DEBUG not WARNING for enable_thinking fallback:** This is expected behavior for older tokenizers, not a warning condition

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed migration for non-existent tables**
- **Found during:** Task 1 (Database migration)
- **Issue:** migrate_schema() crashed on fresh database when trying to ALTER TABLE that doesn't exist
- **Fix:** Added check to skip migration if PRAGMA table_info returns no columns
- **Files modified:** backend/mlx_manager/database.py
- **Verification:** Migration works on both fresh and existing databases
- **Committed in:** 3c29cf6

**2. [Rule 3 - Blocking] Extended getattr pattern to all adapters**
- **Found during:** Task 3 (Vision processor fix)
- **Issue:** Not just vision.py but all adapters needed the getattr pattern for apply_chat_template
- **Fix:** Updated base.py, gemma.py, glm4.py, llama.py, mistral.py with getattr pattern
- **Files modified:** All adapter files
- **Verification:** All 20 adapter tests pass, 1248 total tests pass
- **Committed in:** Part of 3f7dbd8 (15-01) and test fixes in 3644d6f

---

**Total deviations:** 2 auto-fixed (both blocking)
**Impact on plan:** Both fixes necessary for correctness. Extended scope to all adapters for consistency.

## Issues Encountered
- Test failures after adapter changes: MagicMock auto-creates .tokenizer attribute when accessed. Fixed by using spec=[] to prevent auto-creation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All blocker bugs from UAT fixed
- Ready for Plan 15-03 (Integration tests for ResponseProcessor)
- UAT can resume after Plan 15-03 completes

---
*Phase: 15-code-cleanup-integration-tests*
*Completed: 2026-02-02*
