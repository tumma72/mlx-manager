---
phase: 15-code-cleanup-integration-tests
plan: 06
subsystem: mlx-server
tags: [mlx, vision, model-detection, gemma, multimodal]

# Dependency graph
requires:
  - phase: 15-05
    provides: Model unload functionality for reload capability
provides:
  - Gemma 3 vision model detection
  - Synchronized detection logic (badge display = model loading)
  - Model type mismatch error handling
  - Model reload with type override
affects: [model-loading, vision-inference, frontend-badges]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Config-first model type detection with image_token_index support
    - Shared detection logic between badge display and model loading

key-files:
  created: []
  modified:
    - backend/mlx_manager/mlx_server/models/detection.py
    - backend/mlx_manager/utils/model_detection.py
    - backend/mlx_manager/mlx_server/services/vision.py
    - backend/mlx_manager/mlx_server/models/pool.py
    - backend/tests/test_model_detection.py

key-decisions:
  - "image_token_index detection: Gemma 3 uses image_token_index instead of image_token_id"
  - "Shared detect_multimodal(): MLX server detection calls shared utility for consistency"
  - "Model type mismatch error: Clear message in vision.py guides user to unload/reload"

patterns-established:
  - "Single source of truth for multimodal detection via detect_multimodal()"

# Metrics
duration: 3min
completed: 2026-02-03
---

# Phase 15 Plan 06: Fix Vision Model Detection & Loading Summary

**Gemma 3 vision detection via image_token_index, synchronized detection logic between badge display and model loading, graceful error handling for type mismatch**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-03T12:52:31Z
- **Completed:** 2026-02-03T12:55:46Z
- **Tasks:** 5
- **Files modified:** 5

## Accomplishments
- Gemma 3 multimodal models now correctly detected as VISION type
- Badge display and model loading use identical detection logic (single source of truth)
- Clear error message when vision request hits non-vision model
- Model reload capability with explicit type override

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Gemma to vision name patterns** - `2477033` (feat)
2. **Task 2: Improve config loading reliability** - `e054405` (fix)
3. **Task 3: Add graceful handling for detection/loading mismatch** - `2010edc` (fix)
4. **Task 4: Synchronize detection logic** - `1beb535` (refactor)
5. **Task 5: Add model reload capability** - `903e4f7` (feat)

## Files Created/Modified
- `backend/mlx_manager/mlx_server/models/detection.py` - Model type detection with shared multimodal detection
- `backend/mlx_manager/utils/model_detection.py` - Added image_token_index detection for Gemma 3
- `backend/mlx_manager/mlx_server/services/vision.py` - Model type mismatch error handling
- `backend/mlx_manager/mlx_server/models/pool.py` - reload_as_type() and _load_model_as_type() methods
- `backend/tests/test_model_detection.py` - Test for image_token_index detection

## Decisions Made
- **image_token_index detection:** Gemma 3 uses `image_token_index` instead of `image_token_id` in config.json. Added to both detection modules.
- **Shared detection logic:** MLX server detection now calls `detect_multimodal()` from utils/model_detection.py instead of duplicating logic. Ensures badge display and model loading are always consistent.
- **Error message guidance:** When vision request hits non-vision model, error message guides user to unload and reload or use a vision-capable model.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added image_token_index to multimodal detection**
- **Found during:** Task 2 (Improve config loading reliability)
- **Issue:** Gemma 3 config uses `image_token_index` instead of `image_token_id`, causing detection to fall through to name patterns
- **Fix:** Added `image_token_index` to vision detection keys in both detection modules
- **Files modified:** detection.py, model_detection.py
- **Verification:** Gemma 3 now detected as VISION from config (not name pattern)
- **Committed in:** e054405 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix was necessary for reliable Gemma 3 detection. No scope creep.

## Issues Encountered
None - plan executed smoothly after discovering Gemma 3's config structure.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Vision model detection now reliable for Gemma 3 and other multimodal models
- Badge display matches server-side detection
- Ready for final UAT verification or v1.2 release

---
*Phase: 15-code-cleanup-integration-tests*
*Completed: 2026-02-03*
