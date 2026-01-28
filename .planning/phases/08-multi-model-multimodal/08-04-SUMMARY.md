---
phase: 08-multi-model-multimodal
plan: 04
subsystem: api
tags: [vision, vlm, multimodal, mlx-vlm, chat-completions, streaming]

# Dependency graph
requires:
  - phase: 08-01
    provides: Model pool with LoadedModel, queue-based threading pattern
  - phase: 08-03
    provides: Model type detection, image preprocessing, vision config loading
provides:
  - Vision generation service with streaming support
  - Chat endpoint multimodal routing
  - 400 error for non-vision models receiving images
affects: [08-admin-endpoints, 09-continuous-batching]

# Tech tracking
tech-stack:
  added: []  # mlx-vlm was added in 08-03
  patterns:
    - "Queue-based threading for vision generation"
    - "Simulated streaming for mlx-vlm (single-chunk response)"
    - "Model type detection before image processing"

key-files:
  created:
    - backend/mlx_manager/mlx_server/services/vision.py
    - backend/tests/mlx_server/test_vision.py
  modified:
    - backend/mlx_manager/mlx_server/api/v1/chat.py

key-decisions:
  - "Simulated streaming for vision: mlx-vlm generate() is non-streaming, we yield complete response as single chunk"
  - "Prompt construction: Combine system/user/assistant messages with role labels for vision models"
  - "Token estimation: ~256 tokens per image for usage statistics"

patterns-established:
  - "Vision service: Same queue-based threading pattern as inference.py"
  - "Multimodal routing: Check for images first, then detect model type"
  - "Error handling: 400 for type mismatch, 500 for generation failures"

# Metrics
duration: 7min
completed: 2026-01-28
---

# Phase 8 Plan 4: Vision Inference Endpoint Summary

**Vision model inference via chat API with multimodal request routing and 400 error for type mismatches**

## Performance

- **Duration:** 7 min
- **Started:** 2026-01-28T11:50:44Z
- **Completed:** 2026-01-28T11:57:40Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Vision generation service with streaming and non-streaming modes using mlx-vlm
- Chat endpoint automatically detects multimodal requests and routes appropriately
- Text-only models return 400 when receiving images with helpful error message
- Comprehensive test coverage for vision service and chat integration

## Task Commits

Each task was committed atomically:

1. **Task 1: Create vision generation service** - `0dd760c` (feat)
2. **Task 2: Update chat endpoint to handle vision requests** - `26cdb2e` (feat)
3. **Task 3: Add unit tests for vision service and chat integration** - `4e8d413` (test)

## Files Created/Modified

- `backend/mlx_manager/mlx_server/services/vision.py` - Vision model inference service with queue-based threading
- `backend/mlx_manager/mlx_server/api/v1/chat.py` - Extended to detect multimodal requests and route to vision service
- `backend/tests/mlx_server/test_vision.py` - 7 unit tests for vision service and chat integration

## Decisions Made

1. **Simulated streaming for vision models** - mlx-vlm's generate() function is non-streaming, so we run generation in a thread and yield the complete response as a single content chunk followed by a finish chunk. This maintains API compatibility while we investigate true token-by-token streaming in mlx-vlm internals.

2. **Prompt construction for vision** - Messages are combined with role labels (System/User/Assistant) into a single text prompt, which is then formatted by mlx-vlm's apply_chat_template with the image count.

3. **Token estimation for images** - Using ~256 tokens per image as a rough approximation for usage statistics, since exact token counts depend on vision encoder output dimensions.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- **Mock path for tests** - Initial test patches used wrong import path for `get_model_pool` since it's lazily imported inside the function. Fixed by patching `mlx_manager.mlx_server.models.pool.get_model_pool` instead.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Vision inference endpoint complete and tested
- Ready for embeddings endpoint (08-05) and admin endpoints (08-06)
- Continuous batching (Phase 9) can extend vision generation with request queuing

---
*Phase: 08-multi-model-multimodal*
*Completed: 2026-01-28*
