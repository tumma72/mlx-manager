---
phase: 08-multi-model-multimodal
plan: 05
subsystem: api
tags: [embeddings, mlx-embeddings, openai-api, batch-processing]

# Dependency graph
requires:
  - phase: 08-01
    provides: Model pool with embeddings model loading
  - phase: 08-03
    provides: Model type detection for embeddings
provides:
  - /v1/embeddings endpoint with OpenAI-compatible response
  - Batch text embedding generation
  - L2-normalized vectors ready for cosine similarity
affects: [phase-09-batching, phase-10-cloud-fallback]

# Tech tracking
tech-stack:
  added: []
  patterns: [queue-based-threading-mlx, batch-tokenization]

key-files:
  created:
    - backend/mlx_manager/mlx_server/services/embeddings.py
    - backend/mlx_manager/mlx_server/api/v1/embeddings.py
    - backend/tests/mlx_server/test_embeddings.py
  modified:
    - backend/mlx_manager/mlx_server/schemas/openai.py
    - backend/mlx_manager/mlx_server/api/v1/__init__.py

key-decisions:
  - "mlx-embeddings text_embeds are already L2-normalized - no post-processing needed"
  - "Token counting uses individual encode per text (not padded batch) for accurate usage"

patterns-established:
  - "Queue-based threading for MLX Metal: same pattern as inference.py for thread affinity"

# Metrics
duration: 6min
completed: 2026-01-28
---

# Phase 8 Plan 5: Embeddings Endpoint Summary

**/v1/embeddings endpoint using mlx-embeddings with batch support and L2-normalized vectors in OpenAI-compatible format**

## Performance

- **Duration:** 6 min
- **Started:** 2026-01-28T11:50:40Z
- **Completed:** 2026-01-28T11:56:09Z
- **Tasks:** 4
- **Files modified:** 5

## Accomplishments

- OpenAI-compatible /v1/embeddings endpoint accepting single string or array
- Embeddings service with queue-based threading for MLX Metal affinity
- L2-normalized vectors ready for cosine similarity (via mlx-embeddings text_embeds)
- Model type validation returning 400 for non-embedding models

## Task Commits

Each task was committed atomically:

1. **Task 1: Add embeddings schemas** - `4af3c5b` (feat)
2. **Task 2: Create embeddings service** - `01f754d` (feat)
3. **Task 3: Create endpoint and register router** - `153cfdb` (feat)
4. **Task 4: Add unit tests** - `590a4a5` (test)

## Files Created/Modified

- `backend/mlx_manager/mlx_server/schemas/openai.py` - Added EmbeddingRequest/Response schemas
- `backend/mlx_manager/mlx_server/services/embeddings.py` - Batch embedding generation with queue threading
- `backend/mlx_manager/mlx_server/api/v1/embeddings.py` - /v1/embeddings endpoint
- `backend/mlx_manager/mlx_server/api/v1/__init__.py` - Registered embeddings_router
- `backend/tests/mlx_server/test_embeddings.py` - 9 unit tests

## Decisions Made

- **mlx-embeddings output format:** `text_embeds` attribute is already L2-normalized by the model, no additional normalization needed
- **Token counting approach:** Count tokens per text individually (not from padded batch) for accurate usage reporting

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Embeddings endpoint complete and ready for use
- Ready for Phase 9 continuous batching (can batch embedding requests)
- Ready for Phase 10 cloud fallback (can route embedding requests to cloud)

---
*Phase: 08-multi-model-multimodal*
*Completed: 2026-01-28*
