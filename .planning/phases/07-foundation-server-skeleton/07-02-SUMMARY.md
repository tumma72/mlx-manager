---
phase: 07-foundation-server-skeleton
plan: 02
subsystem: api
tags: [pydantic, openai-api, fastapi, schemas, mlx-server]

# Dependency graph
requires:
  - phase: 07-01
    provides: MLX server skeleton (main.py, config.py)
provides:
  - OpenAI-compatible Pydantic v2 request/response schemas
  - /v1/models endpoint returning hot + loadable models
  - MLX server config with available_models setting
affects: [07-03, 07-04, 07-05, 07-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Pydantic v2 Field constraints for validation
    - FastAPI APIRouter with prefix and tags
    - lru_cache for settings singleton

key-files:
  created:
    - backend/mlx_manager/mlx_server/schemas/openai.py
    - backend/mlx_manager/mlx_server/schemas/__init__.py
    - backend/mlx_manager/mlx_server/api/__init__.py
    - backend/mlx_manager/mlx_server/api/v1/__init__.py
    - backend/mlx_manager/mlx_server/api/v1/models.py
  modified:
    - backend/mlx_manager/mlx_server/config.py
    - backend/mlx_manager/mlx_server/main.py

key-decisions:
  - "available_models is a simple list of HuggingFace model IDs configured via env"
  - "Models endpoint returns both hot (loaded) and loadable (configured) models"
  - "Model IDs use HuggingFace format: org/model-name"

patterns-established:
  - "OpenAI schema pattern: Request → Response → Chunk for streaming"
  - "Router structure: api/v1/{resource}.py with router variable"
  - "get_settings() singleton with lru_cache for config access"

# Metrics
duration: 3min
completed: 2026-01-27
---

# Phase 07 Plan 02: OpenAI Schemas Summary

**OpenAI-compatible Pydantic v2 schemas with Field constraints and /v1/models endpoint returning available models**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-27T16:15:06Z
- **Completed:** 2026-01-27T16:18:01Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Created complete OpenAI API-compatible Pydantic v2 schemas (ChatCompletion, Completion, streaming)
- Implemented /v1/models endpoint that returns configured available models
- Extended MLX server config with available_models list and get_settings() pattern
- Wired v1_router into main FastAPI app

## Task Commits

Each task was committed atomically:

1. **Task 1: Create OpenAI-compatible Pydantic schemas** - `dc18011` (feat)
2. **Task 2: Create MLX server config with available models** - `a858330` (feat)
3. **Task 3: Create /v1/models endpoint with hot + loadable models** - `a139cc9` (feat)

## Files Created/Modified

- `backend/mlx_manager/mlx_server/schemas/openai.py` - Request/response models for OpenAI API
- `backend/mlx_manager/mlx_server/schemas/__init__.py` - Package exports
- `backend/mlx_manager/mlx_server/api/__init__.py` - API package marker
- `backend/mlx_manager/mlx_server/api/v1/__init__.py` - v1 router combining all endpoints
- `backend/mlx_manager/mlx_server/api/v1/models.py` - /v1/models and /v1/models/{id} endpoints
- `backend/mlx_manager/mlx_server/config.py` - Added available_models, default_model, max_cache_size_gb
- `backend/mlx_manager/mlx_server/main.py` - Added v1_router import and include_router

## Decisions Made

- **available_models as simple list:** Model IDs are stored as HuggingFace format strings (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit"). This can be extended later with a model registry.
- **get_settings() with lru_cache:** Following the FastAPI best practice for configuration singletons
- **Model path parameter with :path:** Using `{model_id:path}` to handle model IDs containing slashes

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all verifications passed on first attempt.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Schemas ready for chat completions endpoint (Plan 03)
- Config available_models ready for model pool integration
- Router structure in place for adding /v1/chat/completions and /v1/completions

---
*Phase: 07-foundation-server-skeleton*
*Plan: 02*
*Completed: 2026-01-27*
