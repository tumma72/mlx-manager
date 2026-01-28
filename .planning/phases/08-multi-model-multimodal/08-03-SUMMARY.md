---
phase: 08-multi-model-multimodal
plan: 03
subsystem: api
tags: [mlx-vlm, mlx-embeddings, vision, multimodal, image-processing, pil]

# Dependency graph
requires:
  - phase: 08-01
    provides: ModelPoolManager with LRU eviction
provides:
  - Model type detection (text-gen/vision/embeddings) from config.json and name patterns
  - Vision model loading via mlx-vlm
  - Embeddings model loading via mlx-embeddings
  - Image preprocessing (base64, URL, file) with auto-resize
  - OpenAI-compatible vision content blocks
affects: [08-04, 08-05, phase-9]

# Tech tracking
tech-stack:
  added: [mlx-vlm, mlx-embeddings, PIL/Pillow]
  patterns: [model-type-detection, content-block-union, image-preprocessing]

key-files:
  created:
    - backend/mlx_manager/mlx_server/models/detection.py
    - backend/mlx_manager/mlx_server/services/image_processor.py
  modified:
    - backend/mlx_manager/mlx_server/models/pool.py
    - backend/mlx_manager/mlx_server/schemas/openai.py

key-decisions:
  - "Config-first detection: vision_config, image_token_id, architectures before name patterns"
  - "Store processor as tokenizer field: vision models return (model, processor), reusing LoadedModel structure"
  - "Use Any type for PIL Images: avoids ImageFile vs Image type mismatches"
  - "Resampling.LANCZOS: Pillow 10+ API instead of deprecated Image.LANCZOS"

patterns-established:
  - "Model type detection chain: config fields -> model_type -> architectures -> name patterns -> default"
  - "Conditional loader imports: mlx-lm, mlx-vlm, mlx-embeddings based on detected type"
  - "Content block union: str | list[TextContentBlock | ImageContentBlock] for multimodal messages"
  - "Image preprocessing pipeline: decode/fetch -> RGB convert -> resize"

# Metrics
duration: 4min
completed: 2026-01-28
---

# Phase 8 Plan 3: Model Type Detection and Vision Infrastructure Summary

**Model type detection from config.json with vision/embeddings loading via mlx-vlm and mlx-embeddings, plus image preprocessing service for base64/URL inputs**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-28T11:44:49Z
- **Completed:** 2026-01-28T11:49:00Z
- **Tasks:** 4 + 1 fix
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments

- Model type detection using config.json fields (vision_config, image_token_id, architectures) with name pattern fallback
- Pool manager routes loading to mlx-vlm (vision), mlx-embeddings (embeddings), or mlx-lm (text-gen)
- Image preprocessing service handles base64 data URIs, HTTP URLs, and local files with auto-resize
- OpenAI schemas extended with vision content blocks (ImageContentBlock, TextContentBlock, extract_content_parts)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create model type detection module** - `7dfd663` (feat)
2. **Task 2: Update pool.py to load vision/embeddings models** - `8e9e935` (feat)
3. **Task 3: Create image preprocessing service** - `ce5da38` (feat)
4. **Task 4: Add vision content blocks to OpenAI schemas** - `9c378d9` (feat)
5. **Type/lint fixes** - `91a16e2` (fix)

## Files Created/Modified

- `backend/mlx_manager/mlx_server/models/detection.py` - Model type detection from config.json and name patterns
- `backend/mlx_manager/mlx_server/services/image_processor.py` - Image preprocessing (base64, URL, file, resize)
- `backend/mlx_manager/mlx_server/models/pool.py` - Multi-loader support (mlx-lm, mlx-vlm, mlx-embeddings)
- `backend/mlx_manager/mlx_server/schemas/openai.py` - Vision content blocks and extract helper

## Decisions Made

1. **Config-first detection chain**: Check vision_config, image_token_id first (most reliable), then model_type field, then architectures list, then name patterns, finally default to text-gen
2. **Processor stored as tokenizer**: Vision models return (model, processor) - we store processor in the tokenizer field of LoadedModel to reuse existing structure
3. **Any type for PIL Images**: Using Any return type avoids type mismatches between PIL.Image.Image and PIL.ImageFile.ImageFile
4. **Pillow 10+ API**: Use Resampling.LANCZOS instead of deprecated Image.LANCZOS

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Type checking and linting errors**
- **Found during:** Final verification
- **Issue:** Pillow type mismatches, deprecated LANCZOS attribute, Union vs pipe syntax
- **Fix:** Used Any types for PIL returns, Resampling.LANCZOS, type: ignore for untyped imports
- **Files modified:** detection.py, image_processor.py, pool.py, openai.py
- **Verification:** mypy and ruff pass
- **Committed in:** 91a16e2

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Type system fix necessary for CI. No scope creep.

## Issues Encountered

None - all tasks executed as planned.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Model type detection infrastructure ready for 08-04 (vision inference endpoint)
- Image preprocessing service ready for vision model inputs
- Pool manager can load vision and embeddings models
- OpenAI schemas support multimodal content blocks
- Ready to implement actual vision inference using mlx-vlm

---
*Phase: 08-multi-model-multimodal*
*Completed: 2026-01-28*
