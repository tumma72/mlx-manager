---
phase: 15-code-cleanup-integration-tests
plan: 10
subsystem: testing
tags: [e2e, vision, mlx-vlm, pytest, golden-prompts]

dependency-graph:
  requires: ["15-03", "15-06"]
  provides: ["vision-e2e-tests", "e2e-pytest-infrastructure", "golden-vision-fixtures"]
  affects: ["15-11", "15-12", "15-13"]

tech-stack:
  added: []
  patterns: ["tiered-e2e-testing", "golden-prompt-fixtures", "fallback-model-resolution", "asgi-transport-with-pool-init"]

key-files:
  created:
    - backend/tests/e2e/__init__.py
    - backend/tests/e2e/conftest.py
    - backend/tests/e2e/test_vision_e2e.py
    - backend/tests/fixtures/images/generate_test_images.py
    - backend/tests/fixtures/images/red_square.png
    - backend/tests/fixtures/images/blue_circle.png
    - backend/tests/fixtures/images/text_sample.png
    - backend/tests/fixtures/golden/vision/describe_image.txt
    - backend/tests/fixtures/golden/vision/compare_images.txt
    - backend/tests/fixtures/golden/vision/ocr_text.txt
  modified:
    - backend/pyproject.toml

decisions:
  - id: fallback-model-resolution
    summary: "Prefer qat variants over DWQ for Gemma models due to VisionConfig compatibility"
    context: "gemma-3-27b-it-4bit-DWQ fails with VisionConfig.__init__() missing arguments in mlx-vlm 0.3.11"
  - id: asgi-pool-init
    summary: "Initialize model pool manually in app_client fixture since ASGITransport lacks lifespan support"
    context: "httpx ASGITransport does not trigger FastAPI lifespan handlers"
  - id: cleanup-after-each-test
    summary: "Unload all models after each test to free memory"
    context: "Vision models are 7-16GB; accumulation would exhaust memory"

metrics:
  duration: "10 min"
  completed: "2026-02-05"
---

# Phase 15 Plan 10: Vision E2E Tests Summary

**One-liner:** Tiered vision E2E test suite with golden prompts and generated test images, validated against Gemma-3 models via ASGI transport

## What Was Done

### Task 1: Configure E2E pytest marker infrastructure
- Added 7 pytest markers (e2e, e2e_vision, e2e_vision_quick, e2e_vision_full, e2e_anthropic, e2e_embeddings, e2e_audio)
- Configured `addopts = "-m 'not e2e'"` to exclude E2E from default test runs
- Created `e2e/conftest.py` with model availability checking, fallback resolution, ASGI app client with manual pool initialization

### Task 2: Create test images and golden prompts
- Generated 3 test images via PIL: red square, blue circle, "Hello MLX" text (all 256x256 PNG)
- Created 3 golden prompt fixtures: describe_image, compare_images, ocr_text
- Included `generate_test_images.py` for reproducible image generation

### Task 3: Create tiered E2E test suite
- 6 quick tier tests: describe red square, describe blue circle, compare two images, OCR text, streaming, error handling
- 4 full tier tests: accurate red square, accurate blue circle, accurate OCR, detailed comparison
- All 10 tests pass with available models
- Error handling test validates 400 rejection when sending images to text-only models

## Test Results

| Tier | Tests | Status | Model Used |
|------|-------|--------|------------|
| Quick | 6 | All pass | gemma-3-12b-it-qat-4bit (fallback) |
| Full | 4 | All pass | gemma-3-27b-it-qat-4bit (fallback) |
| Existing | 1326 | All pass | N/A (E2E excluded by default) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] ASGITransport does not trigger lifespan handlers**
- **Found during:** Task 3
- **Issue:** httpx.ASGITransport creates HTTP client but does not invoke FastAPI lifespan, so model pool was never initialized ("Model pool not initialized" error)
- **Fix:** Initialize ModelPoolManager and database manually in app_client fixture before creating client
- **Files modified:** backend/tests/e2e/conftest.py
- **Commit:** 64809b8

**2. [Rule 1 - Bug] Gemma-3-27b-it-4bit-DWQ VisionConfig incompatibility**
- **Found during:** Task 3
- **Issue:** `gemma-3-27b-it-4bit-DWQ` fails to load with `VisionConfig.__init__() missing 6 required positional arguments` in mlx-vlm 0.3.11
- **Fix:** Added fallback model lists; qat variants load successfully. Updated model priority to prefer qat over DWQ
- **Files modified:** backend/tests/e2e/conftest.py
- **Commit:** 64809b8

**3. [Rule 1 - Bug] Qwen2-VL-2B not available, Qwen3-VL incompatible**
- **Found during:** Task 3
- **Issue:** Plan's primary quick model (Qwen2-VL-2B) not downloaded; alternative Qwen3-VL-8B has NoneType iteration error
- **Fix:** Added gemma-3-12b-it-qat-4bit as quick tier fallback (smaller, compatible)
- **Files modified:** backend/tests/e2e/conftest.py
- **Commit:** 64809b8

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Fallback model resolution | Library compatibility issues require trying multiple models |
| Manual pool init in fixture | ASGITransport doesn't support lifespan; explicit init needed |
| 48GB memory limit for tests | Supports loading large vision models (Gemma-3-27b = 15.7GB) |
| Cleanup after each test | Prevents memory accumulation from 7-16GB vision models |

## Next Phase Readiness

The E2E infrastructure established here (markers, conftest patterns, app_client fixture with pool init) directly supports:
- **15-11 (Cross-protocol E2E):** Can reuse app_client fixture, add text model fixtures
- **15-12 (Embeddings E2E):** Can reuse e2e_embeddings marker, add embeddings model fixtures
- **15-13 (Audio E2E):** Can reuse e2e_audio marker, add audio model fixtures

### Known Issues
- `gemma-3-27b-it-4bit-DWQ` has VisionConfig incompatibility with mlx-vlm 0.3.11 (tracked)
- Qwen3-VL-8B has NoneType iteration error during loading (tracked)
