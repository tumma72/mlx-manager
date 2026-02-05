---
phase: 15-code-cleanup-integration-tests
plan: 13
subsystem: audio
tags: [mlx-audio, tts, stt, kokoro, whisper, audio, openai-api]

# Dependency graph
requires:
  - phase: 15-10
    provides: "E2E pytest marker infrastructure and conftest fixtures"
  - phase: 15-12
    provides: "Embeddings model type in Profile UI and E2E patterns"
provides:
  - "ModelType.AUDIO enum value with comprehensive detection"
  - "Audio model loading in ModelPoolManager via mlx-audio"
  - "TTS service (generate_speech) and STT service (transcribe_audio)"
  - "POST /v1/audio/speech endpoint (OpenAI-compatible)"
  - "POST /v1/audio/transcriptions endpoint (OpenAI-compatible)"
  - "22 unit tests for audio schemas, endpoints, and services"
  - "5 E2E tests for audio TTS generation (pytest -m e2e_audio)"
  - "Audio model type in Profile UI dropdown"
affects: ["15-UAT"]

# Tech tracking
tech-stack:
  added:
    - "mlx-audio (>=0.3.1) integration for TTS and STT inference"
  patterns:
    - "mlx_audio.utils.load_model for auto-detecting TTS vs STT models"
    - "model.generate() returning iterable of GenerationResult with audio data"
    - "mlx_audio.stt.generate.generate_transcription for STT with file path input"
    - "Queue-based threading for MLX Metal thread affinity (consistent with inference.py and embeddings.py)"
    - "BytesIO buffer with audio_write for in-memory WAV/FLAC/MP3 encoding"

key-files:
  created:
    - "backend/mlx_manager/mlx_server/services/audio.py"
    - "backend/mlx_manager/mlx_server/api/v1/speech.py"
    - "backend/mlx_manager/mlx_server/api/v1/transcriptions.py"
    - "backend/tests/mlx_server/test_audio.py"
    - "backend/tests/mlx_server/models/__init__.py"
    - "backend/tests/mlx_server/models/test_detection_audio.py"
    - "backend/tests/e2e/test_audio_e2e.py"
  modified:
    - "backend/mlx_manager/mlx_server/models/types.py"
    - "backend/mlx_manager/mlx_server/models/detection.py"
    - "backend/mlx_manager/mlx_server/models/pool.py"
    - "backend/mlx_manager/mlx_server/schemas/openai.py"
    - "backend/mlx_manager/mlx_server/api/v1/__init__.py"
    - "backend/tests/e2e/conftest.py"
    - "frontend/src/lib/components/profiles/ProfileForm.svelte"

key-decisions:
  - "Audio models have tokenizer=None in LoadedModel since they don't use text tokenization"
  - "TTS generates via model.generate() which returns GenerationResult iterable with audio mx.array"
  - "STT writes audio to temp file since generate_transcription expects file path, not bytes"
  - "SpeechRequest.input max 4096 chars, speed 0.25-4.0 range per OpenAI spec"
  - "WAV is default audio format (no ffmpeg dependency required unlike MP3)"

patterns-established:
  - "Audio detection: config fields (audio_config, tts_config, stt_config, vocoder_config, codec_config)"
  - "Audio detection: architecture names and model_type values from mlx-audio remapping dicts"
  - "Audio detection: name patterns (kokoro, whisper, tts, stt, speech, bark, etc.)"

# Metrics
duration: 8min
completed: 2026-02-05
---

# Phase 15 Plan 13: Audio Integration Summary

Full audio model support (TTS + STT) via mlx-audio library with model detection, pool loading, inference services, API endpoints, and E2E tests.

## What Was Done

### Task 1: AUDIO Model Type and Detection (89c2455)
- Added `ModelType.AUDIO = "audio"` to the enum
- Config-based detection: 5 config field indicators (audio_config, tts_config, stt_config, vocoder_config, codec_config)
- Architecture-based detection: 16 audio architecture patterns
- model_type field detection: 14 known model type values from mlx-audio
- Name-based fallback: 17 audio name patterns
- 58 parametrized detection tests covering all detection paths and negative cases

### Task 2: Audio Model Loading in Pool (ccc0c23)
- Added AUDIO branch in `_load_model()` and `_load_model_as_type()`
- Uses `mlx_audio.utils.load_model()` which auto-detects TTS vs STT
- Audio models stored with `tokenizer=None`
- All existing pool tests pass unchanged

### Task 3: Audio Inference Service (fdad492)
- `generate_speech()`: TTS via model.generate() returning GenerationResult objects
  - Concatenates multi-segment audio, writes to BytesIO via audio_write
  - Supports wav/flac/mp3 output formats
- `transcribe_audio()`: STT via generate_transcription()
  - Writes uploaded bytes to temp file (library expects file path)
  - Returns dict with text, segments, language
- Both use queue-based threading for MLX Metal thread affinity
- LogFire span instrumentation for observability

### Task 4: TTS and STT API Endpoints (5251c47)
- `SpeechRequest` schema: model, input (1-4096 chars), voice, response_format (wav/flac/mp3), speed (0.25-4.0)
- `TranscriptionResponse` schema: text field
- POST /v1/audio/speech: returns raw audio bytes with Content-Type header
- POST /v1/audio/transcriptions: multipart form with file upload
- Both routers registered in v1_router

### Task 5: Unit Tests (df96bc5)
- Schema validation: defaults, bounds, format validation, empty input rejection
- Endpoint tests: audio bytes response, content type mapping, language hints, empty file 400
- Service signature tests: parameter verification, async checks, pool usage
- Route registration tests: both endpoints registered as POST
- 22 tests total, all passing

### Task 6: E2E Tests (4b968b1)
- Added `audio_tts_model` fixture for Kokoro-82M-4bit with graceful skip
- WAV validation: RIFF header, WAVE format, sample rate, data size
- Proportionality test: longer text produces more audio
- Speed parameter test: 2x speed produces shorter audio
- Error handling: nonexistent model, invalid speed rejected
- 5 E2E tests, all marked with @pytest.mark.e2e_audio

### Task 7: Profile UI (18b1d89)
- Added 'audio' to model type validation array
- Added 'Audio (TTS/STT)' option to Select dropdown
- Frontend type checks pass (svelte-check clean)

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

1. ModelType.AUDIO exists: PASS
2. Audio detection tests: 58 passed
3. Unit tests: 22 passed
4. Audio endpoints registered: /v1/audio/speech and /v1/audio/transcriptions confirmed
5. E2E tests collected with correct markers (skipped without model download)
6. Frontend type checks: 0 errors, 0 warnings
7. Full test suite: 1406 passed, 36 deselected (E2E)
8. Ruff lint: clean on all audio files
9. Ruff format: clean on all audio files
