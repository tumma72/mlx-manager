---
phase: "adhoc"
plan: "P3-1"
subsystem: "mlx-server-openapi"
tags: ["mlx-server", "openapi", "documentation", "schemas", "fastapi"]
depends_on: []
provides: ["openapi-request-examples", "openapi-response-descriptions", "openapi-audio-speech-responses", "openapi-export-query-descriptions"]
affects: ["api-consumers", "openapi-spec", "swagger-ui"]
tech-stack:
  added: []
  patterns: ["model_config ConfigDict json_schema_extra for Pydantic OpenAPI examples", "FastAPI responses={200: {description}} for streaming endpoint docs"]
key-files:
  created: []
  modified:
    - backend/mlx_manager/mlx_server/schemas/openai.py
    - backend/mlx_manager/mlx_server/schemas/anthropic.py
    - backend/mlx_manager/mlx_server/api/v1/chat.py
    - backend/mlx_manager/mlx_server/api/v1/completions.py
    - backend/mlx_manager/mlx_server/api/v1/messages.py
    - backend/mlx_manager/mlx_server/api/v1/speech.py
    - backend/mlx_manager/mlx_server/api/v1/admin.py
decisions:
  - "Use model_config = ConfigDict(json_schema_extra={'examples': [...]}) at class level — not per-field — for concise full-request examples"
  - "ConfigDict imported alongside BaseModel and Field to keep import changes minimal"
  - "200 description on streaming endpoints describes both JSON and SSE variants in one sentence to guide API consumers"
  - "audio/speech gets 422/404/500 error schemas matching other inference endpoints for consistency"
  - "export_audit_logs Query() params all got descriptions; authorization header param left undescribed (FastAPI-generated, not user-facing)"
metrics:
  duration: "~7 min"
  completed: "2026-03-04"
---

# Phase Adhoc Plan P3-1: OpenAPI Spec Enrichment Summary

**One-liner:** Added `model_config` request examples to four Pydantic schemas and `responses={}` blocks to four endpoints, making the MLX Server OpenAPI spec self-documenting for API consumers.

## What Was Built

Purely additive enrichment of the OpenAPI spec — no logic changes, no new endpoints, no behavioral changes.

### Request Schema Examples Added

**`ChatCompletionRequest`** (`schemas/openai.py`):
- Full example: Qwen3-0.6B model, single user message, temperature=0.7, max_tokens=256, stream=false

**`CompletionRequest`** (`schemas/openai.py`):
- Full example: Qwen3-0.6B model, simple prompt, max_tokens=64, temperature=1.0

**`EmbeddingRequest`** (`schemas/openai.py`):
- Full example: all-MiniLM-L6-v2-4bit model, single input string

**`AnthropicMessagesRequest`** (`schemas/anthropic.py`):
- Full example: Qwen3-0.6B model, user message, system prompt, max_tokens=256, temperature=0.7

All examples use real mlx-community model names for accuracy.

### Response Descriptions Added to Streaming Endpoints

**`POST /chat/completions`** (`api/v1/chat.py`):
- 200: describes both JSON (ChatCompletionResponse) and SSE (ChatCompletionChunk events + [DONE])

**`POST /completions`** (`api/v1/completions.py`):
- 200: describes both JSON (CompletionResponse) and SSE completion chunks + [DONE]

**`POST /messages`** (`api/v1/messages.py`):
- 200: describes both JSON (AnthropicMessagesResponse) and full Anthropic SSE event sequence

### Audio Speech Endpoint `responses={}` Block Added

**`POST /audio/speech`** (`api/v1/speech.py`):
- 200: "Audio data in the requested format (WAV, FLAC, or MP3 bytes)"
- 422: ProblemDetail, "Validation Error"
- 404: ProblemDetail, "Model Not Found"
- 500: ProblemDetail, "Internal Server Error"
- Also imported `ProblemDetail` from `mlx_manager.mlx_server.errors`

### Query Parameter Descriptions Added

**`GET /admin/audit-logs/export`** (`api/v1/admin.py`):
- `model`: "Filter by model name"
- `backend_type`: "Filter by backend type (local, openai, anthropic)"
- `status`: "Filter by status (success, error, timeout)"
- `start_time`: "Start of time range (ISO 8601)"
- `end_time`: "End of time range (ISO 8601)"

## Verification

- OpenAPI spec generates without error (`create_app().openapi()` succeeds)
- All four schemas show `examples=True` in generated spec components
- All three streaming endpoints have 200 descriptions in generated spec
- `/audio/speech` shows responses `['200', '404', '422', '500']` in generated spec
- All 1966 existing `tests/mlx_server/` tests continue to pass

## Deviations from Plan

None — plan executed exactly as written.

## Next Phase Readiness

This change is purely additive to the OpenAPI spec. No breaking changes, no new dependencies. Safe to proceed with any subsequent P3-x tasks.
