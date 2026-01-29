---
phase: 10-dual-protocol-cloud-fallback
plan: 01
subsystem: api
tags: [anthropic, pydantic, schemas, messages-api, streaming]

# Dependency graph
requires:
  - phase: 09-continuous-batching
    provides: inference engine and MLX server foundation
provides:
  - Anthropic Messages API request/response schemas
  - Streaming event schemas for SSE
  - Content block types (TextBlockParam, ImageBlockParam)
  - Content extraction helper function
affects: [10-02 protocol-translator, 10-03 anthropic-endpoint]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Pydantic v2 discriminated unions for content blocks
    - Literal types for Anthropic-specific enums

key-files:
  created:
    - backend/mlx_manager/mlx_server/schemas/anthropic.py
    - backend/tests/mlx_server/schemas/test_anthropic.py
  modified: []

key-decisions:
  - "max_tokens required field (no default) - matches Anthropic API requirement"
  - "System message stored separately from messages array - Anthropic pattern"
  - "Temperature bounds 0.0-1.0 (stricter than OpenAI's 0.0-2.0)"

patterns-established:
  - "Content block union: str | list[TextBlockParam | ImageBlockParam]"
  - "Streaming event sequence: message_start -> content_block_start -> deltas -> content_block_stop -> message_delta -> message_stop"

# Metrics
duration: 2.5min
completed: 2026-01-29
---

# Phase 10 Plan 01: Anthropic Schemas Summary

**Pydantic v2 schemas for Anthropic Messages API with required max_tokens, separate system field, content blocks, and streaming events**

## Performance

- **Duration:** 2.5 min
- **Started:** 2026-01-29T15:01:42Z
- **Completed:** 2026-01-29T15:04:08Z
- **Tasks:** 2
- **Files created:** 3

## Accomplishments
- Complete Anthropic Messages API request schema with required max_tokens validation
- Response schema with proper stop_reason enum (end_turn, max_tokens, stop_sequence, tool_use)
- All 6 streaming event types for SSE (MessageStart, ContentBlockStart, ContentBlockDelta, ContentBlockStop, MessageDelta, MessageStop)
- 31 comprehensive tests covering all schema components

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Anthropic Messages API schemas** - `f2b0c0d` (feat)
2. **Task 2: Add schema unit tests** - `fbaf1a3` (test)

## Files Created/Modified
- `backend/mlx_manager/mlx_server/schemas/anthropic.py` - Anthropic request/response/streaming schemas (226 lines)
- `backend/tests/mlx_server/schemas/__init__.py` - Test package init
- `backend/tests/mlx_server/schemas/test_anthropic.py` - Schema validation tests (405 lines, 31 tests)

## Decisions Made
- **max_tokens required:** Unlike OpenAI's optional max_tokens, Anthropic requires it - enforced with `Field(ge=1)` and no default
- **System message separate:** Stored in `system` field not in `messages` array - matches Anthropic API spec
- **Temperature bounds 0.0-1.0:** Anthropic uses stricter bounds than OpenAI (which allows up to 2.0)
- **Content block union types:** Support both string and list of content blocks for flexibility

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed without issues.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Anthropic schemas ready for protocol translator (Plan 02)
- All exports available: AnthropicMessagesRequest, AnthropicMessagesResponse, MessageParam, ContentBlock
- Helper function extract_anthropic_content available for content normalization

---
*Phase: 10-dual-protocol-cloud-fallback*
*Completed: 2026-01-29*
