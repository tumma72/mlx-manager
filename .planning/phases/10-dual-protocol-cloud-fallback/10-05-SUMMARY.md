---
phase: 10-dual-protocol-cloud-fallback
plan: 05
subsystem: api
tags: [anthropic, sse, streaming, messages-api, protocol-translation]

# Dependency graph
requires:
  - phase: 10-01
    provides: Anthropic message schemas (AnthropicMessagesRequest/Response)
  - phase: 10-03
    provides: Protocol translator (anthropic_to_internal, openai_stop_to_anthropic)
  - phase: 09
    provides: generate_chat_completion inference service
provides:
  - POST /v1/messages Anthropic-compatible endpoint
  - Streaming support with Anthropic SSE event format
  - Non-streaming response with Anthropic response format
  - stop_reason translation (end_turn, max_tokens, stop_sequence)
affects: [10-08, cloud-fallback, anthropic-sdk-compatibility]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Anthropic SSE streaming with named event types"
    - "Protocol translation using cast() for type safety"

key-files:
  created:
    - backend/mlx_manager/mlx_server/api/v1/messages.py
    - backend/tests/mlx_server/api/v1/test_messages.py
    - backend/tests/mlx_server/api/__init__.py
    - backend/tests/mlx_server/api/v1/__init__.py
  modified:
    - backend/mlx_manager/mlx_server/api/v1/__init__.py

key-decisions:
  - "cast() for union return types: Use typing.cast() to handle generate_chat_completion return type (AsyncGenerator | dict) based on stream parameter"
  - "AnthropicStopReason type alias: Define Literal type for stop_reason to satisfy mypy strict checking"

patterns-established:
  - "Anthropic SSE event sequence: message_start -> content_block_start -> content_block_delta* -> content_block_stop -> message_delta -> message_stop"
  - "Protocol translation pattern: Translate Anthropic request to internal, call inference, translate response back"

# Metrics
duration: 3min
completed: 2026-01-29
---

# Phase 10 Plan 05: Anthropic Messages Endpoint Summary

**POST /v1/messages Anthropic-compatible endpoint with Anthropic SSE streaming format and stop_reason translation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-29T15:16:04Z
- **Completed:** 2026-01-29T15:19:28Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- POST /v1/messages endpoint accepting Anthropic-format request body
- Streaming response with Anthropic SSE events (message_start, content_block_delta, etc.)
- Non-streaming response with AnthropicMessagesResponse format
- stop_reason translation between OpenAI and Anthropic formats
- 18 comprehensive tests covering all functionality

## Task Commits

Each task was committed atomically:

1. **Task 1: Create /v1/messages endpoint** - `4462865` (feat)
2. **Task 2: Register messages router** - `4d9f5bd` (feat)
3. **Task 3: Add messages endpoint tests** - `be812ff` (test)

## Files Created/Modified
- `backend/mlx_manager/mlx_server/api/v1/messages.py` - Anthropic Messages API endpoint with streaming/non-streaming support
- `backend/mlx_manager/mlx_server/api/v1/__init__.py` - Added messages_router to v1 API
- `backend/tests/mlx_server/api/v1/test_messages.py` - 18 tests for messages endpoint (512 lines)
- `backend/tests/mlx_server/api/__init__.py` - Test package init
- `backend/tests/mlx_server/api/v1/__init__.py` - Test subpackage init

## Decisions Made
- **cast() for type safety:** Used typing.cast() to handle the union return type from generate_chat_completion (AsyncGenerator | dict) since mypy cannot narrow based on the stream parameter value
- **AnthropicStopReason type alias:** Defined a Literal type alias for stop_reason to satisfy mypy strict checking when assigning the translated stop reason

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Anthropic Messages API endpoint complete and registered
- Applications using Anthropic SDK can now connect to local MLX inference via /v1/messages
- Ready for cloud fallback integration (10-08) which will route requests to local or cloud based on model availability

---
*Phase: 10-dual-protocol-cloud-fallback*
*Plan: 05*
*Completed: 2026-01-29*
