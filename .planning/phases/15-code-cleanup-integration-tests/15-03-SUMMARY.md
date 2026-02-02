---
phase: 15-code-cleanup-integration-tests
plan: 03
subsystem: testing
tags: [pytest, golden-files, integration-tests, response-processor, streaming]

# Dependency graph
requires:
  - phase: 14-model-adapter-enhancements
    provides: ResponseProcessor and StreamingProcessor implementations
provides:
  - Golden file test fixtures for all 6 model families
  - Parametrized integration tests for ResponseProcessor
  - Streaming processor test coverage
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Golden file testing for model output validation"
    - "Parametrized tests with directory scanning"

key-files:
  created:
    - backend/tests/fixtures/golden/qwen/tool_calls.txt
    - backend/tests/fixtures/golden/llama/tool_calls.txt
    - backend/tests/fixtures/golden/llama/python_tag.txt
    - backend/tests/fixtures/golden/glm4/tool_calls.txt
    - backend/tests/fixtures/golden/glm4/duplicate_tools.txt
    - backend/tests/fixtures/golden/hermes/tool_calls.txt
    - backend/tests/fixtures/golden/minimax/tool_calls.txt
    - backend/tests/fixtures/golden/gemma/tool_calls.txt
    - backend/tests/fixtures/golden/qwen/thinking.txt
    - backend/tests/fixtures/golden/qwen/stream/thinking_chunks.txt
    - backend/tests/fixtures/golden/qwen/stream/tool_call_chunks.txt
    - backend/tests/mlx_server/test_response_processor_golden.py
  modified: []

key-decisions:
  - "Golden files use real model output patterns for realistic testing"
  - "Parametrized tests auto-discover golden files via directory scanning"
  - "Streaming chunks use line-per-token format for cross-boundary testing"

patterns-established:
  - "Golden file directory structure: fixtures/golden/{family}/[type].txt"
  - "Streaming test format: one line per chunk in stream/*.txt files"

# Metrics
duration: 2min
completed: 2026-02-02
---

# Phase 15 Plan 03: Integration Tests for ResponseProcessor Summary

**Golden file integration tests for ResponseProcessor covering all 6 model families with 26 parametrized tests validating tool extraction, marker removal, and streaming pattern filtering**

## Performance

- **Duration:** 2 min 19 sec
- **Started:** 2026-02-02T17:30:54Z
- **Completed:** 2026-02-02T17:33:13Z
- **Tasks:** 3
- **Files created:** 12

## Accomplishments
- Created golden file directory structure with tool call files for all 6 model families (qwen, llama, glm4, hermes, minimax, gemma)
- Added thinking/reasoning golden files with streaming chunk variants
- Created comprehensive parametrized test suite with 26 tests covering tool extraction, marker removal, deduplication, and streaming

## Task Commits

Each task was committed atomically:

1. **Task 1: Create golden file directory structure and tool call golden files** - `b6c6c7f` (test)
2. **Task 2: Create thinking/streaming golden files** - `0b537e2` (test)
3. **Task 3: Create parametrized integration test file** - `15b91b8` (test)

## Files Created

### Golden File Directory Structure
- `backend/tests/fixtures/golden/qwen/tool_calls.txt` - Qwen/Hermes JSON-style tool calls
- `backend/tests/fixtures/golden/llama/tool_calls.txt` - Llama XML-style tool calls
- `backend/tests/fixtures/golden/llama/python_tag.txt` - Llama Python tag format
- `backend/tests/fixtures/golden/glm4/tool_calls.txt` - GLM4 XML-style tool calls
- `backend/tests/fixtures/golden/glm4/duplicate_tools.txt` - GLM4 duplicate tool call deduplication test
- `backend/tests/fixtures/golden/hermes/tool_calls.txt` - Hermes JSON-style tool calls
- `backend/tests/fixtures/golden/minimax/tool_calls.txt` - MiniMax JSON-style tool calls
- `backend/tests/fixtures/golden/gemma/tool_calls.txt` - Gemma JSON-style tool calls

### Thinking/Streaming Files
- `backend/tests/fixtures/golden/qwen/thinking.txt` - Complete response with thinking tags
- `backend/tests/fixtures/golden/qwen/stream/thinking_chunks.txt` - Streaming chunks with thinking tags split across lines
- `backend/tests/fixtures/golden/qwen/stream/tool_call_chunks.txt` - Streaming chunks with tool call split across lines

### Test File
- `backend/tests/mlx_server/test_response_processor_golden.py` - Parametrized integration tests

## Test Coverage

The test file includes 26 tests across 3 test classes:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestResponseProcessorToolCalls | 20 | Tool extraction, marker removal, text preservation, GLM4 dedup, Llama Python tag |
| TestResponseProcessorThinking | 2 | Thinking extraction, tag removal |
| TestStreamingProcessor | 4 | Pattern filtering, finalize() extraction |

## Decisions Made
None - followed plan as specified

## Deviations from Plan
None - plan executed exactly as written

## Issues Encountered
None

## User Setup Required
None - no external service configuration required

## Next Phase Readiness
- Phase 15 complete with all 3 plans executed
- Full test suite passes (1274 tests)
- ResponseProcessor architecture validated with comprehensive integration tests

---
*Phase: 15-code-cleanup-integration-tests*
*Completed: 2026-02-02*
