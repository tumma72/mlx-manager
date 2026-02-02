---
phase: 14-model-adapter-enhancements
plan: 07
subsystem: api
tags: [pydantic, response-processing, tool-calling, reasoning, regex]

# Dependency graph
requires:
  - phase: 14-06
    provides: Chat endpoint integration with tool calling infrastructure
provides:
  - Unified ResponseProcessor for single-pass extraction
  - Pydantic models (ParseResult, ToolCall) for type safety
  - Tool call marker removal from content (bug fix)
  - GLM4 deduplication via content hash
affects: [14-08-streaming-processor, mlx-server-maintenance]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Callback-based pattern registration for extensibility
    - Singleton processor with compiled patterns
    - Span-based content removal (reverse order)

key-files:
  created:
    - backend/mlx_manager/mlx_server/services/response_processor.py
    - backend/tests/mlx_server/test_response_processor.py
  modified:
    - backend/mlx_manager/mlx_server/services/inference.py

key-decisions:
  - "Use Pydantic BaseModel for ParseResult and ToolCall for type safety and serialization"
  - "Single-pass extraction with span-based removal instead of multi-pass regex substitution"
  - "GLM4 deduplication via MD5 hash of (name, arguments) tuple"
  - "Empty thinking tags still removed from content even if no reasoning extracted"

patterns-established:
  - "Callback-based pattern registration: register_tool_pattern(pattern, parser_callback)"
  - "Span merging for overlapping matches before removal"
  - "Singleton pattern for response processor (compile patterns once)"

# Metrics
duration: 6min
completed: 2026-02-02
---

# Phase 14 Plan 07: Unified ResponseProcessor Summary

**Single-pass ResponseProcessor with Pydantic models extracts tool calls, reasoning, and cleans content in one scan, fixing the bug where tool call markers remained in output**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-02T12:29:31Z
- **Completed:** 2026-02-02T12:35:22Z
- **Tasks:** 4 (3 automated + 1 verification)
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- Created unified ResponseProcessor replacing multi-pass adapter parsing
- Fixed bug where tool call markers (`<tool_call>`, `<function=...>`) remained in content
- Implemented GLM4 deduplication to handle known duplicate tag bug
- Added comprehensive test suite with 44 tests covering all patterns and edge cases
- Integrated into inference service for both streaming and non-streaming paths

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ResponseProcessor with Pydantic Models** - `d7efc3a` (feat)
2. **Task 2: Update Inference Service** - `10c3737` (fix)
3. **Task 3: Add Comprehensive Tests** - `771b25f` (test)
4. **Task 4: Verification** - `8b426de` (style - formatting)

## Files Created/Modified

- `backend/mlx_manager/mlx_server/services/response_processor.py` - New unified processor with Pydantic models, pattern registration, single-pass extraction
- `backend/mlx_manager/mlx_server/services/inference.py` - Updated to use ResponseProcessor instead of adapter methods
- `backend/tests/mlx_server/test_response_processor.py` - 44 comprehensive tests

## Decisions Made

1. **Pydantic BaseModel over dataclass** - Better serialization with model_dump(), validation support, and FastAPI integration
2. **Python re module over third-party regex** - C-accelerated, sufficient for our patterns, no new dependency
3. **Span-based removal in reverse order** - Preserves string indices during removal, handles overlapping matches
4. **Empty tags still removed** - Changed to remove empty thinking tags from content even when no reasoning content extracted

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Empty thinking tags not removed from content**
- **Found during:** Task 3 (test execution)
- **Issue:** Empty `<think></think>` tags were not removed from content because we only recorded spans when content was non-empty
- **Fix:** Always record span for removal, only add to reasoning_parts if content non-empty
- **Files modified:** response_processor.py
- **Verification:** Test `test_empty_tags` now passes
- **Committed in:** 771b25f (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix was necessary for correct behavior with edge case input. No scope creep.

## Issues Encountered

- **5 failing tests in full suite** - These are from uncommitted working directory changes from previous phases (Qwen adapter, chat streaming tests), not from this plan's changes. All 44 ResponseProcessor tests pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ResponseProcessor ready for streaming integration (Plan 14-08)
- Pattern registration system allows easy addition of new tool formats
- Singleton pattern ensures compiled patterns reused across requests

**Blockers:** None

**Notes for 14-08 (StreamingProcessor):**
- Can build on ResponseProcessor patterns
- Need incremental extraction for streaming (not full-text processing)
- Consider shared pattern registry

---
*Phase: 14-model-adapter-enhancements*
*Plan: 07*
*Completed: 2026-02-02*
