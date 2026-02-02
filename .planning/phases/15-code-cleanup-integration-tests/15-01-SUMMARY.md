---
phase: 15-code-cleanup-integration-tests
plan: 01
subsystem: mlx-server
tags: [adapters, parsers, tool-calling, reasoning, cleanup, refactor]

# Dependency graph
requires:
  - phase: 14-model-adapter-enhancements
    provides: ResponseProcessor for unified tool call and reasoning extraction
provides:
  - Clean adapter codebase without dead parser code
  - Streamlined ModelAdapter protocol
  - Inline format_tools_for_prompt in each adapter
affects: [15-02, 15-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ResponseProcessor singleton for all tool/reasoning extraction"
    - "Adapter format_tools_for_prompt inlined per model family"

key-files:
  created: []
  modified:
    - backend/mlx_manager/mlx_server/models/adapters/base.py
    - backend/mlx_manager/mlx_server/models/adapters/qwen.py
    - backend/mlx_manager/mlx_server/models/adapters/glm4.py
    - backend/mlx_manager/mlx_server/models/adapters/llama.py
    - backend/tests/mlx_server/test_tool_calling.py
    - backend/tests/mlx_server/test_reasoning.py
  deleted:
    - backend/mlx_manager/mlx_server/models/adapters/parsers/ (entire directory)

key-decisions:
  - "Inline format_tools_for_prompt in each adapter instead of delegating to deleted parsers"
  - "Keep supports_tool_calling and supports_reasoning_mode in adapters as capability flags"
  - "Processor compatibility pattern added to all adapters for vision model support"

patterns-established:
  - "ResponseProcessor as single source of truth: All tool call parsing and reasoning extraction via ResponseProcessor.process()"
  - "Adapter responsibility boundary: Adapters handle chat templates, stop tokens, tool prompt formatting; ResponseProcessor handles response parsing"

# Metrics
duration: 6min
completed: 2026-02-02
---

# Phase 15 Plan 01: Dead Parser Code Removal Summary

**Removed 617 lines of dead parser code, streamlined ModelAdapter protocol, and consolidated tool/reasoning extraction in ResponseProcessor**

## Performance

- **Duration:** 6 min 35 sec
- **Started:** 2026-02-02T17:21:51Z
- **Completed:** 2026-02-02T17:28:26Z
- **Tasks:** 3
- **Files deleted:** 5 (entire parsers/ directory)
- **Files modified:** 8

## Accomplishments
- Deleted entire adapters/parsers/ directory (QwenToolParser, LlamaToolParser, GLM4ToolParser, base.py, __init__.py)
- Removed parse_tool_calls() and extract_reasoning() from ModelAdapter protocol and all implementations
- Inlined format_tools_for_prompt() in each adapter (was delegating to deleted parsers)
- Updated tests to remove parser class tests and adapter method tests
- All 1248 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove parser imports and delete parsers directory** - `b118c78` (refactor)
2. **Task 2: Remove parse_tool_calls and extract_reasoning from adapters** - `3f7dbd8` (refactor)
3. **Task 3: Update tests and verify clean build** - `d7016fe` (test)

## Files Deleted
- `backend/mlx_manager/mlx_server/models/adapters/parsers/__init__.py`
- `backend/mlx_manager/mlx_server/models/adapters/parsers/base.py`
- `backend/mlx_manager/mlx_server/models/adapters/parsers/glm4.py`
- `backend/mlx_manager/mlx_server/models/adapters/parsers/llama.py`
- `backend/mlx_manager/mlx_server/models/adapters/parsers/qwen.py`

## Files Modified
- `backend/mlx_manager/mlx_server/models/adapters/base.py` - Removed parse_tool_calls/extract_reasoning from Protocol
- `backend/mlx_manager/mlx_server/models/adapters/qwen.py` - Removed dead methods, inlined format_tools_for_prompt
- `backend/mlx_manager/mlx_server/models/adapters/glm4.py` - Removed dead methods, inlined format_tools_for_prompt
- `backend/mlx_manager/mlx_server/models/adapters/llama.py` - Removed dead methods, inlined format_tools_for_prompt
- `backend/mlx_manager/mlx_server/models/adapters/gemma.py` - Processor compatibility pattern (linter auto-fix)
- `backend/mlx_manager/mlx_server/models/adapters/mistral.py` - Processor compatibility pattern (linter auto-fix)
- `backend/tests/mlx_server/test_tool_calling.py` - Removed parser tests, kept adapter support tests
- `backend/tests/mlx_server/test_reasoning.py` - Removed adapter.extract_reasoning tests

## Decisions Made

1. **Inline format_tools_for_prompt**: The adapter methods were delegating to `_tool_parser.format_tools(tools)` which referenced deleted parsers. Instead of creating a new abstraction, inlined the formatting logic directly in each adapter since each model family has unique formatting requirements.

2. **Keep capability flags**: Retained `supports_tool_calling()` and `supports_reasoning_mode()` in adapters since inference.py uses these to decide whether to inject tools into prompts.

3. **Processor compatibility**: Linter propagated the `getattr(tokenizer, "tokenizer", tokenizer)` pattern to all adapters for consistent vision model Processor support.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed broken format_tools_for_prompt methods**
- **Found during:** Task 2 (removing parse_tool_calls)
- **Issue:** Adapters' format_tools_for_prompt methods referenced deleted `_tool_parser` variable
- **Fix:** Inlined the tool formatting logic directly in each adapter with model-specific formats
- **Files modified:** qwen.py, glm4.py, llama.py
- **Verification:** Tests pass, format_tools_for_prompt returns expected formats
- **Committed in:** 3f7dbd8 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (blocking issue)
**Impact on plan:** Essential fix to maintain tool calling functionality. No scope creep.

## Issues Encountered
- Pre-existing mypy errors in chat.py, system.py, settings.py, admin.py, database.py (20 errors) - documented in STATE.md, unrelated to this plan

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Dead parser code removed
- Codebase is cleaner with clear separation: adapters handle templates/tokens, ResponseProcessor handles parsing
- Ready for Plan 15-02 (bug fixes) and Plan 15-03 (integration tests)

---
*Phase: 15-code-cleanup-integration-tests*
*Completed: 2026-02-02*
