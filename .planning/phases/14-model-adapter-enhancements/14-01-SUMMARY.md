---
phase: 14-model-adapter-enhancements
plan: 01
subsystem: api
tags: [pydantic, tool-calling, openai-api, model-adapters, protocol]

# Dependency graph
requires:
  - phase: 08-server-foundation
    provides: ModelAdapter protocol with chat templates and stop tokens
provides:
  - OpenAI-compatible tool calling schemas (Tool, ToolCall, FunctionCall, FunctionDefinition)
  - ResponseFormat schema for structured output
  - Extended ModelAdapter protocol with 7 optional methods
  - Default implementations in DefaultAdapter for tool calling, reasoning, message conversion
affects: [14-02 tool parsers, 14-03 reasoning extraction, 14-04 message converters, 14-05 structured output, 14-06 LoRA]

# Tech tracking
tech-stack:
  added: []
  patterns: [optional protocol methods with default implementations, adapter inheritance for protocol compliance]

key-files:
  created: []
  modified:
    - backend/mlx_manager/mlx_server/schemas/openai.py
    - backend/mlx_manager/mlx_server/models/adapters/base.py
    - backend/mlx_manager/mlx_server/models/adapters/llama.py
    - backend/mlx_manager/mlx_server/models/adapters/qwen.py
    - backend/mlx_manager/mlx_server/models/adapters/mistral.py
    - backend/mlx_manager/mlx_server/models/adapters/gemma.py

key-decisions:
  - "Adapters inherit from DefaultAdapter for protocol compliance"
  - "ToolChoiceOption as union type alias for flexibility"
  - "ChatMessage content nullable to support tool-only messages"

patterns-established:
  - "Optional protocol methods: Add to Protocol, implement defaults in DefaultAdapter, adapters inherit"
  - "Tool call format: {id, type, function: {name, arguments}}"

# Metrics
duration: 4min
completed: 2026-02-01
---

# Phase 14 Plan 01: Extended OpenAI Schemas & ModelAdapter Protocol Summary

**OpenAI-compatible tool calling schemas and extended ModelAdapter protocol with optional methods for tool calling, reasoning extraction, and message conversion**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-01T18:00:00Z
- **Completed:** 2026-02-01T18:04:00Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Added FunctionDefinition, Tool, FunctionCall, ToolCall schemas for tool calling
- Added ResponseFormat schema for structured output (text/json_object/json_schema)
- Added ToolCallDelta for streaming tool calls
- Extended ChatMessage with tool_calls and tool_call_id fields
- Extended ChatCompletionRequest with tools, tool_choice, response_format
- Extended ModelAdapter protocol with 7 new optional methods
- DefaultAdapter implements all new methods with sensible defaults
- All existing adapters (Llama, Qwen, Mistral, Gemma) inherit from DefaultAdapter

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Tool Calling Schemas to openai.py** - `5f3fa94` (feat)
2. **Task 2: Extend ModelAdapter Protocol** - `94a4b46` (feat)
3. **Task 3: Run Quality Checks** - `d66c845` (chore)

## Files Created/Modified
- `backend/mlx_manager/mlx_server/schemas/openai.py` - Tool calling schemas (Tool, ToolCall, FunctionCall, etc.)
- `backend/mlx_manager/mlx_server/models/adapters/base.py` - Extended ModelAdapter protocol + DefaultAdapter
- `backend/mlx_manager/mlx_server/models/adapters/llama.py` - Now inherits from DefaultAdapter
- `backend/mlx_manager/mlx_server/models/adapters/qwen.py` - Now inherits from DefaultAdapter
- `backend/mlx_manager/mlx_server/models/adapters/mistral.py` - Now inherits from DefaultAdapter
- `backend/mlx_manager/mlx_server/models/adapters/gemma.py` - Now inherits from DefaultAdapter

## Decisions Made
- **Adapters inherit from DefaultAdapter:** Makes existing adapters (Llama, Qwen, Mistral, Gemma) protocol-compliant without duplicating code
- **ToolChoiceOption as type alias:** `Literal["none", "auto", "required"] | dict[str, Any] | None` provides flexibility for tool_choice parameter
- **ChatMessage content nullable:** Allows tool-only assistant messages where content is None but tool_calls is populated
- **ToolCallDelta for streaming:** Separate model for partial tool calls in streaming responses

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Made existing adapters inherit from DefaultAdapter**
- **Found during:** Task 3 (Quality Checks)
- **Issue:** mypy reported type errors in registry.py because LlamaAdapter, QwenAdapter, MistralAdapter, GemmaAdapter didn't implement new Protocol methods
- **Fix:** Made all adapters inherit from DefaultAdapter to get default implementations
- **Files modified:** llama.py, qwen.py, mistral.py, gemma.py
- **Verification:** mypy passes with no errors
- **Committed in:** d66c845 (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Auto-fix necessary for type safety. No scope creep - adapters now properly inherit defaults.

## Issues Encountered
- ruff auto-fixed 1 unused import in openai.py (Any was imported twice)
- No other issues

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Tool calling schemas ready for Plan 02 (Tool Parsers)
- ModelAdapter protocol ready for Plan 03 (Reasoning Extraction)
- Message conversion methods ready for Plan 04 (Message Converters)
- ResponseFormat schema ready for Plan 05 (Structured Output)
- All foundation types and interfaces in place

---
*Phase: 14-model-adapter-enhancements*
*Plan: 01*
*Completed: 2026-02-01*
