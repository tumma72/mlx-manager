---
phase: 14-model-adapter-enhancements
plan: 02
subsystem: api
tags: [tool-calling, llama, qwen, glm4, parsers, adapters]

# Dependency graph
requires:
  - phase: 14-01
    provides: Extended ModelAdapter protocol with tool calling methods
provides:
  - LlamaToolParser for XML-style function calls
  - QwenToolParser for Hermes-style tool calls
  - GLM4ToolParser for XML tool calls with deduplication
  - GLM4Adapter with full tool calling support
  - Tool calling wired into Llama, Qwen, and GLM4 adapters
affects: [14-03, inference-service]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Model-specific tool parsers in parsers/ subdirectory"
    - "Module-level parser instances for efficiency"
    - "Content hash deduplication for known model bugs"

key-files:
  created:
    - backend/mlx_manager/mlx_server/models/adapters/parsers/__init__.py
    - backend/mlx_manager/mlx_server/models/adapters/parsers/base.py
    - backend/mlx_manager/mlx_server/models/adapters/parsers/llama.py
    - backend/mlx_manager/mlx_server/models/adapters/parsers/qwen.py
    - backend/mlx_manager/mlx_server/models/adapters/parsers/glm4.py
    - backend/mlx_manager/mlx_server/models/adapters/glm4.py
  modified:
    - backend/mlx_manager/mlx_server/models/adapters/llama.py
    - backend/mlx_manager/mlx_server/models/adapters/qwen.py
    - backend/mlx_manager/mlx_server/models/adapters/registry.py

key-decisions:
  - "GLM4 deduplication via MD5 content hash - handles known duplicate tag bug"
  - "Module-level parser instances - avoid repeated instantiation"
  - "Parsers return empty list, adapters convert to None - consistent with protocol"

patterns-established:
  - "Tool parsers in adapters/parsers/ directory"
  - "Each model family has dedicated ToolCallParser subclass"
  - "Adapters delegate to parsers via module-level instances"

# Metrics
duration: 7min
completed: 2026-02-01
---

# Phase 14 Plan 02: Tool Parsers Summary

**Model-specific tool call parsers for Llama (<function=>), Qwen (Hermes-style), and GLM4 (XML) with OpenAI-compatible output format**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-01T11:06:23Z
- **Completed:** 2026-02-01T11:13:04Z
- **Tasks:** 3
- **Files created:** 6
- **Files modified:** 3

## Accomplishments

- Created ToolCallParser abstract base class defining parse() and format_tools() interface
- Implemented LlamaToolParser supporting XML-style `<function=name>{args}</function>` format
- Implemented QwenToolParser supporting Hermes-style `<tool_call>{json}</tool_call>` format
- Implemented GLM4ToolParser with XML parsing and duplicate tag deduplication
- Created GLM4Adapter with full tool calling, reasoning mode, and chat template support
- Wired tool parsers into Llama, Qwen, and GLM4 adapters
- Registered GLM4Adapter and added "glm"/"chatglm" pattern detection

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Tool Parser Base and Llama Parser** - `033b9a6` (feat)
2. **Task 2: Create Qwen and GLM4 Parsers** - `bb98087` (feat)
3. **Task 3: Wire Parsers into Adapters and Add GLM4 Adapter** - `6ba50bd` (feat - included in 14-05 commit)

_Note: Task 3 was committed as part of plan 14-05 due to parallel execution_

## Files Created/Modified

### Created
- `adapters/parsers/__init__.py` - Exports all parsers
- `adapters/parsers/base.py` - ToolCallParser ABC with parse() and format_tools()
- `adapters/parsers/llama.py` - LlamaToolParser for XML-style function calls
- `adapters/parsers/qwen.py` - QwenToolParser for Hermes-style JSON
- `adapters/parsers/glm4.py` - GLM4ToolParser with deduplication
- `adapters/glm4.py` - GLM4Adapter with tool calling and reasoning support

### Modified
- `adapters/llama.py` - Added tool calling support via LlamaToolParser
- `adapters/qwen.py` - Added tool calling support via QwenToolParser
- `adapters/registry.py` - Registered GLM4Adapter, added glm/chatglm detection

## Decisions Made

1. **GLM4 deduplication via MD5 content hash** - GLM4 has a known bug where it outputs duplicate `<tool_call>` markers. Using content hash (name + arguments) to deduplicate ensures each unique tool call is returned only once.

2. **Module-level parser instances** - Parser classes are stateless, so module-level instances (`_tool_parser = LlamaToolParser()`) avoid repeated instantiation overhead.

3. **Parsers return empty list, adapters convert to None** - Parsers always return `list[dict]` (empty if no calls). Adapters wrap this to return `None` when empty, matching the protocol's semantics.

4. **Tool call stop tokens via tokenizer lookup** - Rather than hardcoding token IDs, adapters look up stop tokens like `<|eom_id|>` from the tokenizer to handle model version differences.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all implementations worked as expected.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Tool parsing infrastructure complete and tested
- All three major model families (Llama, Qwen, GLM4) support tool calling
- Ready for inference service integration in future work
- Reasoning extraction (14-03) can proceed independently

---
*Phase: 14-model-adapter-enhancements*
*Completed: 2026-02-01*
