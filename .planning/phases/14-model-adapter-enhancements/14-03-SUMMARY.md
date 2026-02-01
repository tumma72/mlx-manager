---
phase: 14-model-adapter-enhancements
plan: 03
subsystem: api
tags: [mlx, reasoning, openai-compatible, chain-of-thought, deepseek-r1, qwen3-thinking]

# Dependency graph
requires:
  - phase: 14-01
    provides: Extended ModelAdapter protocol with supports_reasoning_mode() and extract_reasoning() methods
provides:
  - ReasoningExtractor service for chain-of-thought extraction
  - LlamaAdapter and QwenAdapter with reasoning mode support
  - reasoning_content field in OpenAI-compatible schemas
affects: [14-05-message-converters, inference-service, chat-router]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Module-level extractor instances for efficiency
    - Regex-based pattern extraction with re.DOTALL for multiline

key-files:
  created:
    - backend/mlx_manager/mlx_server/services/reasoning.py
  modified:
    - backend/mlx_manager/mlx_server/models/adapters/llama.py
    - backend/mlx_manager/mlx_server/models/adapters/qwen.py
    - backend/mlx_manager/mlx_server/schemas/openai.py

key-decisions:
  - "Module-level ReasoningExtractor instances for efficiency - no per-call overhead"
  - "Four tag patterns: <think>, <thinking>, <reasoning>, <reflection>"
  - "Adapters report capability via supports_reasoning_mode(); extraction only when tags present"
  - "reasoning_content field follows Anthropic Claude API pattern"

patterns-established:
  - "Reasoning extraction delegated to central ReasoningExtractor service"
  - "Adapters inherit reasoning support from DefaultAdapter, override as needed"

# Metrics
duration: 8min
completed: 2026-02-01
---

# Phase 14 Plan 03: Reasoning Extraction Summary

**ReasoningExtractor service for chain-of-thought content extraction from `<think>`, `<thinking>`, `<reasoning>`, and `<reflection>` tags with adapter integration and OpenAI schema support**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-01T12:05:00Z
- **Completed:** 2026-02-01T12:13:00Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Created ReasoningExtractor service handling four different tag patterns
- Added reasoning mode support to LlamaAdapter and QwenAdapter
- Extended ChatMessage and ChatCompletionChunkDelta with reasoning_content field
- Full multiline content support and multiple tag extraction

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ReasoningExtractor Service** - `3042df0` (feat)
2. **Task 2: Add Reasoning Support to Adapters** - `5f251af` (feat)
3. **Task 3: Extend OpenAI Schemas for Reasoning Content** - `853f230` (feat)

## Files Created/Modified

- `backend/mlx_manager/mlx_server/services/reasoning.py` - ReasoningExtractor class with extract() and has_reasoning_tags() methods
- `backend/mlx_manager/mlx_server/models/adapters/llama.py` - Added supports_reasoning_mode() and extract_reasoning()
- `backend/mlx_manager/mlx_server/models/adapters/qwen.py` - Added supports_reasoning_mode() and extract_reasoning()
- `backend/mlx_manager/mlx_server/schemas/openai.py` - Added reasoning_content field to ChatMessage and ChatCompletionChunkDelta

## Decisions Made

1. **Module-level extractor instances** - Created `_reasoning_extractor` at module level in adapters for efficiency, avoiding per-call instantiation overhead
2. **Four tag pattern support** - Supporting `<think>`, `<thinking>`, `<reasoning>`, and `<reflection>` tags covers DeepSeek-R1, Qwen3-thinking, and other reasoning models
3. **Capability vs actual output** - Adapters report `supports_reasoning_mode() = True` but extraction only occurs when model actually outputs reasoning tags
4. **Anthropic-style reasoning_content** - Following the pattern used by Anthropic's Claude API for "thinking" content

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- ruff not installed in venv - used `uvx ruff` instead for linting
- Pre-existing mypy error in pool.py unrelated to this plan's changes

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Reasoning extraction infrastructure complete
- Ready for integration in inference service to populate reasoning_content in responses
- Plan 14-04 (Message Converters) can proceed

---
*Phase: 14-model-adapter-enhancements*
*Completed: 2026-02-01*
