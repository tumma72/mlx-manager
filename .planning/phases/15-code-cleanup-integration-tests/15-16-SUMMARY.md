# Phase 15 Plan 16: MLX Server Architecture Compliance Summary

Architecture compliance refactor bringing the MLX Server implementation into alignment with the ARCHITECTURE.md blueprint. Deleted dead code, unified duplicate types, fixed message conversion data loss bug, added adapter convert_messages() overrides, and made StreamingProcessor family-aware.

## Tasks Completed

| Task | Issue | Type | Commit | Key Files |
|------|-------|------|--------|-----------|
| 1 | Delete ReasoningExtractor | Cleanup | d95ac9f | services/reasoning.py (deleted), test_reasoning.py (deleted), test_adapters.py |
| 2 | Unify ToolCall models | Refactor | b8f8b54 | services/response_processor.py, api/v1/chat.py, test_response_processor.py |
| 3 | Fix message conversion gap | Bug fix | b17eaaf | api/v1/chat.py |
| 4 | Add convert_messages() overrides | Bug fix | dd68ac6 | adapters/base.py, adapters/qwen.py, adapters/llama.py, test_adapters.py |
| 5 | Family-aware StreamingProcessor | Refactor | 1859906 | services/response_processor.py, test_response_processor.py |
| 6 | Thread management duplication | Deferred | -- | -- |

## Changes Summary

### Issue 1: Dead Code Removal (P5)
- Deleted `services/reasoning.py` (ReasoningExtractor) - zero production references
- Deleted `tests/mlx_server/test_reasoning.py` (19 tests)
- Migrated 4 adapter reasoning support tests + added 2 new ones to `test_adapters.py`
- Net: -19 tests (dead), +6 tests (migrated/new)

### Issue 2: Canonical ToolCall Type (P4)
- Removed duplicate `ToolCall`/`ToolCallFunction` from `response_processor.py`
- All tool call handling now uses `schemas/openai.py::ToolCall` and `FunctionCall`
- Simplified `_convert_tool_calls()` to use `ToolCall.model_validate()`
- No bridging code needed between response processor and API layer

### Issue 3: Message Conversion Data Loss Fix (P3)
- Added `_convert_messages_to_dicts()` helper preserving all fields:
  `role`, `content`, `tool_calls` (serialized), `tool_call_id`
- Replaced 4 inline conversion loops that dropped tool fields
- Handles `None` content for assistant messages with only tool_calls
- Root cause fix for multi-turn tool use failures via OpenAI API

### Issue 4: Adapter convert_messages() (P2)
- `QwenAdapter.convert_messages()`: Converts tool messages using Hermes-style `<tool_call>` tags
- `LlamaAdapter.convert_messages()`: Converts tool messages using `<function=name>` tags
- `DefaultAdapter.convert_messages()`: Safe fallback converting `role="tool"` to `role="user"`
- Root cause fix for "Can only get item pairs from a mapping" Jinja error
- Added 15 tests covering all adapters including multi-turn scenarios

### Issue 5: Family-Aware StreamingProcessor (P2, P5)
- Replaced hardcoded class variables with instance-level config from `ModelFamilyPatterns`
- Added `streaming_tool_markers` field to `ModelFamilyPatterns`
- Added helper methods: `get_thinking_starts()`, `get_all_pattern_ends()`
- Qwen processor only detects `<tool_call>`, not `<function=` (family isolation)
- Llama processor only detects `<function=`, not `<tool_call>` (family isolation)
- Default processor maintains all markers for backward compatibility
- Added 7 tests verifying family-specific streaming isolation

## Architecture Principles Addressed

| Principle | Issue(s) | Status |
|-----------|----------|--------|
| P1: Single inference pipeline | -- | Already compliant |
| P2: Adapter-driven model handling | 4, 5 | Fixed: all tool adapters override convert_messages(), streaming uses family config |
| P3: No data loss through layers | 3 | Fixed: tool_calls, tool_call_id preserved in message conversion |
| P4: One canonical type per concept | 2 | Fixed: single ToolCall/FunctionCall from schemas/openai.py |
| P5: Shared infrastructure | 1, 5 | Fixed: dead code removed, streaming config derived from shared patterns |

## Deviations from Plan

None - plan executed exactly as written.

## Test Results

- Tests before: 1413 passed
- Tests after: 1422 passed (+9 net: -19 dead reasoning tests, +6 migrated, +15 convert_messages, +7 streaming)
- All existing tests continue to pass
- Pre-existing mypy errors (22) unchanged

## Metrics

- Duration: ~12 minutes
- Completed: 2026-02-05
- Files created: 0
- Files modified: 7
- Files deleted: 2
- Net lines: +522 added, -410 removed
