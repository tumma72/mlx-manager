# Phase 15: Code Cleanup & Integration Tests - Context

**Gathered:** 2026-02-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove dead parser code that duplicates ResponseProcessor functionality, fix blocker bugs discovered during Phase 14 UAT, and create comprehensive integration tests for the core inference pipeline. This ensures the ResponseProcessor-based architecture works correctly with all model families.

</domain>

<decisions>
## Implementation Decisions

### Dead Code Removal Scope
- Delete entire `adapters/parsers/` folder (QwenToolParser, LlamaToolParser, GLM4ToolParser, base.py)
- Remove `parse_tool_calls()` and `extract_reasoning()` methods from all adapters
- Remove these methods from the ModelAdapter protocol in base.py
- Delete related tests that test the removed parser classes and adapter methods
- Clean up imports that referenced the deleted code

### Integration Test Strategy
- Use golden files only (no live model generation at test time)
- Organize golden files by model family first, then output format: `backend/tests/fixtures/golden/{family}/{format}.txt`
- Include both complete response tests AND streaming chunk sequences
- Streaming tests stored in `{family}/stream/` subfolder

### Bug Fix Approach
- **DB Migration:** Claude decides safest approach for adding api_type and name columns with existing data handling
- **Qwen Exception:** Catch specific exception types (TypeError, ValueError, KeyError) for enable_thinking fallback
- **Logging Audit:** Audit entire application backend for logging levels, not just specific fix
- **LogFire Fix:** Include LogFire instrumentation investigation and fix - traces not reaching Pydantic LogFire endpoint
- **Vision Fix:** Fix processor attribute access issue for vision models

### Test Coverage Goals
- **Tool calling:** Golden files for ALL families: Hermes, Qwen, Llama, GLM4, MiniMax, Gemma
- **Thinking/reasoning:** Golden files for all models that could output thinking tags
- **Streaming:** Golden chunk sequences for pattern filtering validation
- **Vision:** Fix and test processor handling
- **Pass criteria:** ALL golden tests must pass - no exceptions

### Claude's Discretion
- Exact DB migration strategy (defaults, backfill approach)
- Which specific exception types to catch beyond TypeError, ValueError, KeyError
- Golden file naming conventions within the structure
- LogFire diagnostic approach

</decisions>

<specifics>
## Specific Ideas

- Family-first organization: `backend/tests/fixtures/golden/qwen/tool_calls.txt`, `backend/tests/fixtures/golden/qwen/stream/thinking.txt`
- ResponseProcessor replaced the parsers - now we test ResponseProcessor, not the deleted code
- LogFire issue: "running capture on logfire-eu.pydantic.dev for hours and not a single trace reached" - needs investigation

</specifics>

<deferred>
## Deferred Ideas

- Server gauges showing per-model metrics instead of system metrics - UI issue, separate phase
- Thinking content display in Chat UI thinking bubble - frontend issue, separate phase
- Multiple custom providers UI redesign - Phase 14 UAT issue, separate phase

</deferred>

---

*Phase: 15-code-cleanup-integration-tests*
*Context gathered: 2026-02-02*
