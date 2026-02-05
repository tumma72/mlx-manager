---
phase: 15-code-cleanup-integration-tests
plan: 11
subsystem: testing
tags: [e2e, cross-protocol, openai, anthropic, protocol-translator, pytest]

dependency-graph:
  requires: ["15-10"]
  provides: ["cross-protocol-e2e-tests", "golden-prompt-fixtures", "text-model-e2e-infrastructure"]
  affects: ["15-12", "15-13"]

tech-stack:
  added: []
  patterns: ["cross-protocol-comparison", "golden-prompt-fixtures", "thinking-model-token-budget"]

key-files:
  created:
    - backend/tests/fixtures/golden/prompts/simple_greeting.txt
    - backend/tests/fixtures/golden/prompts/factual_question.txt
    - backend/tests/fixtures/golden/prompts/tool_call_request.txt
    - backend/tests/fixtures/golden/prompts/system_instruction.txt
    - backend/tests/e2e/test_cross_protocol_e2e.py
  modified:
    - backend/tests/e2e/conftest.py

decisions:
  - id: thinking-model-token-budget
    description: "System message tests use 512 max_tokens (not 128) because Qwen3 thinking models consume tokens for internal reasoning before producing visible output"
    rationale: "With 128 tokens, Qwen3 spent all budget on reasoning content, leaving empty visible response"

metrics:
  duration: "3.5 min"
  completed: "2026-02-05"
---

# Phase 15 Plan 11: Cross-Protocol E2E Tests Summary

**Cross-protocol E2E test suite validating OpenAI vs Anthropic API equivalence using Qwen3-0.6B-4bit-DWQ model with shared golden prompt fixtures**

## What Was Done

Created an end-to-end test suite that sends identical prompts through both the OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` endpoints, verifying that the same local model produces valid, structurally correct responses through both protocols.

### Task 1: Golden Prompt Fixtures (318291c)
Created 4 shared golden prompt files in `backend/tests/fixtures/golden/prompts/`:
- `simple_greeting.txt` - single-sentence greeting (short, verifiable)
- `factual_question.txt` - capital of France (one-word factual answer)
- `tool_call_request.txt` - weather tool call trigger
- `system_instruction.txt` - French translation instruction

### Task 2: E2E Conftest Additions (cafd1aa)
Extended existing `backend/tests/e2e/conftest.py` with:
- `TEXT_MODEL_QUICK` constant pointing to `mlx-community/Qwen3-0.6B-4bit-DWQ`
- `PROMPTS_DIR` path to golden prompt fixtures
- `WEATHER_TOOL_OPENAI` shared tool definition for function calling tests
- `text_model_quick` session-scoped fixture with `is_model_available()` check

### Task 3: Cross-Protocol E2E Test Suite (130510f)
Created `backend/tests/e2e/test_cross_protocol_e2e.py` with 5 test classes (11 tests total):

| Class | Tests | What It Validates |
|-------|-------|-------------------|
| TestCrossProtocolSimple | 3 | Greeting + factual via both APIs, semantic equivalence |
| TestCrossProtocolSystemMessages | 2 | System message handling (array vs field) |
| TestCrossProtocolStreaming | 2 | SSE streaming for both protocols |
| TestCrossProtocolToolCalling | 1 | Tool call structure via OpenAI endpoint |
| TestProtocolResponseStructure | 3 | Spec compliance, field presence, stop reason translation |

All tests marked with `@pytest.mark.e2e` and `@pytest.mark.e2e_anthropic`. Run with `pytest -m e2e_anthropic -v`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Thinking model exhausts token budget on reasoning**
- **Found during:** Task 3 (initial test run)
- **Issue:** Qwen3-0.6B with enable_thinking=True consumed all 128 max_tokens for internal reasoning, producing empty visible content in the Anthropic system message test
- **Fix:** Increased max_tokens from 128 to 512 for system message tests, giving thinking models room for both reasoning and visible output
- **Files modified:** backend/tests/e2e/test_cross_protocol_e2e.py
- **Commit:** 130510f

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 512 max_tokens for system message tests | Thinking models (Qwen3) need token budget for internal reasoning + visible output; 128 is insufficient |
| Tool call test allows content-only fallback | Small models may not always trigger tool calls, so test accepts either tool_call or content response |
| Both protocol markers on all tests | `@pytest.mark.e2e` and `@pytest.mark.e2e_anthropic` enable running as part of full E2E or isolated |

## Verification Results

1. Golden prompts exist: 4 files in `backend/tests/fixtures/golden/prompts/`
2. Tests collect correctly: 11 tests with `e2e_anthropic` marker
3. Cross-protocol tests pass: 11/11 passed in 10.74s
4. Both protocol responses structurally valid (OpenAI: chat.completion + choices + usage; Anthropic: message + content blocks + usage)
5. Stop reason translation verified (stop -> end_turn)
6. Existing tests still pass: 1326 passed, 31 deselected

## Next Phase Readiness

The cross-protocol E2E infrastructure is ready for:
- **15-12 (Embeddings E2E)**: Can reuse text_model_quick fixture and ASGI client patterns
- **15-13 (Audio E2E)**: Golden prompt pattern extends naturally to audio test cases
