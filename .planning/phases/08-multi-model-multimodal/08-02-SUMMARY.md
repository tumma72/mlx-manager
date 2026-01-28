---
phase: 08-multi-model-multimodal
plan: 02
subsystem: adapters
tags: [qwen, mistral, gemma, chat-template, stop-tokens, chatml]

# Dependency graph
requires:
  - phase: 08-01
    provides: "Adapter protocol and LlamaAdapter reference implementation"
provides:
  - "QwenAdapter with ChatML <|im_end|> stop token"
  - "MistralAdapter with system message prepending for v1/v2 compatibility"
  - "GemmaAdapter with <end_of_turn> stop token"
  - "Registry with all four model families registered"
affects: [08-03, 08-04, inference-service]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Model family adapters follow consistent protocol"
    - "Stop tokens include both eos_token_id and turn markers"
    - "Graceful fallback when special tokens unavailable"

key-files:
  created:
    - backend/mlx_manager/mlx_server/models/adapters/qwen.py
    - backend/mlx_manager/mlx_server/models/adapters/mistral.py
    - backend/mlx_manager/mlx_server/models/adapters/gemma.py
    - backend/tests/mlx_server/test_adapters.py
  modified:
    - backend/mlx_manager/mlx_server/models/adapters/registry.py

key-decisions:
  - "Qwen uses ChatML format with <|im_end|> as end-of-turn marker"
  - "Mistral v1/v2 system messages prepended to first user message"
  - "Gemma uses <end_of_turn> as end-of-turn marker"
  - "All adapters gracefully handle missing special tokens"

patterns-established:
  - "Adapter pattern: family property, apply_chat_template, get_stop_tokens"
  - "Stop token handling: always include eos_token_id plus family-specific markers"
  - "Exception handling: silently fall back when special tokens unavailable"

# Metrics
duration: 3min
completed: 2026-01-28
---

# Phase 8 Plan 2: Model Family Adapters Summary

**Qwen, Mistral, and Gemma adapters with proper stop token handling to prevent runaway generation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-28T11:39:48Z
- **Completed:** 2026-01-28T11:42:XX
- **Tasks:** 4
- **Files modified:** 5

## Accomplishments

- QwenAdapter handling ChatML format with `<|im_end|>` stop token for Qwen/Qwen2/Qwen2.5/Qwen3 models
- MistralAdapter with system message prepending for v1/v2 compatibility and `</s>` stop token
- GemmaAdapter with `<end_of_turn>` stop token for Gemma/Gemma2/Gemma3 models
- All adapters registered in registry with auto-detection by model ID
- 17 unit tests covering all adapter behaviors

## Task Commits

Each task was committed atomically:

1. **Task 1: Create QwenAdapter** - `8811f09` (feat)
2. **Task 2: Create MistralAdapter** - `94ee9b5` (feat)
3. **Task 3: Create GemmaAdapter and register all** - `bef821f` (feat)
4. **Task 4: Add unit tests** - `7789b7b` (test)

## Files Created/Modified

- `backend/mlx_manager/mlx_server/models/adapters/qwen.py` - Qwen family adapter with ChatML stop tokens
- `backend/mlx_manager/mlx_server/models/adapters/mistral.py` - Mistral family adapter with system message handling
- `backend/mlx_manager/mlx_server/models/adapters/gemma.py` - Gemma family adapter with end_of_turn stop token
- `backend/mlx_manager/mlx_server/models/adapters/registry.py` - Updated with all new adapters
- `backend/tests/mlx_server/test_adapters.py` - 17 tests for adapter behaviors

## Decisions Made

- **Qwen ChatML format:** Uses `<|im_end|>` as the end-of-turn marker, consistent with ChatML specification
- **Mistral system message handling:** Prepends system content to first user message for v1/v2 compatibility (v3+ handles natively)
- **Gemma turn markers:** Uses `<end_of_turn>` as the end-of-turn marker
- **Graceful fallback:** All adapters catch exceptions when special tokens are not available and fall back to eos_token_id only

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Four model families now fully supported: Llama, Qwen, Mistral, Gemma
- Registry auto-detects model family from model ID
- Ready for Plan 03 (multimodal vision support with mlx-vlm)

---
*Phase: 08-multi-model-multimodal*
*Completed: 2026-01-28*
