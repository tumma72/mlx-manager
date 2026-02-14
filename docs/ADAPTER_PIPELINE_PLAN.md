# MLX Server: 3-Layer Adapter Pipeline Refactor - Implementation Plan

**Version:** 1.0
**Date:** 2026-02-13
**Status:** In Progress (Phases 0-3 complete)
**Target Architecture:** Section 1.6 of ARCHITECTURE.md

---

## Executive Summary

This plan details the incremental refactoring of the MLX Server adapter architecture from the current 2-layer design (ModelAdapter + StreamingProcessor) into a clean 3-layer pipeline (ModelAdapter + StreamProcessor + ProtocolFormatter). The refactor will:

1. **Unify all model types** under a single adapter abstraction (text, vision, embeddings, audio)
2. **Eliminate protocol awareness** from adapters via Intermediate Representation (IR) types
3. **Consolidate protocol translation** by absorbing ProtocolTranslator into ProtocolFormatter
4. **Preserve existing functionality** through incremental, testable phases
5. **Maintain 140+ Anthropic tests** and extensive E2E coverage throughout

### Key Constraints

- **Parser architecture is untouchable**: ToolCallParser and ThinkingParser stay exactly as-is
- **Family registry pattern stays**: Adapters still created per-family via `FAMILY_REGISTRY`
- **Model pool loading unchanged**: Adapter still created at load time in `LoadedModel.adapter`
- **Metal thread affinity preserved**: Generation still runs on dedicated Metal thread
- **All 140+ tests must pass**: Especially Anthropic API compatibility tests

### Success Criteria

- All existing tests pass (unit + E2E)
- Vision models use text adapter pipeline (not separate path)
- ProtocolTranslator code deleted (absorbed into formatters)
- StreamingProcessor renamed to StreamProcessor
- Clean IR flows through all layers (no protocol leakage)
- Each phase independently deployable

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Phase 0: Foundation - IR Types](#phase-0-foundation---ir-types)
3. [Phase 1: StreamProcessor Refactor](#phase-1-streamprocessor-refactor)
4. [Phase 2: ProtocolFormatter Layer](#phase-2-protocolformatter-layer)
5. [Phase 3: ModelAdapter PreparedInput](#phase-3-modeladapter-preparedinput)
6. [Phase 4: Vision Unification](#phase-4-vision-unification)
7. [Phase 5: Embeddings & Audio Integration](#phase-5-embeddings--audio-integration)
8. [Phase 6: Cleanup & Documentation](#phase-6-cleanup--documentation)
9. [Risk Assessment](#risk-assessment)
10. [Testing Strategy](#testing-strategy)
11. [Rollback Procedures](#rollback-procedures)

---

## Current State (Post Phase 3)

### What Works: TEXT_GEN Full Pipeline

The 3-layer pipeline is fully operational for TEXT_GEN models:

```
Request → [ProtocolFormatter] → adapter.prepare_input() → generate → adapter.process_complete() → [ProtocolFormatter] → Response
```

#### Layer 1: ModelAdapter (composable.py)
- **Created once** at model load, lives in `LoadedModel.adapter`
- **Full input pipeline**: `prepare_input(messages, tools, enable_prompt_injection)` → `PreparedInput`
  - Encapsulates: `convert_messages()` + `apply_chat_template()` + stop token aggregation
- **Full output pipeline**: `process_complete(raw_text, finish_reason)` → `TextResult`
  - Encapsulates: tool parsing + thinking parsing + response cleaning
- **Stream factory**: `create_stream_processor(prompt)` → `StreamProcessor`
- **Family adapters**: Qwen, GLM4, Llama, Gemma, Mistral, Liquid, Default
- **Stateless**: Safe for parallel requests from different protocols

#### Layer 2: StreamProcessor (response_processor.py)
- **Created per-request** via `adapter.create_stream_processor()`
- **Returns IR**: `feed(token)` → `StreamEvent`, `finalize()` → `TextResult`
- Backward-compat alias `StreamingProcessor` kept (deleted in Phase 6)

#### Layer 3: ProtocolFormatter (services/formatters/)
- **OpenAIFormatter**: IR → OpenAI chat completion chunks/responses
- **AnthropicFormatter**: IR → Anthropic message events/responses
- **Created per-request** in routers (stateless, protocol-specific)
- **Handles**: streaming SSE events, non-streaming responses, tool calls, reasoning content

#### IR Types (models/ir.py)
- `PreparedInput`: prompt + stop_token_ids (+ pixel_values placeholder for Phase 4)
- `StreamEvent`: type + content/reasoning_content/tool_call_delta
- `TextResult`: content + reasoning_content + tool_calls + finish_reason
- `EmbeddingResult`, `AudioResult`, `TranscriptionResult`: defined but unused (Phase 5)
- `InferenceResult`: wrapper with prompt_tokens + completion_tokens

### What Doesn't Work Yet: Vision, Embeddings, Audio

| Capability | TEXT_GEN | VISION | EMBEDDINGS | AUDIO |
|---|---|---|---|---|
| Adapter exists | Yes (7 families) | No (gets generic) | No | Yes (minimal) |
| Uses `prepare_input()` | Yes | No | No | No |
| Uses `process_complete()` | Yes | No | No | No |
| Uses formatters | OpenAI + Anthropic | Neither | Neither | Neither |
| Uses adapter pipeline | Yes | No (`vision.py`) | No (`embeddings.py`) | No (`audio.py`) |

#### Vision (separate path — Phase 4 target)
- `vision.py` bypasses adapters entirely, calls `mlx_vlm.apply_chat_template` directly
- Returns raw OpenAI dicts, not IR types
- `chat.py` router explicitly branches: `if has_images → vision path`
- No vision-specific adapter classes exist

#### Embeddings (separate path — Phase 5 target)
- `embeddings.py` calls `mlx_embeddings` directly, no adapter involvement
- No `EmbeddingsAdapter` class exists
- Returns raw OpenAI EmbeddingResponse

#### Audio (separate path — Phase 5 target)
- `audio.py` calls `mlx_audio` directly
- `WhisperAdapter` and `KokoroAdapter` exist but have no `prepare_input()`/`process_complete()`
- Only used for `post_load_configure()` hooks, not for inference pipeline

### Legacy Code (to be cleaned up in Phase 6)
- `ProtocolTranslator` in `protocol.py` — still used by `messages.py` for request translation (input side only)
- `ParseResult` in `response_processor.py` — kept for backward compat
- `StreamingProcessor` alias — kept for backward compat
- `generate_chat_completion()` legacy wrapper — uses new pipeline internally

### Files in Current Architecture
1. `services/inference.py` — TEXT_GEN via adapter pipeline (IR-based)
2. `services/vision.py` — VISION, separate path (bypasses adapters)
3. `services/embeddings.py` — EMBEDDINGS, separate path (no adapters)
4. `services/audio.py` — AUDIO, separate path (adapters exist but unused for inference)
5. `services/formatters/` — ProtocolFormatter layer (OpenAI + Anthropic)
6. `api/v1/chat.py` — routes to text or vision, uses OpenAIFormatter for text
7. `api/v1/messages.py` — uses AnthropicFormatter, TEXT_GEN only
8. `models/adapters/composable.py` — all adapter classes
9. `models/pool.py` — creates adapters at load time for all model types
10. `models/ir.py` — IR types

---

## Phase 0: Foundation - IR Types

**Goal**: Define protocol-neutral IR types that flow through the pipeline.

**Scope**: New types only, zero breaking changes.

### Files to Create

#### `mlx_server/models/ir.py`

```python
"""Intermediate Representation types for adapter pipeline."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal


# ── Input IR ──────────────────────────────────────────────────────

@dataclass
class PreparedInput:
    """Model-ready input after adapter processing."""
    prompt: str
    token_ids: list[int] | None = None
    stop_token_ids: list[int] | None = None
    pixel_values: Any | None = None  # Vision: preprocessed images
    generation_params: dict[str, Any] | None = None


# ── Output IR ─────────────────────────────────────────────────────

@dataclass
class StreamEvent:
    """Single event emitted during streaming."""
    type: Literal["content", "reasoning_content", "tool_call_delta"]
    content: str | None = None
    reasoning_content: str | None = None
    tool_call_delta: dict | None = None


class AdapterResult(ABC):
    """Base result type for all adapters."""
    finish_reason: str

    @abstractmethod
    def to_dict(self) -> dict: ...


@dataclass
class TextResult(AdapterResult):
    """Text generation result (TEXT_GEN and VISION)."""
    content: str
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str = "stop"

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
        }


@dataclass
class EmbeddingResult(AdapterResult):
    """Embedding generation result."""
    embeddings: list[list[float]]
    dimensions: int
    finish_reason: str = "stop"

    def to_dict(self) -> dict:
        return {
            "embeddings": self.embeddings,
            "dimensions": self.dimensions,
            "finish_reason": self.finish_reason,
        }


@dataclass
class AudioResult(AdapterResult):
    """TTS audio generation result."""
    audio_bytes: bytes
    sample_rate: int
    format: str
    finish_reason: str = "stop"

    def to_dict(self) -> dict:
        return {
            "audio_bytes": self.audio_bytes,
            "sample_rate": self.sample_rate,
            "format": self.format,
            "finish_reason": self.finish_reason,
        }


@dataclass
class TranscriptionResult(AdapterResult):
    """STT transcription result."""
    text: str
    segments: list[dict] | None = None
    finish_reason: str = "stop"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": self.segments,
            "finish_reason": self.finish_reason,
        }
```

### Dependencies
- **None** (pure data types)

### Test Strategy
- Unit tests for IR type instantiation, serialization, inheritance hierarchy

### Migration Notes
- Purely additive — no existing code changes
- Types available for import but not yet used

### Definition of Done
- [x] `ir.py` created with all IR types
- [x] Unit tests pass
- [x] Types importable from `mlx_manager.mlx_server.models.ir`

---

## Phase 1: StreamProcessor Refactor

**Goal**: Rename StreamingProcessor → StreamProcessor, return IR types, wire via adapter factory method.

**Scope**: response_processor.py refactor + adapter interface change

### Files to Modify

1. **`services/response_processor.py`**
   - Rename `StreamingProcessor` → `StreamProcessor`
   - `feed()` returns IR `StreamEvent` instead of old `StreamEvent`
   - `finalize()` returns IR `TextResult` instead of `ParseResult`
   - Keep `ParseResult` and alias for backward compat (deleted in Phase 6)

2. **`models/adapters/composable.py`**
   - Add `create_stream_processor()` factory method to `ModelAdapter` ABC
   - Default implementation creates `StreamProcessor(adapter=self)`
   - Family-specific overrides where needed (e.g., GLM4 starts_in_thinking)

3. **`services/inference.py`**
   - Replace `StreamingProcessor(adapter=adapter)` with `adapter.create_stream_processor()`
   - Consume IR `StreamEvent` from `feed()`
   - Consume IR `TextResult` from `finalize()`

### Dependencies
- Phase 0 (IR types)

### Test Strategy
- Unit: StreamProcessor returns correct IR types
- Unit: Adapter factory method creates properly configured processor
- Integration: Streaming inference still works end-to-end
- Regression: All existing response_processor tests pass

### Migration Notes
- `StreamingProcessor = StreamProcessor` alias for backward compat
- `ParseResult` kept temporarily with conversion helper
- Routers see same dict format (changes are internal to inference)

### Definition of Done
- [x] StreamProcessor class created (renamed from StreamingProcessor, backward-compat alias kept)
- [x] `create_stream_processor()` on all adapters (factory method on ModelAdapter ABC)
- [x] inference.py uses adapter factory (`adapter.create_stream_processor(prompt=prompt)`)
- [x] `finalize()` returns IR `TextResult` instead of `ParseResult`
- [x] All unit + E2E tests pass

---

## Phase 2: ProtocolFormatter Layer

**Goal**: Create Layer 3 (ProtocolFormatter), absorb ProtocolTranslator, wire into routers.

**Scope**: New formatter layer + router refactors

### Files to Create

1. **`services/formatters.py`**
   - `ProtocolFormatter` ABC with `format_stream_event()`, `format_response()`, `format_final_chunk()`
   - `OpenAIFormatter`: IR → OpenAI chat completion chunks/responses
   - `AnthropicFormatter`: IR → Anthropic message events/responses (absorbs ProtocolTranslator logic)

### Files to Modify

1. **`api/v1/chat.py`**
   - Create `OpenAIFormatter` per request
   - Pipe IR events through formatter for SSE chunks
   - Pipe IR result through formatter for complete responses

2. **`api/v1/messages.py`**
   - Create `AnthropicFormatter` per request
   - Replace ProtocolTranslator usage with formatter
   - Handle Anthropic-specific SSE events (message_start, content_block_delta, etc.)

3. **`services/protocol.py`**
   - Mark as deprecated (deleted in Phase 6)

### Dependencies
- Phase 1 (StreamProcessor returns IR)

### Test Strategy
- Unit: OpenAIFormatter converts IR correctly
- Unit: AnthropicFormatter matches existing ProtocolTranslator behavior
- Integration: All 140+ Anthropic tests pass unchanged
- E2E: Streaming works for both protocols

### Migration Notes
- ProtocolTranslator kept until Phase 6
- Routers gradually migrate to formatters
- AnthropicFormatter must produce byte-identical output to ProtocolTranslator

### Definition of Done
- [x] `services/formatters/` package with `ProtocolFormatter` ABC, `OpenAIFormatter`, `AnthropicFormatter`
- [x] chat.py uses `OpenAIFormatter` for streaming + non-streaming
- [x] messages.py uses `AnthropicFormatter` for streaming + non-streaming
- [x] inference.py returns IR types: `generate_chat_stream()` yields `StreamEvent`/`TextResult`, `generate_chat_complete_response()` returns `InferenceResult`
- [x] All 140+ Anthropic tests pass
- [x] All 2233 unit tests pass

### Implementation Notes
- Created as package `services/formatters/` (not single file) with `base.py`, `openai.py`, `anthropic.py`
- `generate_chat_stream` refactored from async generator to coroutine returning async generator (for `asyncio.wait_for()` compat)
- Legacy `generate_chat_completion` wrapper preserved for backward compat, uses formatters internally
- 46 dedicated formatter tests in `test_formatters.py`

---

## Phase 3: ModelAdapter PreparedInput

**Goal**: Adapters own full INPUT pipeline via `prepare_input()` returning IR.

**Scope**: Adapter interface expansion + inference.py simplification

### Files to Modify

1. **`models/adapters/composable.py`**
   - Add `prepare_input(messages, tools, ...) → PreparedInput` abstract method
   - Add `process_complete(raw_output) → AdapterResult` abstract method
   - Implement in all concrete adapters (Qwen, GLM4, Llama, Gemma, Mistral, Liquid)
   - `prepare_input()` encapsulates: `convert_messages()` + `apply_chat_template()` + stop token aggregation

2. **`services/inference.py`**
   - Replace manual template application with `adapter.prepare_input()`
   - Replace manual post-processing with `adapter.process_complete()`
   - Inference becomes thin: get model → prepare → generate → process

3. **`api/v1/messages.py`**
   - Remove manual Anthropic→OpenAI message translation
   - Adapter handles message format differences in `prepare_input()`

### Dependencies
- Phase 0 (PreparedInput IR)
- Phase 1 (StreamProcessor)
- Phase 2 (ProtocolFormatter)

### Test Strategy
- Unit: Each adapter's `prepare_input()` produces correct PreparedInput
- Unit: Each adapter's `process_complete()` produces correct TextResult
- Integration: Full inference flow using prepare_input → generate → process_complete
- Regression: All tests pass

### Migration Notes
- `apply_chat_template()` and `convert_messages()` become internal to `prepare_input()`
- External callers switch to `prepare_input()` only
- Old methods kept as private for internal use

### Definition of Done
- [x] `prepare_input()` on `ModelAdapter` ABC (encapsulates `convert_messages` + `apply_chat_template` + stop token aggregation)
- [x] `process_complete()` on `ModelAdapter` ABC (encapsulates tool parsing + thinking parsing + response cleaning)
- [x] inference.py `_prepare_generation()` uses `adapter.prepare_input()` → `PreparedInput`
- [x] inference.py `_complete_chat_ir()` and `_stream_chat_ir()` use `adapter.process_complete()` → `TextResult`
- [x] All 2233 unit tests pass

### Implementation Notes
- `prepare_input()` returns `PreparedInput(prompt=..., stop_token_ids=[...])` — fully encapsulates the input pipeline
- `process_complete()` returns `TextResult(content=..., reasoning_content=..., tool_calls=..., finish_reason=...)` — fully encapsulates output post-processing
- Both methods implemented on base `ModelAdapter` class (no need for per-family overrides)
- inference.py tests consolidated: 6 tool-related tests → 3 focused tests on `prepare_input()` contract
- TYPE_CHECKING imports for `PreparedInput` and `TextResult` to avoid circular imports

---

## Phase 4: Vision Unification

**Goal**: Vision models use text adapter pipeline, eliminating separate vision.py path.

**Scope**: Vision adapter creation + router consolidation

### Files to Create

1. **`models/adapters/vision.py`**
   - `QwenVisionAdapter(QwenAdapter)`: Extends Qwen with image preprocessing in `prepare_input()`
   - `GemmaVisionAdapter(GemmaAdapter)`: Extends Gemma with image preprocessing
   - Vision adapters share parent's tool/thinking parsers (full output pipeline)

### Files to Modify

1. **`models/adapters/registry.py`**
   - Add vision family patterns (qwen-vision, gemma-vision)
   - `create_adapter()` accepts `model_type` parameter for vision detection

2. **`models/pool.py`**
   - Pass `model_type` to `create_adapter()` for vision models

3. **`services/inference.py`**
   - Support `pixel_values` in PreparedInput (branch to mlx-vlm generate)
   - Unified path for text + vision (no separate function)

4. **`api/v1/chat.py`**
   - Remove separate `_handle_vision_request()` path
   - Single flow: extract images → pass to inference → adapter handles the rest

5. **`services/vision.py`**
   - Mark as deprecated (deleted in Phase 6)

### Dependencies
- Phase 0 (PreparedInput.pixel_values)
- Phase 3 (adapter.prepare_input pattern)

### Test Strategy
- Unit: Vision adapters extend text adapters correctly
- Unit: Vision prepare_input() handles images
- Unit: Vision adapters inherit tool/thinking parsers
- E2E: Vision models work through unified pipeline
- E2E: Vision + tool calling (new capability!)

### Migration Notes
- Old vision.py path kept as fallback during migration
- Feature flag optional for gradual rollout
- Vision E2E tests verify tool/thinking support

### Definition of Done
- [ ] Vision adapters created extending text adapters
- [ ] Pool creates vision adapters for VISION models
- [ ] Unified inference path for text + vision
- [ ] chat.py has single flow (no separate vision handler)
- [ ] Vision E2E tests pass
- [ ] Vision models support tool calling + thinking

---

## Phase 5: Embeddings & Audio Integration

**Goal**: Embeddings and Audio models use adapter abstraction.

**Scope**: Create EmbeddingsAdapter, refactor AudioAdapters with pipeline methods

### Files to Create

1. **`models/adapters/embeddings.py`**
   - `EmbeddingsAdapter`: `prepare_input(texts)`, `process_complete(embeddings) → EmbeddingResult`
   - No streaming, no tool/thinking parsers (NullToolParser, NullThinkingParser)

### Files to Modify

1. **`models/adapters/audio.py`**
   - Extend existing WhisperAdapter/KokoroAdapter with pipeline methods
   - `prepare_input()` and `process_complete()` → AudioResult / TranscriptionResult

2. **`services/embeddings.py`**
   - Use EmbeddingsAdapter for processing

3. **`services/audio.py`**
   - Use AudioAdapter for processing

4. **`api/v1/embeddings.py`**, **`api/v1/speech.py`**, **`api/v1/transcriptions.py`**
   - Use formatter for response formatting (if applicable)

### Dependencies
- Phase 0 (EmbeddingResult, AudioResult, TranscriptionResult)
- Phase 3 (adapter.prepare_input pattern)

### Test Strategy
- Unit: EmbeddingsAdapter prepare_input and process_complete
- Unit: Audio adapters with pipeline methods
- E2E: Embeddings via adapter
- E2E: Audio TTS/STT via adapter

### Migration Notes
- Embeddings and audio services simplified
- No breaking changes to API endpoints

### Definition of Done
- [ ] EmbeddingsAdapter created
- [ ] Audio adapters extended with pipeline methods
- [ ] Services use adapters
- [ ] All E2E tests pass

---

## Phase 6: Cleanup & Documentation

**Goal**: Remove deprecated code, consolidate, update documentation.

**Scope**: Delete old code, final docs

### Files to Delete
1. `services/protocol.py` (absorbed into AnthropicFormatter)
2. `services/vision.py` (unified into inference.py)
3. `ParseResult` from response_processor.py
4. `StreamingProcessor` alias

### Files to Update
1. All imports → use `models/ir.py` for IR types
2. ARCHITECTURE.md docs → mark as "Implemented"
3. Create `ADAPTER_MIGRATION_GUIDE.md`

### Dependencies
- All prior phases (0-5)

### Test Strategy
- Verify deleted code is truly unused (import tests)
- Full regression suite (unit + E2E + Anthropic 140+)

### Definition of Done
- [ ] All deprecated code deleted
- [ ] All imports updated
- [ ] Documentation reflects final state
- [ ] ALL tests pass with zero regressions

---

## Risk Assessment

### High Risk
| Risk | Impact | Mitigation |
|------|--------|------------|
| Anthropic API compat (140+ tests) | Breaking API contract | AnthropicFormatter produces identical output; keep ProtocolTranslator until Phase 6 |
| Vision model regression | Broken vision inference | Keep old vision.py as fallback; feature flag for gradual rollout |
| Metal thread affinity | GPU errors | Never touch metal.py; generation loop stays on Metal thread |

### Medium Risk
| Risk | Impact | Mitigation |
|------|--------|------------|
| Model pool loading | Adapter creation failure | Minimal pool.py changes; extensive unit tests |
| Streaming state leakage | Corrupted responses | StreamProcessor is request-scoped; factory ensures clean state |

### Low Risk
| Risk | Impact | Mitigation |
|------|--------|------------|
| IR type bugs | Wrong data shapes | Simple dataclasses; extensive unit tests |
| Documentation drift | Confusion | Updated incrementally per phase |

---

## Testing Strategy

### Per-Phase Regression
```bash
# After every phase:
pytest backend/tests/mlx_server/ -v -m "not e2e"     # Fast unit tests
pytest backend/tests/mlx_server/anthropic/ -v          # Anthropic compat (critical)
pytest backend/tests/ -v -m "e2e"                      # E2E (requires models)
```

### Performance Verification
- Streaming latency (TTFT + tok/s) — no regression
- Memory usage — model pool eviction still works
- Concurrent requests — no state leakage

---

## Rollback Procedures

Each phase is independently deployable. To rollback:
1. `git revert` the phase's commits
2. Run tests to verify clean state
3. Redeploy previous version

For high-risk phases (Phase 4 Vision), consider environment variable feature flag for gradual rollout.

---

## Estimated Complexity

| Phase | Complexity | Dependencies | Status |
|-------|------------|--------------|--------|
| Phase 0: IR Types | Low | None | COMPLETE |
| Phase 1: StreamProcessor | Medium | Phase 0 | COMPLETE |
| Phase 2: ProtocolFormatter | High | Phase 1 | COMPLETE |
| Phase 3: PreparedInput | Medium | Phases 0-2 | COMPLETE |
| Phase 4: Vision Unification | High | Phases 0-3 | TODO |
| Phase 5: Embeddings/Audio | Low | Phases 0, 3 | TODO |
| Phase 6: Cleanup | Low | All phases | TODO |
