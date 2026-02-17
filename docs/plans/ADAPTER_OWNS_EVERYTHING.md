# Adapter Owns Everything — Refactor Plan (IMPLEMENTED)

> **Status**: Fully implemented. All tasks complete, all tests pass.
> See `mlx_server/ARCHITECTURE.md` for the authoritative architecture description.

## Motivation

The MLX Server serves models at maximum speed on Apple Silicon. The ModelAdapter should be the **single authority** for all model-specific behavior, configured once at Profile load time and acting as a zero-check pass-through at request time. Currently, tool injection flags and template options are threaded through the entire inference call chain as runtime parameters, and Profile settings (system prompt, tool injection preference) never reach the adapter at all.

## Architecture (Target State)

### Profile Lifecycle = Model + Adapter Lifecycle

```
User starts Profile
  → Pool loads Model (I/O heavy, seconds)
  → Pool creates ModelAdapter IN PARALLEL with Profile settings:
      - system_prompt (from Profile)
      - enable_tool_injection (from Profile)
      - template_options (from Profile's model_options)
      - tool_parser (from probe capabilities)
      - thinking_parser (from probe capabilities)
      - family config (from probe/detection)
  → Both stored on LoadedModel, tied to Profile lifecycle
```

### Request Flow (Zero Runtime Checks)

```
POST /v1/chat/completions {model, messages, tools, ...}
  → pool.get_model(model_id) → LoadedModel with pre-configured adapter
  → adapter.prepare_input(messages, tools, images)
      1. System prompt: if messages[0] isn't system → prepend stored default
      2. Tool handling (based on creation-time config, NOT runtime flags):
         - Native tools? → pass tools= to tokenizer template
         - Experimental injection? → inject instructions into system prompt (idempotent)
         - Neither? → ignore tools
      3. Apply chat template with stored template_options
      4. Return PreparedInput (prompt, stop_tokens, pixel_values)
  → model generates tokens
  → adapter parses output (tool calls, thinking)
  → ProtocolFormatter converts IR → wire format (OpenAI/Anthropic)
  → Response streamed to client
```

### Key Principle: Idempotent Injection

The OpenAI/Anthropic API is stateless — clients resend full conversation history. So:
- **System prompt**: check if messages already have one → inject only if missing
- **Tool instructions** (experimental): check if system message already contains them → inject only if missing
- No state tracking, no counters — just a quick check on the first message

## Gap Analysis

### GAP 1: Profile settings never reach ModelAdapter
- `create_adapter()` receives only family, tokenizer, parsers, model_id
- Profile's `default_system_prompt`, `default_enable_tool_injection`, `model_options` are stored in DB but never flow to inference

### GAP 2: `enable_prompt_injection` threaded as runtime parameter
- 13+ call sites across inference.py and composable.py
- Should be adapter creation-time config, not per-request

### GAP 3: System prompt not applied
- `default_system_prompt` stored on ExecutionProfile but never injected into messages
- No code anywhere prepends it

### GAP 4: `native_tools=False` for qwen causes XML prompt injection
- Only glm4 and liquid have `native_tools=True`
- Qwen's template already handles `tools=` natively
- XML `<tools>` blocks get injected instead

### GAP 5: Pool loads by model_id, unaware of Profile
- `pool.get_model(model_id)` has no Profile context
- Need a registry so Profile start/stop propagates settings

### GAP 6: `template_options` threaded as runtime parameter
- Should be stored on adapter at creation from Profile's model_options

## Implementation Tasks

### Phase A: Adapter Accepts Profile Settings (No Behavioral Change)

| # | Task | Depends On |
|---|------|------------|
| 1 | Add `system_prompt`, `enable_tool_injection`, `template_options` fields to ModelAdapter + create_adapter() | — |
| 2 | Profile settings registry in ModelPoolManager | 1 |
| 3 | Hook Profile start/stop to pool settings registry | 2 |

### Phase B: Move Logic Into Adapter (Behavioral Change)

| # | Task | Depends On |
|---|------|------------|
| 4 | System prompt injection in adapter.prepare_input() | 1 |
| 5 | Consolidate tool injection into adapter (use stored flag, not runtime param) | 1 |
| 6 | Remove `enable_prompt_injection` from entire inference call chain | 5 |
| 7 | Remove `template_options` from inference call chain (adapter uses stored value) | 1, 6 |
| 8 | Set `native_tools=True` for qwen family | 5 |

### Phase C: Tests

| # | Task | Depends On |
|---|------|------------|
| 9 | Update all affected tests | 6, 7, 8 |

### Dependency Graph

```
[1: Adapter fields] ──→ [2: Pool registry] ──→ [3: Profile lifecycle hook]
       │
       ├──→ [4: System prompt injection]
       │
       ├──→ [5: Tool injection consolidation] ──→ [6: Remove enable_prompt_injection]
       │                                               │
       │                                               ├──→ [7: Remove template_options param]
       │                                               │
       │    [8: Qwen native_tools] ◄────────────────────┘
       │
       └──→ [9: Update tests] (after 6, 7, 8)
```

### Execution Order

Tasks 4 + 5 can run in parallel (both depend only on 1).
Tasks 7 + 8 can run in parallel (both depend on 5/6).
Task 3 (Profile lifecycle hook) is independent of Phase B and can be done last.

Recommended: 1 → (4 || 5) → 6 → (7 || 8) → 9 → 2 → 3

## Files Affected

| File | Changes |
|------|---------|
| `mlx_server/models/adapters/composable.py` | Add fields, remove params, system prompt + tool injection logic |
| `mlx_server/models/adapters/configs.py` | `native_tools=True` for qwen |
| `mlx_server/services/inference.py` | Remove `enable_prompt_injection` + `template_options` params |
| `mlx_server/models/pool.py` | Profile settings registry, pass to create_adapter() |
| `services/server_manager.py` (or equivalent) | Register/unregister Profile settings on start/stop |
| `tests/` (multiple) | Update for removed params, new adapter config, new behaviors |
