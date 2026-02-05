# MLX Server — Architecture Blueprint

This document defines the target architecture of the `mlx_server` component: an
embedded inference server providing OpenAI and Anthropic-compatible APIs for MLX
models on Apple Silicon. It is the authoritative reference for how the component
**should** work. Deviations from this document are tracked separately as
compliance issues.

---

## 1. Principles

1. **Single inference pipeline**: All protocols (OpenAI, Anthropic, frontend UI)
   converge into one service layer. There is no per-endpoint inference logic.
2. **Adapter-driven model handling**: Every model-family-specific behavior lives
   in a `ModelAdapter` subclass. The service layer is family-agnostic.
3. **No data loss through layers**: Messages, including tool calls, tool results,
   and multimodal content, pass through every layer with full fidelity.
4. **One canonical type per concept**: Each domain concept (tool call, message,
   model type) has exactly one Pydantic model. No duplicates, no bridging.
5. **Shared infrastructure**: Cross-cutting concerns like Metal thread management
   and memory cleanup are factored into reusable utilities, not duplicated.

---

## 2. Component Map

```
mlx_server/
  main.py                 # FastAPI app factory, lifespan, health check
  config.py               # MLXServerSettings (env: MLX_SERVER_*)
  database.py             # SQLite engine for audit logs

  api/v1/                 # Protocol endpoints (thin — validate, dispatch, format)
    chat.py               # POST /v1/chat/completions (OpenAI)
    completions.py        # POST /v1/completions (OpenAI legacy)
    messages.py           # POST /v1/messages (Anthropic)
    embeddings.py         # POST /v1/embeddings
    speech.py             # POST /v1/audio/speech (TTS)
    transcriptions.py     # POST /v1/audio/transcriptions (STT)
    models.py             # GET /v1/models
    admin.py              # Pool status, preload, unload, audit

  schemas/                # One canonical Pydantic model per concept
    openai.py             # OpenAI request/response types
    anthropic.py          # Anthropic request/response types

  models/                 # Model lifecycle and family-specific behavior
    types.py              # ModelType enum (TEXT_GEN, VISION, EMBEDDINGS, AUDIO)
    detection.py          # detect_model_type() — config.json → ModelType
    pool.py               # ModelPoolManager — LRU cache with type-aware loading
    adapters/
      base.py             # ModelAdapter (abstract), DefaultAdapter
      registry.py         # detect_model_family(), get_adapter()
      qwen.py             # QwenAdapter
      glm4.py             # GLM4Adapter
      llama.py            # LlamaAdapter
      gemma.py            # GemmaAdapter
      mistral.py          # MistralAdapter

  services/               # Inference orchestration and processing
    inference.py          # Text chat/completion generation (mlx-lm)
    vision.py             # Vision generation (mlx-vlm)
    embeddings.py         # Embedding generation (mlx-embeddings)
    audio.py              # TTS / STT (mlx-audio)
    response_processor.py # Tool call + reasoning extraction (streaming & batch)
    protocol.py           # ProtocolTranslator (Anthropic ↔ OpenAI internal)
    structured_output.py  # JSON schema validation for responses
    image_processor.py    # Image URL fetching + resizing for vision
    audit.py              # Request tracking + logging
    cloud/                # Cloud backend routing (experimental)
    batching/             # Continuous batching (experimental)

  errors/                 # RFC 7807 error handling
  middleware/             # Per-endpoint timeouts
  observability/          # LogFire instrumentation
  utils/                  # Shared utilities (Metal thread runner, memory)
  benchmark/              # Token throughput benchmarks
```

---

## 3. Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer                           │
│  Thin protocol endpoints. Validates requests against    │
│  schemas, converts to internal dict format preserving   │
│  ALL fields (content, tool_calls, tool_call_id, role),  │
│  dispatches to services, formats responses.             │
├─────────────────────────────────────────────────────────┤
│                   Service Layer                         │
│  Family-agnostic inference orchestration. Delegates all │
│  model-specific behavior to adapters. Owns response     │
│  processing, protocol translation, structured output.   │
├─────────────────────────────────────────────────────────┤
│                    Model Layer                          │
│  Model pool with LRU eviction. Adapters provide         │
│  family-specific message conversion, chat template      │
│  formatting, tool prompt injection, stop tokens, and    │
│  response pattern configuration.                        │
├─────────────────────────────────────────────────────────┤
│                 Infrastructure Layer                    │
│  Configuration, persistence, error handling, Metal      │
│  thread utilities, timeouts, observability.             │
└─────────────────────────────────────────────────────────┘
```

**Layer rules:**
- Each layer communicates only with adjacent layers.
- The API layer never imports from `models/adapters/`.
- The service layer is the **only** consumer of adapters.
- Adapters never call services or API endpoints.

---

## 4. Request Flow

### 4.1 Chat Completion (text) — all protocols converge

```
POST /v1/chat/completions (OpenAI)         POST /v1/messages (Anthropic)
    │                                           │
    ▼                                           ▼
api/v1/chat.py                             api/v1/messages.py
    │                                           │
    ├── Validate against schema                 ├── Validate against schema
    ├── Convert ChatMessage → dict              ├── ProtocolTranslator
    │   preserving ALL fields:                  │   .anthropic_to_internal()
    │   role, content, tool_calls,              │
    │   tool_call_id, name                      │
    │                                           │
    └──────────────┬────────────────────────────┘
                   │
                   ▼  list[dict] + tools + params
          services/inference.py
          ::generate_chat_completion()
                   │
                   ├── pool.get_model()         → LoadedModel
                   ├── get_adapter()            → ModelAdapter
                   │
                   │  ┌─── Adapter Pipeline ───────────────────────────┐
                   │  │                                                │
                   ├──│  1. convert_messages()                         │
                   │  │     Transform tool/special roles to format     │
                   │  │     the tokenizer can handle                   │
                   │  │                                                │
                   ├──│  2. format_tools_for_prompt() + inject         │
                   │  │     OR pass tools natively to template         │
                   │  │                                                │
                   ├──│  3. apply_chat_template()                      │
                   │  │     Messages → prompt string                   │
                   │  │                                                │
                   ├──│  4. get_stop_tokens()                          │
                   │  │     + get_tool_call_stop_tokens()              │
                   │  │                                                │
                   │  └────────────────────────────────────────────────┘
                   │
                   ▼  prompt string + stop tokens
          Metal Thread (stream_generate / generate)
                   │
                   ▼  token stream
          StreamingProcessor (family-aware)
                   │
                   ├── reasoning_content (thinking tags)
                   ├── content (regular text)
                   └── tool_calls (extracted in finalize via ResponseProcessor)
                   │
                   ▼
          OpenAI-compatible response dict
```

### 4.2 Vision

```
POST /v1/chat/completions (with image content blocks)
    │
    ▼
api/v1/chat.py → detect_model_type() == VISION
    │
    ├── preprocess_images() → PIL Image list
    │
    ▼
services/vision.py::generate_vision_completion()
    │
    ├── pool.get_model() (VISION type → mlx-vlm)
    ├── Metal thread generation
    └── OpenAI-compatible response dict
```

### 4.3 Embeddings

```
POST /v1/embeddings
    │
    ▼
services/embeddings.py::generate_embeddings()
    │
    ├── pool.get_model() (EMBEDDINGS type → mlx-embeddings)
    ├── Metal thread embedding
    └── OpenAI-compatible embedding response
```

### 4.4 Audio (TTS / STT)

```
POST /v1/audio/speech        → services/audio.py::generate_speech()
POST /v1/audio/transcriptions → services/audio.py::transcribe_audio()
    │
    ├── pool.get_model() (AUDIO type → mlx-audio)
    ├── Metal thread generation
    └── Audio bytes / transcription text
```

---

## 5. Adapter Pattern (Strategy)

The `ModelAdapter` hierarchy encapsulates all family-specific behavior. The
service layer calls adapters through a uniform interface and never contains
family-specific logic.

```
ModelAdapter (abstract)
  ├── DefaultAdapter          # Fallback for unknown families
  ├── QwenAdapter             # Qwen, Qwen2, Qwen3, Qwen3-Coder
  ├── GLM4Adapter             # GLM-4, GLM-4.7-Flash
  ├── LlamaAdapter            # Llama 3.x, CodeLlama
  ├── GemmaAdapter            # Gemma 2, 3
  └── MistralAdapter          # Mistral, Mixtral
```

### 5.1 Required Adapter Contract

Every adapter that supports tool calling **must** implement all of these methods.
The base `DefaultAdapter` provides safe defaults for families that don't support
tools or reasoning.

| Method                        | Responsibility                                           |
|-------------------------------|----------------------------------------------------------|
| `family` (property)          | Return family name for pattern registry lookup            |
| `apply_chat_template()`      | Format messages → prompt string using tokenizer           |
| `get_stop_tokens()`          | Return EOS / EOT token IDs for this family               |
| `convert_messages()`         | Transform `tool` and `assistant+tool_calls` messages to   |
|                               | a format the tokenizer can handle. **Every adapter that** |
|                               | **supports tool calling must override this.**             |
| `supports_tool_calling()`    | Return `True` if family can handle tools                  |
| `format_tools_for_prompt()`  | Format tool definitions for system prompt injection       |
| `get_tool_call_stop_tokens()`| Additional stop tokens when tools are active              |
| `has_native_tool_support()`  | Whether tokenizer handles tools natively (rare)           |
| `supports_reasoning_mode()`  | Whether family produces `<think>` tags                    |

### 5.2 Message Conversion Contract

`convert_messages()` must handle these OpenAI message types:

1. **`role: "tool"`** — Tool result messages. Convert to a format the tokenizer
   can process (typically `role: "user"` with structured text).
2. **`role: "assistant"` with `tool_calls`** — Assistant requesting tool use.
   Convert `tool_calls` list to inline text representation.
3. All other roles — Pass through unchanged.

This method is called **before** tool prompt injection and chat template
application. It ensures the tokenizer never sees message structures it cannot
handle.

---

## 6. Response Processing

### 6.1 Canonical Types

There is **one** set of Pydantic models for tool calls, used by both the
response processor and API schemas:

- `schemas/openai.py::ToolCall` — The canonical tool call type
- `schemas/openai.py::FunctionCall` — Function name + arguments string

The response processor produces these types directly. No bridging or conversion.

### 6.2 ResponseProcessor (non-streaming)

Single-pass extraction and cleaning of complete model output:

1. Extract reasoning content from thinking tags
2. Extract tool calls using family-specific patterns
3. Remove matched spans from content
4. Clean special tokens

Configured per model family via `MODEL_FAMILY_PATTERNS` registry.

### 6.3 StreamingProcessor (streaming)

Token-by-token filter for the streaming path. Uses the same family-specific
pattern configuration as `ResponseProcessor`:

- **Thinking markers** → yield as `reasoning_content`
- **Tool markers** → buffer silently, extract in `finalize()`
- **Regular content** → yield as `content`
- **`finalize()`** → delegates to `ResponseProcessor` for final extraction

Both processors use the same `ModelFamilyPatterns` definitions, ensuring
consistent behavior between streaming and non-streaming paths.

---

## 7. Metal Thread Affinity

MLX Metal operations have GPU thread affinity — they must run on the same
thread that initialized the Metal context. All inference services use a
shared utility pattern:

```
    Async generator (main event loop)
         ▲
         │ Queue.get() via run_in_executor
         │
    Dedicated Thread (owns Metal context)
         │ Queue.put() per token/result
         ▼
    mlx_lm / mlx_vlm / mlx_embeddings / mlx_audio
```

The thread management boilerplate (Queue creation, thread spawn, error
propagation via Queue, cache clearing in `finally`) is factored into
`utils/metal.py::run_on_metal_thread()` to avoid duplication across the
four inference services.

---

## 8. Protocol Translation

The `ProtocolTranslator` enables a single inference pipeline to serve multiple
API protocols:

```
Anthropic Request → anthropic_to_internal() → list[dict] (OpenAI-format messages)
                                                    │
                                                    ▼
                                          generate_chat_completion()
                                                    │
                                                    ▼
Anthropic Response ← internal_to_anthropic_response() ← OpenAI result dict
```

The frontend chat endpoint (`mlx_manager/routers/chat.py`) also calls
`generate_chat_completion()` directly with the same dict message format,
converting the streaming response into UI-specific SSE events.

---

## 9. Model Lifecycle

### 9.1 Two-Phase Detection

Model identification has two independent axes:

| Axis       | Function                | Determines              | Source            |
|------------|-------------------------|--------------------------|-------------------|
| **Type**   | `detect_model_type()`   | Which inference service  | `config.json`     |
| **Family** | `detect_model_family()` | Which adapter            | Model ID string   |

Type determines the MLX library to use (mlx-lm, mlx-vlm, etc.).
Family determines formatting, stop tokens, tool handling.

### 9.2 Pool Management

`ModelPoolManager` maintains a bounded set of hot models:

- **On-demand loading**: Models loaded when first requested
- **LRU eviction**: Least-recently-used models evicted at memory/count limits
- **Type-aware loading**: Uses `ModelType` to select the correct loader
- **Preload protection**: Pinned models exempt from eviction
- **Adapter support**: LoRA adapter loading via `get_model_with_adapter()`

---

## 10. Singleton Services

Services follow the `get_*()` / `reset_*()` pattern. `reset_*()` exists
for test isolation.

| Accessor                    | Returns               | Scope     |
|-----------------------------|-----------------------|-----------|
| `get_model_pool()`          | `ModelPoolManager`    | Global    |
| `get_adapter(model_id)`     | `ModelAdapter`        | Per-call  |
| `get_translator()`          | `ProtocolTranslator`  | Global    |
| `get_processor_for_family()`| `ResponseProcessor`   | Per-family cached |
| `get_settings()`            | `MLXServerSettings`   | Global    |
| `get_router()`              | `BackendRouter`       | Global    |
| `audit_service`             | `AuditService`        | Module-level |

---

## 11. Cross-Cutting Concerns

### 11.1 Observability

- **LogFire**: Optional spans wrapping inference calls (standalone mode only)
- **Audit logging**: `AuditService` tracks requests with timing, tokens, backend
- **Structured logging**: `loguru` with DEBUG-level prompt/response dumps

### 11.2 Error Handling

- RFC 7807 `ProblemDetail` responses for structured errors
- `TimeoutHTTPException` with per-endpoint configurable timeouts
- Circuit breaker on cloud backend clients

### 11.3 Configuration

`MLXServerSettings` via pydantic-settings (`MLX_SERVER_` prefix):
- Model pool limits (memory GB, max model count)
- Per-endpoint timeout settings
- Feature flags: cloud routing, batching
- Audit log retention

---

## 12. Deployment Modes

| Mode          | Flag                    | Behavior |
|---------------|-------------------------|----------|
| **Embedded**  | `embedded_mode=True`    | Mounted at `/v1` inside MLX Manager. Shares DB. No lifespan, no LogFire. |
| **Standalone**| `embedded_mode=False`   | Independent FastAPI app. Own lifespan, LogFire, DB. |

---

## 13. Experimental Features

Behind configuration flags, not production-ready:

- **Cloud routing** (`enable_cloud_routing`): Rule-based dispatch to cloud
  backends (OpenAI, Anthropic) with circuit breaker and local fallback.
- **Continuous batching** (`enable_batching`): PagedAttention-inspired request
  scheduling for concurrent text inference.
