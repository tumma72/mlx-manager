# MLX Model Manager - Architecture & Design Document

**Version:** 1.2.0-draft
**Date:** January 2026 (updated 2026-01-27)
**Status:** v1.2 Planning — MLX Unified Server
**Target Platform:** macOS with Apple Silicon (M1/M2/M3/M4)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
   - [1.5 v1.2 Architecture Evolution: MLX Unified Server](#15-v12-architecture-evolution-mlx-unified-server)
   - [1.6 3-Layer Adapter Pipeline Architecture](#16-3-layer-adapter-pipeline-architecture)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Database Design](#6-database-design)
7. [Backend API Specification](#7-backend-api-specification)
8. [Frontend Design](#8-frontend-design)
9. [Core Services Implementation](#9-core-services-implementation)
10. [Process Management](#10-process-management)
11. [launchd Integration](#11-launchd-integration)
12. [Security Considerations](#12-security-considerations)
13. [Project Structure](#13-project-structure)
14. [Implementation Phases](#14-implementation-phases)
15. [Testing Strategy](#15-testing-strategy)
16. [Deployment & Distribution](#16-deployment--distribution)
17. [Configuration Files](#17-configuration-files)
18. [External References](#18-external-references)

---

## 1. Executive Summary

### Project Name
**MLX Model Manager** (working name: `mlx-manager`)

### Purpose
A web-based application for managing MLX-optimized language models on Apple Silicon Macs. The application provides a unified interface to:
- Search and download MLX models from HuggingFace
- Configure and launch multiple `mlx-openai-server` instances
- Manage server profiles with persistent configuration
- Optionally configure models as macOS system services via launchd

### Target User
Developers and power users running local LLMs on Apple Silicon who need to manage multiple model instances for different use cases (coding, reasoning, general assistance).

### Hardware Context
The primary target is a MacBook Pro M4 Max with 128GB unified memory, capable of running 2-3 large models (30B-70B parameter range) simultaneously.

---

## 1.5 v1.2 Architecture Evolution: MLX Unified Server

> **Status:** Planning phase (2026-01-27)
> **Reference:** `.planning/research/MLX-SERVER-FEASIBILITY.md`

### v1.2 Vision

v1.2 transforms MLX Manager from a management UI for external servers into a unified inference platform with our own high-performance MLX server.

**Key Change:** Instead of wrapping external servers (mlx-openai-server, vLLM-MLX), we build our own inference engine directly on mlx-lm/mlx-vlm/mlx-embeddings.

### Why Build Our Own Server

| Factor | External Servers | Own Server |
|--------|-----------------|------------|
| Control | Limited to exposed APIs | Full stack control |
| Performance | Single-request only | Continuous batching (2-4x throughput) |
| Memory | Contiguous KV cache (60-80% waste) | Paged KV cache (<4% waste) |
| Maintenance | Multiple dependencies | Single codebase |
| Observability | Varies by server | Native LogFire integration |

### v1.2 High-Level Architecture

```
+----------------------------------------------------------------------+
|                      MLX UNIFIED SERVER                              |
+----------------------------------------------------------------------+
|  API Layer (FastAPI + uvloop + Pydantic v2)                          |
|  +------------------+  +------------------+  +--------------------+  |
|  | /v1/chat/        |  | /v1/messages     |  | /v1/embeddings     |  |
|  | completions      |  | (Anthropic)      |  |                    |  |
|  | (OpenAI)         |  |                  |  |                    |  |
|  +---------+--------+  +---------+--------+  +-----------+--------+  |
|            |                     |                       |           |
|  +---------v---------------------v-----------------------v--------+  |
|  |              3-Layer Adapter Pipeline (per request)            |  |
|  | Layer 3: ProtocolFormatter (OpenAI | Anthropic)               |  |
|  |          ↓ IR → Protocol-specific SSE chunks                   |  |
|  | Layer 2: StreamProcessor (request-scoped state)                |  |
|  |          ↓ Tokens → StreamEvent (IR)                           |  |
|  | Layer 1: ModelAdapter (model-scoped, persistent)               |  |
|  |          ↓ Messages → PreparedInput | Complete → AdapterResult |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                 Continuous Batching Scheduler                  |  |
|  |  Priority Queues | Token-level Batching | Dynamic Replacement   |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    Model Pool Manager                          |  |
|  |  Hot Models (LRU) | Memory Pressure Monitor | On-Demand Load   |  |
|  |  Each model has 1 persistent adapter instance                  |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    Paged KV Cache Manager                      |  |
|  |  Block Pool | Block Tables | Prefix Sharing | Copy-on-Write    |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    Model Adapters (per family)                 |  |
|  |  Text: Qwen | GLM4 | Llama | Gemma | Mistral | Liquid          |  |
|  |  Vision: QwenVision | GemmaVision (extend text adapters)       |  |
|  |  Embeddings | Audio (TTS/STT)                                  |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    MLX Libraries                               |  |
|  |  mlx-lm | mlx-vlm | mlx-embeddings | mlx-audio | MLX Core      |  |
|  +---------------------------------------------------------------+  |
|                                                                      |
|  +---------------------------------------------------------------+  |
|  |                    Observability (Pydantic LogFire)            |  |
|  |  Request Tracing | LLM Metrics | SQLite Spans | Alerts         |  |
|  +---------------------------------------------------------------+  |
+----------------------------------------------------------------------+
```

### v1.2 Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| HTTP Server | FastAPI + uvloop | 2-4x perf vs asyncio, async-native |
| Inference | mlx-lm + mlx-vlm + mlx-embeddings | Apple-maintained, MLX-optimized |
| Validation | Pydantic v2 (Rust core) | 5-50x faster, native FastAPI |
| Observability | Pydantic LogFire | Native instrumentation for entire stack |
| Tokenization | HuggingFace Tokenizers (Rust) | 43x faster than pure Python |

### Performance Targets

| Metric | v1.1 (mlx-openai-server) | v1.2 Target | Technique |
|--------|--------------------------|-------------|-----------|
| Single request | 200-400 tok/s | 200-400 tok/s | Same |
| 5 concurrent | 200-400 tok/s | 800-1200 tok/s | Continuous batching |
| Memory efficiency | ~40% | >90% | Paged KV cache |
| TTFT (cached) | 500-1000ms | 50-100ms | Prefix caching |

### v1.2 Phases

1. **Phase 7: Foundation** — Server skeleton, single model inference, OpenAI API
2. **Phase 8: Multi-Model** — LRU pool, vision/embedding support, model adapters
3. **Phase 9: Batching** — Continuous batching, paged KV cache, prefix caching
4. **Phase 10: Dual Protocol** — Anthropic API, cloud fallback routing
5. **Phase 11: Configuration** — UI for pool, providers, routing rules
6. **Phase 12: Hardening** — LogFire metrics, error handling, audit logging

---

## 1.6 3-Layer Adapter Pipeline Architecture

> **Status:** Active architecture (2026-02)
> **Scope:** MLX Server inference pipeline design

### Overview

The MLX Server uses a **3-layer streaming pipeline** that cleanly separates concerns between model-specific processing, request-scoped streaming state, and protocol formatting. This architecture unifies all model types (text, vision, embeddings, audio) under a single adapter abstraction while maintaining protocol independence.

### Design Principles

1. **1 Model + 1 Adapter Instance**: Each loaded model gets exactly one adapter instance, configured once at load time and reused across all requests
2. **Request-Scoped Sessions**: Each request gets its own StreamProcessor + ProtocolFormatter instances for isolated state
3. **Protocol-Neutral IR**: Intermediate representation (IR) types flow through the pipeline, eliminating protocol awareness from adapters
4. **No Duplication**: Stream events flow through the pipeline without buffering or copying
5. **Vision = Text + Multimodal Input**: Vision models are text adapters with image/video preprocessing, not a separate category

### The 3 Layers

#### Layer 1: ModelAdapter (model-scoped, persistent)

**Lifecycle**: Created once at model load time, lives in `LoadedModel.adapter`

**Responsibility**: Full pipeline ownership for both INPUT preparation and OUTPUT processing

**Key Methods**:
```python
class ModelAdapter(ABC):
    """Base adapter for all model types."""

    # INPUT: Prepare request for generation
    @abstractmethod
    def prepare_input(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs
    ) -> PreparedInput:
        """Convert messages to model-ready input (chat template + tokenization)."""
        ...

    # OUTPUT: Create request-scoped processor
    @abstractmethod
    def create_stream_processor(self) -> StreamProcessor:
        """Factory method for per-request streaming processor."""
        ...

    # OUTPUT: Non-streaming processing
    @abstractmethod
    def process_complete(self, raw_output: str) -> AdapterResult:
        """Process complete (non-streaming) generation output."""
        ...

    # Pre-computed configuration
    stop_tokens: list[str]  # Model-specific stop sequences
    stream_markers: dict[str, str]  # Patterns for streaming detection
```

**Configuration**: Adapters compose family-specific parsers and configuration:
```python
class QwenAdapter(TextAdapter):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tool_parser: ToolCallParser | None = None,
        thinking_parser: ThinkingParser | None = None,
    ):
        self.tokenizer = tokenizer
        self.tool_parser = tool_parser
        self.thinking_parser = thinking_parser
        self.stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        self.stream_markers = {
            "thinking_start": "<think>",
            "thinking_end": "</think>",
            "tool_call_start": "<tool_call>",
        }
```

**Model Type Coverage**:
- **TEXT_GEN**: Message conversion, chat template, tool/thinking parsing, response cleaning
- **VISION**: Extends text adapters with image/video preprocessing, shares output pipeline
- **EMBEDDINGS**: Simple tokenization, no chat template or parsing
- **AUDIO**: Direct audio preprocessing, format-specific output handling

**Vision Integration**:
Vision models are NOT a separate adapter category. They extend text adapters:
```python
class QwenVisionAdapter(QwenAdapter):
    """Qwen2-VL extends Qwen text adapter with vision input."""

    def prepare_input(self, messages: list[dict], **kwargs) -> PreparedInput:
        # Process images/videos first
        processed_messages = self._process_vision_content(messages)
        # Use parent text adapter for chat template
        return super().prepare_input(processed_messages, **kwargs)
```

This means vision models share:
- Tool call parsing (same `tool_parser`)
- Thinking extraction (same `thinking_parser`)
- Response cleaning (same post-processing)
- Streaming logic (same `StreamProcessor`)

#### Layer 2: StreamProcessor (request-scoped, ephemeral)

**Lifecycle**: Created per-request via `adapter.create_stream_processor()`

**Responsibility**: Incremental token-by-token parsing with stateful pattern matching

**Key Methods**:
```python
class StreamProcessor(ABC):
    """Per-request streaming state manager."""

    @abstractmethod
    def feed(self, token: str) -> StreamEvent:
        """
        Process next token, update internal state, yield IR event.

        Returns:
            StreamEvent: Protocol-neutral event (content | reasoning | tool_call_delta)
        """
        ...

    @abstractmethod
    def finalize(self) -> AdapterResult:
        """
        Finalize processing after stream ends.

        Returns:
            AdapterResult: Complete IR with extracted data
        """
        ...
```

**State Management**: Each processor maintains:
- `accumulated_text`: Full output buffer
- `current_content`: Current content chunk being built
- `current_reasoning`: Reasoning buffer (if in <think> mode)
- `tool_call_buffer`: Active tool call being parsed
- `pattern_buffers`: Partial pattern matches across token boundaries

**Example Flow**:
```python
# Request 1 (streaming)
processor1 = adapter.create_stream_processor()
for token in model.generate(...):
    event = processor1.feed(token)  # -> StreamEvent
    formatter.format_stream_event(event)  # -> SSE chunk
result = processor1.finalize()  # -> AdapterResult

# Request 2 (parallel, same model)
processor2 = adapter.create_stream_processor()  # Independent state
for token in model.generate(...):
    event = processor2.feed(token)
    formatter.format_stream_event(event)
result = processor2.finalize()
```

#### Layer 3: ProtocolFormatter (request-scoped, pluggable)

**Lifecycle**: Created per-request by router based on API endpoint

**Responsibility**: Convert protocol-neutral IR to protocol-specific responses

**Key Methods**:
```python
class ProtocolFormatter(ABC):
    """Protocol-specific response formatting."""

    @abstractmethod
    def format_stream_event(self, event: StreamEvent) -> str:
        """Convert IR stream event to protocol SSE chunk."""
        ...

    @abstractmethod
    def format_response(self, result: AdapterResult) -> dict:
        """Convert IR result to protocol-specific response."""
        ...
```

**Implementations**:

1. **OpenAIFormatter**: Formats to OpenAI API spec
   ```python
   class OpenAIFormatter(ProtocolFormatter):
       def format_stream_event(self, event: StreamEvent) -> str:
           if event.type == "content":
               chunk = {
                   "id": self.request_id,
                   "object": "chat.completion.chunk",
                   "choices": [{
                       "index": 0,
                       "delta": {"content": event.content},
                       "finish_reason": None,
                   }],
               }
               return f"data: {json.dumps(chunk)}\n\n"
           ...
   ```

2. **AnthropicFormatter**: Formats to Anthropic API spec (absorbs old ProtocolTranslator)
   ```python
   class AnthropicFormatter(ProtocolFormatter):
       def format_stream_event(self, event: StreamEvent) -> str:
           if event.type == "content":
               chunk = {
                   "type": "content_block_delta",
                   "index": 0,
                   "delta": {
                       "type": "text_delta",
                       "text": event.content,
                   },
               }
               return f"event: content_block_delta\ndata: {json.dumps(chunk)}\n\n"
           ...
   ```

### Intermediate Representation (IR) Types

Protocol-neutral data structures that flow through the pipeline:

```python
# Streaming events
@dataclass
class StreamEvent:
    """Single event emitted during streaming."""
    type: Literal["content", "reasoning_content", "tool_call_delta"]
    content: str | None = None
    reasoning_content: str | None = None
    tool_call_delta: dict | None = None

# Complete results (polymorphic hierarchy)
class AdapterResult(ABC):
    """Base result type for all adapters."""
    finish_reason: str

class TextResult(AdapterResult):
    """Text generation result (TEXT_GEN and VISION)."""
    content: str
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str = "stop"

class EmbeddingResult(AdapterResult):
    """Embedding generation result."""
    embeddings: list[list[float]]
    dimensions: int
    finish_reason: str = "stop"

class AudioResult(AdapterResult):
    """TTS audio generation result."""
    audio_bytes: bytes
    sample_rate: int
    format: str
    finish_reason: str = "stop"

class TranscriptionResult(AdapterResult):
    """STT transcription result."""
    text: str
    segments: list[dict] | None = None
    finish_reason: str = "stop"
```

### Request Flow (Streaming Text Example)

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. Router receives POST /v1/chat/completions                     │
│    - Creates OpenAIFormatter                                     │
│    - Extracts messages, tools, etc.                              │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. Get model from pool                                           │
│    - loaded_model = pool.get_model(model_id)                     │
│    - adapter = loaded_model.adapter  (created at load time)      │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. Adapter prepares input                                        │
│    - prepared = adapter.prepare_input(messages, tools)           │
│    - PreparedInput(prompt="<chat template>", token_ids=[...])    │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. Generate tokens (MLX inference)                               │
│    - for token in model.generate(prepared.token_ids):            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. Create request-scoped processor                               │
│    - stream_processor = adapter.create_stream_processor()        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. Stream processing loop                                        │
│    for token in token_stream:                                    │
│        event = stream_processor.feed(token)  # -> StreamEvent    │
│        chunk = formatter.format_stream_event(event)  # -> SSE    │
│        yield chunk                                               │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ 7. Finalize                                                      │
│    - result = stream_processor.finalize()  # -> TextResult (IR)  │
│    - final_chunk = formatter.format_response(result)  # -> SSE   │
│    - yield final_chunk                                           │
└──────────────────────────────────────────────────────────────────┘
```

### Parallel Request Handling

The architecture supports multiple concurrent requests to the same model:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoadedModel (Pool)                           │
│  ├─ model: nn.Module                                            │
│  ├─ adapter: QwenAdapter  ←── SHARED across all requests        │
│  ├─ tokenizer: PreTrainedTokenizer                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Request 1 (OpenAI)        │   │ Request 2 (Anthropic)     │
│ ├─ stream_processor1      │   │ ├─ stream_processor2      │
│ ├─ OpenAIFormatter        │   │ ├─ AnthropicFormatter     │
│ └─ Independent state      │   │ └─ Independent state      │
└───────────────────────────┘   └───────────────────────────┘
```

Requests are queued, each gets:
- Own `StreamProcessor` instance (isolated state)
- Own `ProtocolFormatter` instance (protocol-specific formatting)
- Shared `ModelAdapter` instance (immutable configuration)

### Key Changes from Previous Architecture

1. **Adapters Own the Entire Pipeline**: Previously, adapters only handled message conversion. Now they own both input preparation AND output processing via `create_stream_processor()`.

2. **StreamingProcessor Absorbed into Adapter**: The standalone `StreamingProcessor` is now created and managed by adapters as `StreamProcessor` (factory pattern).

3. **ProtocolTranslator Becomes ProtocolFormatter**: Protocol conversion logic moves from a centralized translator to pluggable formatters, absorbed into the formatter layer.

4. **Vision Is Text + Input**: Vision adapters extend text adapters instead of being separate. All vision models share text output processing (thinking, tools, cleaning).

5. **Unified Model Types**: All model types (text, vision, embeddings, audio) use the same adapter abstraction with type-specific result classes.

6. **Inference Service Becomes Orchestrator**: The inference service no longer handles model-specific logic. It becomes a thin layer that:
   - Gets model from pool
   - Calls `adapter.prepare_input()`
   - Generates tokens
   - Pipes through `stream_processor` → `protocol_formatter`

### Code Organization

```
mlx_server/
├── models/
│   ├── adapters/
│   │   ├── base.py              # ModelAdapter, StreamProcessor ABCs
│   │   ├── text.py              # TextAdapter base class
│   │   ├── vision.py            # VisionAdapter base (extends TextAdapter)
│   │   ├── embeddings.py        # EmbeddingsAdapter
│   │   ├── audio.py             # AudioAdapter (TTS/STT)
│   │   ├── families/
│   │   │   ├── qwen.py          # QwenAdapter, QwenVisionAdapter
│   │   │   ├── glm4.py          # GLM4Adapter
│   │   │   ├── llama.py         # LlamaAdapter
│   │   │   ├── gemma.py         # GemmaAdapter
│   │   │   └── ...
│   │   └── registry.py          # create_adapter() factory
│   ├── parsers/                 # Existing parser architecture (unchanged)
│   │   ├── tool_calls.py        # ToolCallParser ABC
│   │   ├── thinking.py          # ThinkingParser ABC
│   │   └── implementations/
│   └── pool.py                  # Model pool (LoadedModel.adapter field)
├── routers/
│   ├── openai.py                # Creates OpenAIFormatter
│   ├── anthropic.py             # Creates AnthropicFormatter
│   └── ...
├── protocol/
│   ├── formatters.py            # ProtocolFormatter ABC + implementations
│   └── ir.py                    # StreamEvent, AdapterResult hierarchy
└── services/
    └── inference.py             # Thin orchestrator (get, prepare, generate, pipe)
```

### What Stays Unchanged

- **Parser Architecture**: `ToolCallParser` and `ThinkingParser` abstractions remain, composed into adapters
- **Family Registry**: `create_adapter()` factory and `FAMILY_REGISTRY` pattern
- **Pool Loading**: `LoadedModel` instances created at model load time
- **Metal Thread Generation**: Queue-based threading for MLX Metal affinity
- **Probe Logic**: Model family/parser detection and storage in DB

---

## 2. Problem Statement

### Current Pain Points

1. **No Unified Model Discovery**: Finding MLX-optimized models requires manual HuggingFace browsing with no memory-fit filtering
2. **CLI Verbosity**: Starting `mlx-openai-server` requires long command lines with many parameters
3. **No Configuration Persistence**: Server configurations must be remembered and retyped
4. **No Multi-Instance Management**: Running multiple servers requires manual port management and separate terminal sessions
5. **No Visual Monitoring**: No dashboard to see running servers, memory usage, or health status
6. **Manual launchd Configuration**: Creating system services requires writing XML plist files by hand

### Gap Analysis

| Existing Tool | Capability | Missing |
|--------------|------------|---------|
| `mlx-hub` | CLI search/download | No serving, no GUI |
| `mlx-knife` | Ollama-like CLI | No multi-instance, no web UI |
| `llm-mlx` | LLM tool integration | Different paradigm |
| `mlx-openai-server` | Full OpenAI API serving | No model management |
| LM Studio | GUI + serving | Commercial license required |

**Conclusion**: No existing tool provides a free, open-source web GUI combining model discovery, download management, and multi-instance server orchestration.

---

## 3. Solution Overview

### Core Features

1. **Model Browser**
   - Search `mlx-community` HuggingFace organization
   - Filter by model size vs. available memory
   - Show model metadata (architecture, quantization, downloads)
   - One-click download with progress tracking

2. **Server Profile Manager**
   - Create named profiles (e.g., "Coding", "Thinking", "Assistant")
   - Configure all mlx-openai-server parameters
   - Save/load configurations from SQLite database
   - Validate port conflicts before saving

3. **Instance Controller**
   - Start/stop server instances from UI
   - Real-time health monitoring
   - Memory usage display
   - Log viewing

4. **System Service Integration**
   - Generate launchd plist files
   - Install/uninstall system services
   - Configure auto-start on login

### Non-Goals (v1.0)

- Model fine-tuning or training
- Chat interface (use Open WebUI or similar)
- Windows/Linux support
- Model conversion from non-MLX formats

---

## 4. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Browser (localhost:5173)                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    SvelteKit Frontend (Svelte 5)                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐   │  │
│  │  │ ModelBrowser│  │ProfileEditor│  │ ServerDashboard          │   │  │
│  │  │ - Search    │  │ - Create    │  │ - Status                 │   │  │
│  │  │ - Download  │  │ - Edit      │  │ - Start/Stop             │   │  │
│  │  │ - Progress  │  │ - Delete    │  │ - Logs                   │   │  │
│  │  └─────────────┘  └─────────────┘  └──────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/REST + Server-Sent Events
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Backend API (localhost:10242)                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Application (Python)                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │  │
│  │  │ /api/models     │  │ /api/profiles   │  │ /api/servers     │   │  │
│  │  │ - search        │  │ - CRUD          │  │ - start/stop     │   │  │
│  │  │ - download      │  │ - validate      │  │ - status         │   │  │
│  │  │ - list local    │  │                 │  │ - logs (SSE)     │   │  │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────┘   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │  │
│  │  │ /api/system     │  │ HuggingFaceAPI  │  │ ProcessManager   │   │  │
│  │  │ - memory        │  │ (huggingface_hub│  │ (subprocess)     │   │  │
│  │  │ - launchd       │  │  library)       │  │                  │   │  │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                          SQLite Database                                │
│                    (~/.mlx-manager/mlx-manager.db)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ subprocess.Popen
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    mlx-openai-server Instances                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ :10240          │  │ :10241          │  │ :10242                  │  │
│  │ Profile: Coding │  │ Profile: Think  │  │ Profile: Assistant      │  │
│  │ Qwen2.5-Coder   │  │ GLM4-MoE        │  │ Qwen3-8B               │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────────────┐
│ Frontend │────▶│ Backend  │────▶│ HF Hub   │────▶│ ~/.cache/        │
│ Search   │     │ API      │     │ API      │     │ huggingface/hub  │
└──────────┘     └──────────┘     └──────────┘     └──────────────────┘
                      │
                      ▼
                ┌──────────┐
                │ SQLite   │ (profiles, settings)
                └──────────┘
                      │
                      ▼
                ┌──────────┐     ┌──────────────────┐
                │ Process  │────▶│ mlx-openai-server│
                │ Manager  │     │ processes        │
                └──────────┘     └──────────────────┘
```

---

## 5. Technology Stack

### Frontend

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Framework | SvelteKit | 2.x | Modern, fast, excellent DX |
| UI Library | Svelte | 5.x | Runes API, fine-grained reactivity |
| Styling | Tailwind CSS | 4.x | Utility-first, rapid prototyping |
| Components | shadcn-svelte | latest | Accessible, customizable |
| Icons | Lucide Svelte | latest | Comprehensive icon set |
| HTTP Client | Native fetch | - | Built into SvelteKit |
| State | Svelte 5 runes | - | $state, $derived, $effect |

### Backend

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Framework | FastAPI | 0.115.x | Async, automatic OpenAPI docs |
| Python | Python | 3.11+ | Required for MLX compatibility |
| ORM | SQLModel | 0.0.22+ | Pydantic + SQLAlchemy integration |
| Database | SQLite | 3.x | Simple, no server needed |
| HF Integration | huggingface_hub | 0.27.x | Official HuggingFace library |
| Process Mgmt | psutil | 6.x | Cross-platform process utilities |
| Validation | Pydantic | 2.x | Data validation (via FastAPI) |

### External Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| mlx-openai-server | Model serving | `pip install mlx-openai-server` |
| mlx-lm | Model management | `pip install mlx-lm` |

### Version Pinning (pyproject.toml)

```toml
[project]
name = "mlx-manager"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "sqlmodel>=0.0.22",
    "huggingface-hub>=0.27.0",
    "psutil>=6.0.0",
    "aiosqlite>=0.20.0",
    "httpx>=0.28.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
]
```

---

## 6. Database Design

### SQLite Schema

Location: `~/.mlx-manager/mlx-manager.db`

```sql
-- Server profiles store configuration for mlx-openai-server instances
CREATE TABLE server_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    
    -- Model configuration
    model_path TEXT NOT NULL,  -- e.g., "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
    model_type TEXT NOT NULL DEFAULT 'lm',  -- lm, multimodal, whisper, embeddings, image-generation, image-edit
    
    -- Server configuration
    port INTEGER UNIQUE NOT NULL,
    host TEXT DEFAULT '127.0.0.1',
    
    -- Model parameters
    context_length INTEGER,  -- NULL means use model default
    max_concurrency INTEGER DEFAULT 1,
    queue_timeout INTEGER DEFAULT 300,
    queue_size INTEGER DEFAULT 100,
    
    -- Parser configuration
    tool_call_parser TEXT,  -- qwen3, glm4_moe, harmony, minimax, etc.
    reasoning_parser TEXT,
    enable_auto_tool_choice BOOLEAN DEFAULT FALSE,
    
    -- Advanced options
    trust_remote_code BOOLEAN DEFAULT FALSE,
    chat_template_file TEXT,
    
    -- Logging
    log_level TEXT DEFAULT 'INFO',
    log_file TEXT,
    no_log_file BOOLEAN DEFAULT FALSE,
    
    -- System service
    auto_start BOOLEAN DEFAULT FALSE,
    launchd_installed BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Track running instances
CREATE TABLE running_instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES server_profiles(id) ON DELETE CASCADE,
    pid INTEGER NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    health_status TEXT DEFAULT 'starting',  -- starting, healthy, unhealthy, stopped
    last_health_check TIMESTAMP,
    
    UNIQUE(profile_id)  -- Only one instance per profile
);

-- Downloaded models cache (mirrors HuggingFace cache)
CREATE TABLE downloaded_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT UNIQUE NOT NULL,  -- e.g., "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
    local_path TEXT NOT NULL,
    size_bytes INTEGER,
    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);

-- Application settings
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Default settings
INSERT INTO settings (key, value) VALUES
    ('default_port_start', '10240'),
    ('huggingface_cache_path', '~/.cache/huggingface/hub'),
    ('max_memory_percent', '80'),
    ('health_check_interval', '30');

-- Indexes for performance
CREATE INDEX idx_profiles_port ON server_profiles(port);
CREATE INDEX idx_profiles_auto_start ON server_profiles(auto_start);
CREATE INDEX idx_instances_profile ON running_instances(profile_id);
CREATE INDEX idx_models_model_id ON downloaded_models(model_id);
```

### SQLModel Classes

```python
# backend/app/models.py
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class ServerProfileBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None
    model_path: str
    model_type: str = Field(default="lm")
    port: int = Field(unique=True)
    host: str = Field(default="127.0.0.1")
    context_length: Optional[int] = None
    max_concurrency: int = Field(default=1)
    queue_timeout: int = Field(default=300)
    queue_size: int = Field(default=100)
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    enable_auto_tool_choice: bool = Field(default=False)
    trust_remote_code: bool = Field(default=False)
    chat_template_file: Optional[str] = None
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = None
    no_log_file: bool = Field(default=False)
    auto_start: bool = Field(default=False)

class ServerProfile(ServerProfileBase, table=True):
    __tablename__ = "server_profiles"
    id: Optional[int] = Field(default=None, primary_key=True)
    launchd_installed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ServerProfileCreate(ServerProfileBase):
    pass

class ServerProfileUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    port: Optional[int] = None
    # ... all other fields as Optional

class RunningInstance(SQLModel, table=True):
    __tablename__ = "running_instances"
    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="server_profiles.id", unique=True)
    pid: int
    started_at: datetime = Field(default_factory=datetime.utcnow)
    health_status: str = Field(default="starting")
    last_health_check: Optional[datetime] = None

class DownloadedModel(SQLModel, table=True):
    __tablename__ = "downloaded_models"
    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: str = Field(unique=True, index=True)
    local_path: str
    size_bytes: Optional[int] = None
    downloaded_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None

class Setting(SQLModel, table=True):
    __tablename__ = "settings"
    key: str = Field(primary_key=True)
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

---

## 7. Backend API Specification

### Base URL
`http://localhost:10242/api`

### API Endpoints

#### Models API (`/api/models`)

```yaml
GET /api/models/search:
  description: Search MLX models on HuggingFace
  parameters:
    - name: query
      in: query
      type: string
      required: true
      example: "Qwen coder"
    - name: max_size_gb
      in: query
      type: number
      required: false
      description: Filter models smaller than this size
    - name: limit
      in: query
      type: integer
      default: 20
  response:
    200:
      content:
        application/json:
          schema:
            type: array
            items:
              type: object
              properties:
                model_id: string
                author: string
                downloads: integer
                likes: integer
                estimated_size_gb: number
                tags: array[string]
                is_downloaded: boolean

GET /api/models/local:
  description: List locally downloaded MLX models
  response:
    200:
      content:
        application/json:
          schema:
            type: array
            items:
              type: object
              properties:
                model_id: string
                local_path: string
                size_bytes: integer
                downloaded_at: string (ISO 8601)

POST /api/models/download:
  description: Download a model from HuggingFace
  requestBody:
    content:
      application/json:
        schema:
          type: object
          required: [model_id]
          properties:
            model_id: string
  response:
    202:
      description: Download started
      content:
        application/json:
          schema:
            type: object
            properties:
              task_id: string

GET /api/models/download/{task_id}/progress:
  description: SSE endpoint for download progress
  produces: text/event-stream
  response:
    200:
      content:
        text/event-stream:
          schema:
            type: object
            properties:
              status: string  # downloading, completed, failed
              progress: number  # 0-100
              downloaded_bytes: integer
              total_bytes: integer
              speed_mbps: number

DELETE /api/models/{model_id}:
  description: Delete a local model
  response:
    204: No Content
```

#### Profiles API (`/api/profiles`)

```yaml
GET /api/profiles:
  description: List all server profiles
  response:
    200:
      content:
        application/json:
          schema:
            type: array
            items: ServerProfile

GET /api/profiles/{id}:
  description: Get a specific profile
  response:
    200:
      content:
        application/json:
          schema: ServerProfile

POST /api/profiles:
  description: Create a new profile
  requestBody:
    content:
      application/json:
        schema: ServerProfileCreate
  response:
    201:
      content:
        application/json:
          schema: ServerProfile

PUT /api/profiles/{id}:
  description: Update a profile
  requestBody:
    content:
      application/json:
        schema: ServerProfileUpdate
  response:
    200:
      content:
        application/json:
          schema: ServerProfile

DELETE /api/profiles/{id}:
  description: Delete a profile
  response:
    204: No Content

POST /api/profiles/{id}/duplicate:
  description: Duplicate a profile with a new name
  requestBody:
    content:
      application/json:
        schema:
          type: object
          required: [new_name]
          properties:
            new_name: string
  response:
    201:
      content:
        application/json:
          schema: ServerProfile

GET /api/profiles/next-port:
  description: Get the next available port
  response:
    200:
      content:
        application/json:
          schema:
            type: object
            properties:
              port: integer
```

#### Servers API (`/api/servers`)

```yaml
GET /api/servers:
  description: List all running server instances
  response:
    200:
      content:
        application/json:
          schema:
            type: array
            items:
              type: object
              properties:
                profile_id: integer
                profile_name: string
                pid: integer
                port: integer
                health_status: string
                uptime_seconds: integer
                memory_mb: number

POST /api/servers/{profile_id}/start:
  description: Start a server for a profile
  response:
    200:
      content:
        application/json:
          schema:
            type: object
            properties:
              pid: integer
              port: integer

POST /api/servers/{profile_id}/stop:
  description: Stop a running server
  response:
    200:
      content:
        application/json:
          schema:
            type: object
            properties:
              stopped: boolean

POST /api/servers/{profile_id}/restart:
  description: Restart a server
  response:
    200:
      content:
        application/json:
          schema:
            type: object
            properties:
              pid: integer

GET /api/servers/{profile_id}/health:
  description: Check server health
  response:
    200:
      content:
        application/json:
          schema:
            type: object
            properties:
              status: string  # healthy, unhealthy, starting
              response_time_ms: number
              model_loaded: boolean

GET /api/servers/{profile_id}/logs:
  description: SSE endpoint for live logs
  produces: text/event-stream
  parameters:
    - name: lines
      in: query
      type: integer
      default: 100
      description: Number of historical lines to fetch first
```

#### System API (`/api/system`)

```yaml
GET /api/system/memory:
  description: Get system memory information
  response:
    200:
      content:
        application/json:
          schema:
            type: object
            properties:
              total_gb: number
              available_gb: number
              used_gb: number
              percent_used: number
              mlx_recommended_gb: number  # 80% of total

GET /api/system/info:
  description: Get system information
  response:
    200:
      content:
        application/json:
          schema:
            type: object
            properties:
              os_version: string
              chip: string  # "Apple M4 Max"
              memory_gb: integer
              python_version: string
              mlx_version: string
              mlx_openai_server_version: string

POST /api/system/launchd/install/{profile_id}:
  description: Install profile as launchd service
  response:
    200:
      content:
        application/json:
          schema:
            type: object
            properties:
              plist_path: string
              label: string

POST /api/system/launchd/uninstall/{profile_id}:
  description: Uninstall launchd service
  response:
    204: No Content

GET /api/system/launchd/status/{profile_id}:
  description: Get launchd service status
  response:
    200:
      content:
        application/json:
          schema:
            type: object
            properties:
              installed: boolean
              running: boolean
              label: string
```

---

## 8. Frontend Design

### Route Structure

```
src/routes/
├── +layout.svelte           # App shell with navigation
├── +page.svelte             # Dashboard (redirect to /servers)
├── models/
│   ├── +page.svelte         # Model browser & download
│   └── [model_id]/
│       └── +page.svelte     # Model details
├── profiles/
│   ├── +page.svelte         # Profile list
│   ├── new/
│   │   └── +page.svelte     # Create profile form
│   └── [id]/
│       └── +page.svelte     # Edit profile
├── servers/
│   ├── +page.svelte         # Server dashboard
│   └── [id]/
│       └── +page.svelte     # Server detail view
└── settings/
    └── +page.svelte         # Application settings
```

### Component Library

```
src/lib/components/
├── ui/                      # shadcn-svelte components
│   ├── button/
│   ├── card/
│   ├── dialog/
│   ├── input/
│   ├── select/
│   ├── table/
│   ├── toast/
│   └── ...
├── models/
│   ├── ModelCard.svelte     # Model display card
│   ├── ModelSearch.svelte   # Search input with filters
│   ├── DownloadProgress.svelte
│   └── ModelSizeIndicator.svelte
├── profiles/
│   ├── ProfileForm.svelte   # Create/edit form
│   ├── ProfileCard.svelte   # Profile summary card
│   └── ParserSelector.svelte
├── servers/
│   ├── ServerCard.svelte    # Server status card
│   ├── ServerControls.svelte # Start/stop/restart buttons
│   ├── HealthIndicator.svelte
│   ├── MemoryUsage.svelte
│   └── LogViewer.svelte     # Live log display
└── layout/
    ├── Navbar.svelte
    ├── Sidebar.svelte
    └── SystemStatus.svelte  # Memory/chip info
```

### State Management

Using Svelte 5 runes for reactive state:

```typescript
// src/lib/stores/servers.svelte.ts
import { browser } from '$app/environment';

interface ServerState {
  profile_id: number;
  profile_name: string;
  pid: number;
  port: number;
  health_status: 'starting' | 'healthy' | 'unhealthy' | 'stopped';
  memory_mb: number;
}

class ServerStore {
  servers = $state<ServerState[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);
  
  async refresh() {
    this.loading = true;
    try {
      const res = await fetch('/api/servers');
      this.servers = await res.json();
      this.error = null;
    } catch (e) {
      this.error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      this.loading = false;
    }
  }
  
  async start(profileId: number) {
    const res = await fetch(`/api/servers/${profileId}/start`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to start server');
    await this.refresh();
  }
  
  async stop(profileId: number) {
    const res = await fetch(`/api/servers/${profileId}/stop`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to stop server');
    await this.refresh();
  }
}

export const serverStore = new ServerStore();
```

### UI Wireframes

#### Server Dashboard (Main View)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MLX Model Manager                                [Memory: 45/128 GB]   │
├──────────┬──────────────────────────────────────────────────────────────┤
│          │                                                              │
│ 📊 Dash  │  Running Servers                                 [Refresh]  │
│ ───────  │  ┌────────────────────────────────────────────────────────┐  │
│ 📦 Models│  │ 🟢 Coding                              [Stop] [Restart]│  │
│ ⚙️ Profile│  │    Port: 10240 │ PID: 12345 │ Memory: 35.2 GB         │  │
│ 🖥️ Servers│  │    Model: mlx-community/Qwen2.5-Coder-32B-Instruct-4bit│  │
│ ⚙️ Settings│  │    Uptime: 2h 34m │ Health: ✓ 45ms                    │  │
│          │  └────────────────────────────────────────────────────────┘  │
│          │  ┌────────────────────────────────────────────────────────┐  │
│          │  │ 🟢 Thinking                            [Stop] [Restart]│  │
│          │  │    Port: 10241 │ PID: 12346 │ Memory: 8.5 GB           │  │
│          │  │    Model: mlx-community/GLM4-MoE-9B-Instruct-4bit      │  │
│          │  │    Uptime: 1h 12m │ Health: ✓ 32ms                     │  │
│          │  └────────────────────────────────────────────────────────┘  │
│          │  ┌────────────────────────────────────────────────────────┐  │
│          │  │ ⚪ Assistant                                    [Start]│  │
│          │  │    Port: 10242 │ Not running                           │  │
│          │  │    Model: mlx-community/Qwen3-8B-Instruct-4bit         │  │
│          │  └────────────────────────────────────────────────────────┘  │
│          │                                                              │
│          │  Stopped Profiles                         [+ New Profile]   │
│          │  ┌─────────────────────────────────────────────────────────┐│
│          │  │ No stopped profiles                                     ││
│          │  └─────────────────────────────────────────────────────────┘│
└──────────┴──────────────────────────────────────────────────────────────┘
```

#### Model Browser

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Model Browser                                    [Memory: 45/128 GB]   │
├──────────┬──────────────────────────────────────────────────────────────┤
│          │  ┌────────────────────────────────────────────────────────┐  │
│ 📊 Dash  │  │ 🔍 Search mlx-community models...                      │  │
│ 📦 Models│  └────────────────────────────────────────────────────────┘  │
│ ───────  │                                                              │
│ ⚙️ Profile│  Filters: [x] Fits in memory (<102GB)  [ ] Downloaded only  │
│ 🖥️ Servers│                                                              │
│ ⚙️ Settings│  Results (847 models)                     Sort: [Downloads ▼]│
│          │  ┌────────────────────────────────────────────────────────┐  │
│          │  │ mlx-community/Qwen2.5-Coder-32B-Instruct-4bit          │  │
│          │  │ ⬇️ 15.2k │ ❤️ 234 │ Size: ~18GB │ ✓ Downloaded          │  │
│          │  │ Tags: [code] [4bit] [qwen]              [Use] [Delete] │  │
│          │  └────────────────────────────────────────────────────────┘  │
│          │  ┌────────────────────────────────────────────────────────┐  │
│          │  │ mlx-community/Qwen3-30B-Instruct-4bit                  │  │
│          │  │ ⬇️ 8.7k │ ❤️ 156 │ Size: ~17GB │ Not downloaded         │  │
│          │  │ Tags: [chat] [4bit] [qwen]                  [Download] │  │
│          │  └────────────────────────────────────────────────────────┘  │
│          │  ┌────────────────────────────────────────────────────────┐  │
│          │  │ mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit           │  │
│          │  │ ⬇️ 5.2k │ ❤️ 89 │ Size: ~5GB │ Not downloaded          │  │
│          │  │ Tags: [reasoning] [4bit]                    [Download] │  │
│          │  └────────────────────────────────────────────────────────┘  │
│          │                                                              │
│          │  [Load More...]                                              │
└──────────┴──────────────────────────────────────────────────────────────┘
```

#### Profile Editor

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Create Server Profile                            [Memory: 45/128 GB]   │
├──────────┬──────────────────────────────────────────────────────────────┤
│          │  Profile Configuration                                       │
│ 📊 Dash  │  ┌────────────────────────────────────────────────────────┐  │
│ 📦 Models│  │ Name *                                                 │  │
│ ⚙️ Profile│  │ [Coding Assistant                                    ]│  │
│ ───────  │  │                                                        │  │
│ 🖥️ Servers│  │ Description                                           │  │
│ ⚙️ Settings│  │ [Primary coding model for Zed and Goose             ]│  │
│          │  │                                                        │  │
│          │  │ Model *                          [Browse...]           │  │
│          │  │ [mlx-community/Qwen2.5-Coder-32B-Instruct-4bit       ]│  │
│          │  │                                                        │  │
│          │  │ Model Type *                     Port *                │  │
│          │  │ [lm                        ▼]   [10240              ]  │  │
│          │  └────────────────────────────────────────────────────────┘  │
│          │                                                              │
│          │  Advanced Options                              [Expand ▼]   │
│          │  ┌────────────────────────────────────────────────────────┐  │
│          │  │ Context Length        Max Concurrency                  │  │
│          │  │ [32768            ]   [1                          ]    │  │
│          │  │                                                        │  │
│          │  │ Tool Call Parser      Reasoning Parser                 │  │
│          │  │ [qwen3            ▼]  [qwen3                      ▼]   │  │
│          │  │                                                        │  │
│          │  │ [x] Enable Auto Tool Choice                            │  │
│          │  │ [ ] Trust Remote Code                                  │  │
│          │  │ [ ] Start on Login (launchd)                           │  │
│          │  └────────────────────────────────────────────────────────┘  │
│          │                                                              │
│          │  [Cancel]                                    [Save Profile] │
└──────────┴──────────────────────────────────────────────────────────────┘
```

---

## 9. Core Services Implementation

### HuggingFace Client Service

```python
# backend/app/services/hf_client.py
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
import os

class HuggingFaceClient:
    """Service for interacting with HuggingFace Hub."""
    
    def __init__(self):
        self.api = HfApi()
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    async def search_mlx_models(
        self,
        query: str,
        max_size_gb: Optional[float] = None,
        limit: int = 20
    ) -> list[dict]:
        """Search for MLX models in mlx-community organization."""
        
        # Run in executor since huggingface_hub is sync
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(
            None,
            lambda: list(self.api.list_models(
                search=query,
                author="mlx-community",
                sort="downloads",
                direction=-1,
                limit=limit * 2  # Fetch extra for filtering
            ))
        )
        
        results = []
        for model in models:
            estimated_size = await self._estimate_model_size(model.id)
            
            # Filter by size if specified
            if max_size_gb and estimated_size > max_size_gb:
                continue
            
            results.append({
                "model_id": model.id,
                "author": model.author,
                "downloads": model.downloads,
                "likes": model.likes,
                "estimated_size_gb": round(estimated_size, 2),
                "tags": model.tags,
                "is_downloaded": self._is_downloaded(model.id),
                "last_modified": model.last_modified.isoformat() if model.last_modified else None
            })
            
            if len(results) >= limit:
                break
        
        return results
    
    async def _estimate_model_size(self, model_id: str) -> float:
        """Estimate model size in GB based on safetensors files."""
        try:
            loop = asyncio.get_event_loop()
            files = await loop.run_in_executor(
                None,
                lambda: self.api.list_repo_files(model_id)
            )
            
            total_bytes = 0
            for filename in files:
                if filename.endswith(('.safetensors', '.bin', '.gguf')):
                    try:
                        info = await loop.run_in_executor(
                            None,
                            lambda f=filename: self.api.hf_file_info(model_id, f)
                        )
                        total_bytes += info.size
                    except Exception:
                        continue
            
            # Add 20% overhead for KV cache and runtime
            return (total_bytes / 1e9) * 1.2
        except Exception:
            return 0.0
    
    def _is_downloaded(self, model_id: str) -> bool:
        """Check if model is in local cache."""
        # HuggingFace cache structure: models--org--repo
        cache_name = f"models--{model_id.replace('/', '--')}"
        model_path = self.cache_dir / cache_name
        
        # Check for snapshots directory which indicates complete download
        snapshots_dir = model_path / "snapshots"
        return snapshots_dir.exists() and any(snapshots_dir.iterdir())
    
    def get_local_path(self, model_id: str) -> Optional[str]:
        """Get the local path for a downloaded model."""
        cache_name = f"models--{model_id.replace('/', '--')}"
        model_path = self.cache_dir / cache_name / "snapshots"
        
        if model_path.exists():
            # Get the latest snapshot
            snapshots = sorted(model_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                return str(snapshots[0])
        return None
    
    async def download_model(
        self,
        model_id: str,
        progress_callback: Optional[callable] = None
    ) -> AsyncGenerator[dict, None]:
        """Download a model with progress updates."""
        
        loop = asyncio.get_event_loop()
        
        # Get total size first
        total_size = await self._estimate_model_size(model_id)
        
        yield {
            "status": "starting",
            "model_id": model_id,
            "total_size_gb": total_size
        }
        
        try:
            # Use snapshot_download for full model
            local_dir = await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=model_id,
                    local_dir_use_symlinks=True,
                    resume_download=True
                )
            )
            
            yield {
                "status": "completed",
                "model_id": model_id,
                "local_path": local_dir,
                "progress": 100
            }
            
        except Exception as e:
            yield {
                "status": "failed",
                "model_id": model_id,
                "error": str(e)
            }
    
    def list_local_models(self) -> list[dict]:
        """List all locally downloaded MLX models."""
        models = []
        
        if not self.cache_dir.exists():
            return models
        
        for item in self.cache_dir.iterdir():
            if item.name.startswith("models--mlx-community--"):
                model_id = item.name.replace("models--", "").replace("--", "/")
                local_path = self.get_local_path(model_id)
                
                if local_path:
                    size_bytes = sum(
                        f.stat().st_size 
                        for f in Path(local_path).rglob("*") 
                        if f.is_file()
                    )
                    
                    models.append({
                        "model_id": model_id,
                        "local_path": local_path,
                        "size_bytes": size_bytes,
                        "size_gb": round(size_bytes / 1e9, 2)
                    })
        
        return models
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model from local cache."""
        cache_name = f"models--{model_id.replace('/', '--')}"
        model_path = self.cache_dir / cache_name
        
        if model_path.exists():
            import shutil
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: shutil.rmtree(model_path))
            return True
        return False


# Singleton instance
hf_client = HuggingFaceClient()
```

### Server Manager Service

```python
# backend/app/services/server_manager.py
import asyncio
import subprocess
import signal
import sys
from pathlib import Path
from typing import Optional
import psutil
import httpx

from app.models import ServerProfile, RunningInstance
from app.database import get_session

class ServerManager:
    """Manages mlx-openai-server processes."""
    
    def __init__(self):
        self.processes: dict[int, subprocess.Popen] = {}  # profile_id -> process
        self._health_check_interval = 30
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def start_server(self, profile: ServerProfile) -> int:
        """Start an mlx-openai-server instance for the given profile."""
        
        # Check if already running
        if profile.id in self.processes:
            proc = self.processes[profile.id]
            if proc.poll() is None:  # Still running
                raise RuntimeError(f"Server for profile {profile.name} is already running")
        
        # Build command
        cmd = self._build_command(profile)
        
        # Start process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
        )
        
        self.processes[profile.id] = proc
        
        # Wait briefly for startup
        await asyncio.sleep(2)
        
        # Check if process is still alive
        if proc.poll() is not None:
            # Process exited
            stdout, _ = proc.communicate()
            raise RuntimeError(f"Server failed to start: {stdout}")
        
        # Record in database
        async with get_session() as session:
            instance = RunningInstance(
                profile_id=profile.id,
                pid=proc.pid,
                health_status="starting"
            )
            session.add(instance)
            await session.commit()
        
        return proc.pid
    
    def _build_command(self, profile: ServerProfile) -> list[str]:
        """Build the mlx-openai-server command from profile."""
        cmd = [
            sys.executable, "-m", "mlx_openai_server.main",
            "--model-path", profile.model_path,
            "--model-type", profile.model_type,
            "--port", str(profile.port),
            "--host", profile.host,
            "--max-concurrency", str(profile.max_concurrency),
            "--queue-timeout", str(profile.queue_timeout),
            "--queue-size", str(profile.queue_size),
        ]
        
        if profile.context_length:
            cmd.extend(["--context-length", str(profile.context_length)])
        
        if profile.tool_call_parser:
            cmd.extend(["--tool-call-parser", profile.tool_call_parser])
        
        if profile.reasoning_parser:
            cmd.extend(["--reasoning-parser", profile.reasoning_parser])
        
        if profile.enable_auto_tool_choice:
            cmd.append("--enable-auto-tool-choice")
        
        if profile.trust_remote_code:
            cmd.append("--trust-remote-code")
        
        if profile.chat_template_file:
            cmd.extend(["--chat-template-file", profile.chat_template_file])
        
        cmd.extend(["--log-level", profile.log_level])
        
        if profile.no_log_file:
            cmd.append("--no-log-file")
        elif profile.log_file:
            cmd.extend(["--log-file", profile.log_file])
        
        return cmd
    
    async def stop_server(self, profile_id: int, force: bool = False) -> bool:
        """Stop a running server."""
        if profile_id not in self.processes:
            return False
        
        proc = self.processes[profile_id]
        
        if proc.poll() is not None:
            # Already stopped
            del self.processes[profile_id]
            return True
        
        # Send SIGTERM (graceful) or SIGKILL (force)
        sig = signal.SIGKILL if force else signal.SIGTERM
        proc.send_signal(sig)
        
        # Wait for process to exit
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        
        del self.processes[profile_id]
        
        # Remove from database
        async with get_session() as session:
            from sqlmodel import select
            stmt = select(RunningInstance).where(RunningInstance.profile_id == profile_id)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            if instance:
                await session.delete(instance)
                await session.commit()
        
        return True
    
    async def check_health(self, profile: ServerProfile) -> dict:
        """Check health of a running server."""
        url = f"http://{profile.host}:{profile.port}/health"
        
        try:
            async with httpx.AsyncClient() as client:
                start = asyncio.get_event_loop().time()
                response = await client.get(url, timeout=5.0)
                elapsed = (asyncio.get_event_loop().time() - start) * 1000
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "response_time_ms": round(elapsed, 2),
                        "model_loaded": True
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "response_time_ms": round(elapsed, 2),
                        "error": f"HTTP {response.status_code}"
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_server_stats(self, profile_id: int) -> Optional[dict]:
        """Get memory and CPU stats for a running server."""
        if profile_id not in self.processes:
            return None
        
        proc = self.processes[profile_id]
        if proc.poll() is not None:
            return None
        
        try:
            p = psutil.Process(proc.pid)
            memory_info = p.memory_info()
            
            return {
                "pid": proc.pid,
                "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": p.cpu_percent(),
                "status": p.status(),
                "create_time": p.create_time()
            }
        except psutil.NoSuchProcess:
            return None
    
    async def get_logs(self, profile_id: int, lines: int = 100):
        """Generator that yields log lines from a running server."""
        if profile_id not in self.processes:
            return
        
        proc = self.processes[profile_id]
        if proc.stdout is None:
            return
        
        # Read available output
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                await asyncio.sleep(0.1)
                continue
            yield line.strip()
    
    def get_all_running(self) -> list[dict]:
        """Get info about all running servers."""
        running = []
        
        for profile_id, proc in list(self.processes.items()):
            if proc.poll() is not None:
                # Process has exited, clean up
                del self.processes[profile_id]
                continue
            
            stats = self.get_server_stats(profile_id)
            if stats:
                running.append({
                    "profile_id": profile_id,
                    **stats
                })
        
        return running
    
    async def cleanup(self):
        """Stop all servers on shutdown."""
        for profile_id in list(self.processes.keys()):
            await self.stop_server(profile_id, force=True)


# Singleton instance
server_manager = ServerManager()
```

---

## 10. Process Management

### Process Lifecycle

```
┌─────────────┐
│   Stopped   │
└──────┬──────┘
       │ start_server()
       ▼
┌─────────────┐
│  Starting   │◄──────────────────┐
└──────┬──────┘                   │
       │ health check OK          │ restart
       ▼                          │
┌─────────────┐                   │
│   Healthy   │───────────────────┤
└──────┬──────┘                   │
       │ health check fails       │
       ▼                          │
┌─────────────┐                   │
│  Unhealthy  │───────────────────┘
└──────┬──────┘
       │ stop_server()
       ▼
┌─────────────┐
│   Stopped   │
└─────────────┘
```

### Health Check Implementation

```python
# backend/app/services/health_checker.py
import asyncio
from datetime import datetime
from typing import Optional

from app.services.server_manager import server_manager
from app.database import get_session
from app.models import RunningInstance, ServerProfile
from sqlmodel import select

class HealthChecker:
    """Background service that monitors server health."""
    
    def __init__(self, interval: int = 30):
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the health check loop."""
        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """Stop the health check loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                await self._check_all_servers()
            except Exception as e:
                print(f"Health check error: {e}")
            
            await asyncio.sleep(self.interval)
    
    async def _check_all_servers(self):
        """Check health of all running servers."""
        async with get_session() as session:
            # Get all running instances
            stmt = select(RunningInstance)
            result = await session.execute(stmt)
            instances = result.scalars().all()
            
            for instance in instances:
                # Get profile
                profile_stmt = select(ServerProfile).where(
                    ServerProfile.id == instance.profile_id
                )
                profile_result = await session.execute(profile_stmt)
                profile = profile_result.scalar_one_or_none()
                
                if not profile:
                    continue
                
                # Check health
                health = await server_manager.check_health(profile)
                
                # Update instance
                instance.health_status = health["status"]
                instance.last_health_check = datetime.utcnow()
                session.add(instance)
            
            await session.commit()


# Singleton instance
health_checker = HealthChecker()
```

### Graceful Shutdown

```python
# backend/app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.server_manager import server_manager
from app.services.health_checker import health_checker
from app.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    await health_checker.start()
    
    yield
    
    # Shutdown
    await health_checker.stop()
    await server_manager.cleanup()

app = FastAPI(lifespan=lifespan)
```

---

## 11. launchd Integration

### Plist Generation

```python
# backend/app/services/launchd.py
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import plistlib

from app.models import ServerProfile

class LaunchdManager:
    """Manages launchd service configuration."""
    
    def __init__(self):
        self.launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
        self.label_prefix = "com.mlx-manager"
    
    def get_label(self, profile: ServerProfile) -> str:
        """Get the launchd label for a profile."""
        # Sanitize profile name for use in label
        safe_name = profile.name.lower().replace(" ", "-").replace("_", "-")
        return f"{self.label_prefix}.{safe_name}"
    
    def get_plist_path(self, profile: ServerProfile) -> Path:
        """Get the plist file path for a profile."""
        return self.launch_agents_dir / f"{self.get_label(profile)}.plist"
    
    def generate_plist(self, profile: ServerProfile) -> dict:
        """Generate a launchd plist dictionary for a profile."""
        label = self.get_label(profile)
        
        # Build program arguments (same as server_manager._build_command)
        program_args = [
            sys.executable,
            "-m", "mlx_openai_server.main",
            "--model-path", profile.model_path,
            "--model-type", profile.model_type,
            "--port", str(profile.port),
            "--host", profile.host,
            "--max-concurrency", str(profile.max_concurrency),
            "--queue-timeout", str(profile.queue_timeout),
            "--queue-size", str(profile.queue_size),
        ]
        
        if profile.context_length:
            program_args.extend(["--context-length", str(profile.context_length)])
        
        if profile.tool_call_parser:
            program_args.extend(["--tool-call-parser", profile.tool_call_parser])
        
        if profile.reasoning_parser:
            program_args.extend(["--reasoning-parser", profile.reasoning_parser])
        
        if profile.enable_auto_tool_choice:
            program_args.append("--enable-auto-tool-choice")
        
        if profile.trust_remote_code:
            program_args.append("--trust-remote-code")
        
        program_args.extend(["--log-level", profile.log_level])
        program_args.append("--no-log-file")  # Use launchd logging instead
        
        # Build plist dictionary
        plist = {
            "Label": label,
            "ProgramArguments": program_args,
            "RunAtLoad": profile.auto_start,
            "KeepAlive": {
                "SuccessfulExit": False,  # Restart on crash
                "Crashed": True
            },
            "StandardOutPath": f"/tmp/{label}.log",
            "StandardErrorPath": f"/tmp/{label}.err",
            "EnvironmentVariables": {
                "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
                "HOME": str(Path.home()),
                "PYTHONUNBUFFERED": "1"
            },
            "ProcessType": "Interactive",  # Higher priority
            "LowPriorityIO": False,
            "ThrottleInterval": 30,  # Prevent rapid restart loops
        }
        
        return plist
    
    def install(self, profile: ServerProfile) -> str:
        """Install a launchd service for a profile."""
        # Ensure LaunchAgents directory exists
        self.launch_agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and write plist
        plist = self.generate_plist(profile)
        plist_path = self.get_plist_path(profile)
        
        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)
        
        # Load the service
        label = self.get_label(profile)
        subprocess.run(
            ["launchctl", "load", str(plist_path)],
            check=True,
            capture_output=True
        )
        
        return str(plist_path)
    
    def uninstall(self, profile: ServerProfile) -> bool:
        """Uninstall a launchd service."""
        plist_path = self.get_plist_path(profile)
        label = self.get_label(profile)
        
        if not plist_path.exists():
            return False
        
        # Unload the service
        try:
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            pass  # May not be loaded
        
        # Remove plist file
        plist_path.unlink(missing_ok=True)
        
        return True
    
    def is_installed(self, profile: ServerProfile) -> bool:
        """Check if a launchd service is installed."""
        return self.get_plist_path(profile).exists()
    
    def is_running(self, profile: ServerProfile) -> bool:
        """Check if a launchd service is running."""
        label = self.get_label(profile)
        
        result = subprocess.run(
            ["launchctl", "list", label],
            capture_output=True,
            text=True
        )
        
        return result.returncode == 0
    
    def start(self, profile: ServerProfile) -> bool:
        """Start a launchd service."""
        label = self.get_label(profile)
        
        result = subprocess.run(
            ["launchctl", "start", label],
            capture_output=True
        )
        
        return result.returncode == 0
    
    def stop(self, profile: ServerProfile) -> bool:
        """Stop a launchd service."""
        label = self.get_label(profile)
        
        result = subprocess.run(
            ["launchctl", "stop", label],
            capture_output=True
        )
        
        return result.returncode == 0
    
    def get_status(self, profile: ServerProfile) -> dict:
        """Get detailed status of a launchd service."""
        label = self.get_label(profile)
        
        result = subprocess.run(
            ["launchctl", "list", label],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {
                "installed": self.is_installed(profile),
                "running": False,
                "label": label
            }
        
        # Parse output: PID\tStatus\tLabel
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 1:
            parts = lines[0].split("\t")
            pid = int(parts[0]) if parts[0] != "-" else None
            
            return {
                "installed": True,
                "running": pid is not None,
                "pid": pid,
                "label": label
            }
        
        return {
            "installed": True,
            "running": False,
            "label": label
        }


# Singleton instance
launchd_manager = LaunchdManager()
```

### Example Generated Plist

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlx-manager.coding</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/user/.pyenv/versions/3.11.0/bin/python</string>
        <string>-m</string>
        <string>mlx_openai_server.main</string>
        <string>--model-path</string>
        <string>mlx-community/Qwen2.5-Coder-32B-Instruct-4bit</string>
        <string>--model-type</string>
        <string>lm</string>
        <string>--port</string>
        <string>10240</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--context-length</string>
        <string>32768</string>
        <string>--tool-call-parser</string>
        <string>qwen3</string>
        <string>--enable-auto-tool-choice</string>
        <string>--log-level</string>
        <string>INFO</string>
        <string>--no-log-file</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/com.mlx-manager.coding.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/com.mlx-manager.coding.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/user/.pyenv/versions/3.11.0/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>/Users/user</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>LowPriorityIO</key>
    <false/>
    <key>ThrottleInterval</key>
    <integer>30</integer>
</dict>
</plist>
```

---

## 12. Security Considerations

### API Security

1. **Local-Only Binding**: By default, bind to `127.0.0.1` only
2. **No Authentication Required**: Since it's local-only, no auth needed for v1.0
3. **CORS**: Allow only localhost origins

```python
# backend/app/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # SvelteKit dev server
        "http://127.0.0.1:5173",
        "http://localhost:4173",  # SvelteKit preview
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### File System Security

1. **Validate Paths**: Ensure model paths are within expected directories
2. **No Arbitrary Code Execution**: `trust_remote_code` requires explicit opt-in
3. **Log File Permissions**: Create log files with restricted permissions

```python
# backend/app/utils/security.py
from pathlib import Path

ALLOWED_MODEL_DIRS = [
    Path.home() / ".cache" / "huggingface",
    Path.home() / "models",
]

def validate_model_path(path: str) -> bool:
    """Ensure model path is within allowed directories."""
    resolved = Path(path).resolve()
    
    for allowed in ALLOWED_MODEL_DIRS:
        try:
            resolved.relative_to(allowed.resolve())
            return True
        except ValueError:
            continue
    
    return False
```

### Process Security

1. **Signal Handling**: Proper SIGTERM/SIGKILL handling
2. **Resource Limits**: Consider ulimit settings for subprocess
3. **Cleanup on Exit**: Ensure all child processes are terminated

---

## 13. Project Structure

```
mlx-manager/
├── README.md
├── LICENSE
├── docker-compose.yml          # Optional: for isolated development
│
├── backend/
│   ├── pyproject.toml
│   ├── requirements.txt        # Generated from pyproject.toml
│   │
│   └── app/
│       ├── __init__.py
│       ├── main.py             # FastAPI application entry point
│       ├── config.py           # Configuration management
│       ├── database.py         # Database setup and session management
│       ├── models.py           # SQLModel definitions
│       │
│       ├── routers/
│       │   ├── __init__.py
│       │   ├── models.py       # /api/models endpoints
│       │   ├── profiles.py     # /api/profiles endpoints
│       │   ├── servers.py      # /api/servers endpoints
│       │   └── system.py       # /api/system endpoints
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   ├── hf_client.py    # HuggingFace Hub integration
│       │   ├── server_manager.py  # Process management
│       │   ├── health_checker.py  # Background health monitoring
│       │   └── launchd.py      # macOS service management
│       │
│       └── utils/
│           ├── __init__.py
│           └── security.py     # Path validation, etc.
│
├── frontend/
│   ├── package.json
│   ├── svelte.config.js
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   │
│   ├── src/
│   │   ├── app.html
│   │   ├── app.css             # Tailwind imports
│   │   │
│   │   ├── lib/
│   │   │   ├── api/
│   │   │   │   ├── client.ts   # API client
│   │   │   │   ├── models.ts   # Type definitions
│   │   │   │   └── types.ts
│   │   │   │
│   │   │   ├── stores/
│   │   │   │   ├── servers.svelte.ts
│   │   │   │   ├── profiles.svelte.ts
│   │   │   │   └── system.svelte.ts
│   │   │   │
│   │   │   ├── components/
│   │   │   │   ├── ui/         # shadcn-svelte components
│   │   │   │   ├── models/
│   │   │   │   ├── profiles/
│   │   │   │   ├── servers/
│   │   │   │   └── layout/
│   │   │   │
│   │   │   └── utils/
│   │   │       ├── format.ts   # Formatting helpers
│   │   │       └── sse.ts      # Server-Sent Events helper
│   │   │
│   │   └── routes/
│   │       ├── +layout.svelte
│   │       ├── +page.svelte
│   │       ├── models/
│   │       ├── profiles/
│   │       ├── servers/
│   │       └── settings/
│   │
│   └── static/
│       └── favicon.png
│
├── tests/
│   ├── backend/
│   │   ├── test_models.py
│   │   ├── test_profiles.py
│   │   ├── test_servers.py
│   │   └── test_services.py
│   │
│   └── frontend/
│       └── ... (Playwright tests)
│
└── scripts/
    ├── dev.sh                  # Start both frontend and backend
    ├── build.sh                # Production build
    └── install.sh              # One-line installer
```

---

## 14. Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Basic project structure with working backend

- [ ] Initialize Python project with FastAPI
- [ ] Set up SQLite database with SQLModel
- [ ] Implement `/api/profiles` CRUD endpoints
- [ ] Implement `/api/system/memory` endpoint
- [ ] Basic error handling and logging
- [ ] Unit tests for profile management

**Deliverables**:
- Working backend API (no frontend yet)
- Database schema created
- API documentation via Swagger UI

### Phase 2: Model Management (Week 1-2)
**Goal**: HuggingFace integration for model discovery

- [ ] Implement `HuggingFaceClient` service
- [ ] `/api/models/search` endpoint
- [ ] `/api/models/local` endpoint
- [ ] `/api/models/download` with SSE progress
- [ ] Model size estimation
- [ ] Downloaded model tracking

**Deliverables**:
- Search and download models via API
- Progress tracking for downloads

### Phase 3: Server Management (Week 2)
**Goal**: Process management for mlx-openai-server

- [ ] Implement `ServerManager` service
- [ ] `/api/servers` endpoints (start/stop/status)
- [ ] Health checking with background task
- [ ] Log streaming via SSE
- [ ] Process cleanup on shutdown

**Deliverables**:
- Start/stop servers via API
- Real-time health monitoring

### Phase 4: Frontend Foundation (Week 2-3)
**Goal**: Basic SvelteKit application

- [ ] Initialize SvelteKit project with Svelte 5
- [ ] Set up Tailwind CSS and shadcn-svelte
- [ ] Create layout with navigation
- [ ] Implement API client
- [ ] Create Svelte 5 stores

**Deliverables**:
- Working frontend shell
- API integration layer

### Phase 5: Frontend Features (Week 3)
**Goal**: Complete UI implementation

- [ ] Server dashboard with live updates
- [ ] Model browser with search
- [ ] Profile creation/editing forms
- [ ] Download progress UI
- [ ] Log viewer component

**Deliverables**:
- Fully functional web UI

### Phase 6: launchd Integration (Week 3-4)
**Goal**: System service management

- [ ] Implement `LaunchdManager` service
- [ ] Generate plist files
- [ ] Install/uninstall services
- [ ] UI for service management

**Deliverables**:
- Auto-start capability via launchd

### Phase 7: Polish & Documentation (Week 4)
**Goal**: Production-ready release

- [ ] Error handling improvements
- [ ] Loading states and feedback
- [ ] README and usage documentation
- [ ] Installation script
- [ ] End-to-end testing

**Deliverables**:
- v1.0.0 release

---

## 15. Testing Strategy

### Backend Tests

```python
# tests/backend/test_profiles.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_profile():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/profiles", json={
            "name": "Test Profile",
            "model_path": "mlx-community/test-model",
            "port": 10240
        })
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Profile"
        assert data["port"] == 10240

@pytest.mark.asyncio
async def test_port_conflict():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create first profile
        await client.post("/api/profiles", json={
            "name": "Profile 1",
            "model_path": "mlx-community/test-model",
            "port": 10240
        })
        
        # Try to create with same port
        response = await client.post("/api/profiles", json={
            "name": "Profile 2",
            "model_path": "mlx-community/test-model",
            "port": 10240
        })
        assert response.status_code == 409
```

### Frontend Tests (Playwright)

```typescript
// tests/frontend/servers.spec.ts
import { test, expect } from '@playwright/test';

test('server dashboard shows running servers', async ({ page }) => {
  await page.goto('/servers');
  
  // Check for server cards
  const serverCards = page.locator('[data-testid="server-card"]');
  await expect(serverCards).toBeVisible();
});

test('can start a server from profile', async ({ page }) => {
  await page.goto('/servers');
  
  // Click start button
  await page.click('[data-testid="start-server-btn"]');
  
  // Wait for status to change
  await expect(page.locator('[data-testid="server-status"]')).toHaveText('healthy', {
    timeout: 30000
  });
});
```

---

## 16. Deployment & Distribution

MLX Model Manager supports multiple distribution channels for easy installation on macOS.

### PyPI Distribution

The project is published to PyPI as `mlx-manager`, enabling installation via pip.

#### Installation from PyPI

```bash
# Install from PyPI
pip install mlx-manager

# Or using uv (faster)
uvx mlx-manager serve

# Or using pipx for isolated installation
pipx install mlx-manager
```

#### Publishing to PyPI

The release process uses GitHub Actions to automatically publish to PyPI when a new tag is created:

```bash
# 1. Update version in backend/pyproject.toml
# 2. Commit and tag the release
git add backend/pyproject.toml
git commit -m "chore: bump version to X.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main --tags

# The GitHub Action will automatically:
# - Build the wheel and source distribution
# - Publish to PyPI
```

#### Manual Publishing

```bash
cd backend

# Install build tools
pip install build twine

# Build distribution
python -m build

# Upload to PyPI (requires PyPI API token)
twine upload dist/*

# Or upload to TestPyPI first
twine upload --repository testpypi dist/*
```

### Homebrew Distribution

MLX Manager can also be installed via Homebrew using a custom tap.

#### Installation from Homebrew

```bash
# Add the tap
brew tap tumma72/mlx-manager

# Install
brew install mlx-manager

# Run
mlx-manager serve
```

#### Homebrew Formula

The Homebrew formula is maintained in a separate tap repository. Here's the formula structure:

```ruby
# Formula/mlx-manager.rb
class MlxManager < Formula
  include Language::Python::Virtualenv

  desc "Web-based manager for MLX models on Apple Silicon"
  homepage "https://github.com/tumma72/mlx-manager"
  url "https://files.pythonhosted.org/packages/source/m/mlx-manager/mlx_manager-X.Y.Z.tar.gz"
  sha256 "CHECKSUM_HERE"
  license "MIT"

  depends_on "python@3.11"
  depends_on :macos

  def install
    virtualenv_install_with_resources
  end

  service do
    run [opt_bin/"mlx-manager", "serve"]
    keep_alive true
    working_dir var/"mlx-manager"
    log_path var/"log/mlx-manager.log"
    error_log_path var/"log/mlx-manager-error.log"
  end

  test do
    system "#{bin}/mlx-manager", "--version"
  end
end
```

#### Creating a New Homebrew Release

1. Update the formula with new version and checksum:
```bash
# Get the SHA256 of the PyPI release
curl -sL https://pypi.org/pypi/mlx-manager/X.Y.Z/json | jq -r '.urls[] | select(.packagetype=="sdist") | .digests.sha256'
```

2. Update the tap repository with the new formula.

### Docker Distribution (Optional)

For users who prefer containerized deployments:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install mlx-manager
RUN pip install mlx-manager

# Expose the web UI port
EXPOSE 8080

# Note: MLX requires macOS with Apple Silicon
# This Dockerfile is for reference only
CMD ["mlx-manager", "serve", "--host", "0.0.0.0"]
```

Note: Docker support is limited since MLX requires macOS with Apple Silicon hardware. The Docker image is primarily for development/testing of the web interface on non-Mac systems.

### Build System

The project uses a Makefile at the root level for consistent build and test commands:

```bash
# View all available commands
make help

# Install dependencies
make install-dev

# Run all tests
make test

# Build for production
make build

# Run full CI pipeline
make ci
```

See the Makefile for the complete list of available targets.

### Versioning Strategy

The project follows Semantic Versioning (SemVer):

- **Major** (X.0.0): Breaking API changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

Version is defined in `backend/pyproject.toml` and should be updated before each release.

### Release Checklist

1. **Pre-release**
   - [ ] All tests passing (`make test`)
   - [ ] Linting clean (`make lint`)
   - [ ] Type checks passing (`make check`)
   - [ ] Documentation updated
   - [ ] CHANGELOG updated

2. **Release**
   - [ ] Update version in `backend/pyproject.toml`
   - [ ] Commit version bump
   - [ ] Create and push git tag
   - [ ] Verify PyPI release
   - [ ] Update Homebrew formula
   - [ ] Create GitHub Release with notes

3. **Post-release**
   - [ ] Verify installation from PyPI works
   - [ ] Verify Homebrew installation works
   - [ ] Announce release

---

## 17. Configuration Files

### Backend: pyproject.toml

```toml
[project]
name = "mlx-manager-backend"
version = "1.0.0"
description = "Backend API for MLX Model Manager"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "sqlmodel>=0.0.22",
    "aiosqlite>=0.20.0",
    "huggingface-hub>=0.27.0",
    "psutil>=6.0.0",
    "httpx>=0.28.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### Frontend: package.json

```json
{
  "name": "mlx-manager-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "vite dev",
    "build": "vite build",
    "preview": "vite preview",
    "check": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json",
    "lint": "eslint .",
    "format": "prettier --write ."
  },
  "devDependencies": {
    "@sveltejs/adapter-static": "^3.0.0",
    "@sveltejs/kit": "^2.9.0",
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "@tailwindcss/typography": "^0.5.15",
    "@types/node": "^22.0.0",
    "autoprefixer": "^10.4.20",
    "eslint": "^9.0.0",
    "postcss": "^8.4.49",
    "prettier": "^3.4.0",
    "prettier-plugin-svelte": "^3.3.0",
    "svelte": "^5.0.0",
    "svelte-check": "^4.0.0",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.7.0",
    "vite": "^6.0.0"
  },
  "dependencies": {
    "bits-ui": "^1.0.0",
    "clsx": "^2.1.1",
    "lucide-svelte": "^0.469.0",
    "tailwind-merge": "^2.5.0",
    "tailwind-variants": "^0.3.0"
  },
  "type": "module"
}
```

### Frontend: svelte.config.js

```javascript
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: 'index.html',
      precompress: false,
      strict: true
    }),
    alias: {
      $components: 'src/lib/components',
      $stores: 'src/lib/stores',
      $api: 'src/lib/api'
    }
  }
};

export default config;
```

### Frontend: tailwind.config.ts

```typescript
import type { Config } from 'tailwindcss';
import typography from '@tailwindcss/typography';

export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      colors: {
        // Custom colors for MLX branding
        mlx: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
        }
      }
    }
  },
  plugins: [typography]
} satisfies Config;
```

---

## 18. External References

### Core Technologies

| Resource | URL |
|----------|-----|
| **mlx-openai-server** | https://github.com/cubist38/mlx-openai-server |
| **mlx-lm** | https://github.com/ml-explore/mlx-lm |
| **huggingface_hub** | https://huggingface.co/docs/huggingface_hub |
| **MLX Community Models** | https://huggingface.co/mlx-community |

### Frontend

| Resource | URL |
|----------|-----|
| **SvelteKit Docs** | https://svelte.dev/docs/kit |
| **Svelte 5 Runes** | https://svelte.dev/docs/svelte/$state |
| **shadcn-svelte** | https://shadcn-svelte.com |
| **Tailwind CSS** | https://tailwindcss.com/docs |

### Backend

| Resource | URL |
|----------|-----|
| **FastAPI** | https://fastapi.tiangolo.com |
| **SQLModel** | https://sqlmodel.tiangolo.com |
| **Pydantic** | https://docs.pydantic.dev |

### macOS Integration

| Resource | URL |
|----------|-----|
| **launchd.plist** | https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html |
| **launchctl** | `man launchctl` |

### Related Projects (for reference)

| Project | URL | Relevance |
|---------|-----|-----------|
| **mlx-hub** | https://github.com/g-aggarwal/mlx-hub | CLI model management |
| **mlx-knife** | https://github.com/mzau/mlx-knife | Ollama-like CLI |
| **llm-mlx** | https://github.com/simonw/llm-mlx | LLM tool plugin |
| **Open WebUI** | https://github.com/open-webui/open-webui | Chat interface (use with our server) |

---

## Appendix A: API Client TypeScript Types

```typescript
// frontend/src/lib/api/types.ts

export interface ServerProfile {
  id: number;
  name: string;
  description: string | null;
  model_path: string;
  model_type: 'lm' | 'multimodal' | 'whisper' | 'embeddings' | 'image-generation' | 'image-edit';
  port: number;
  host: string;
  context_length: number | null;
  max_concurrency: number;
  queue_timeout: number;
  queue_size: number;
  tool_call_parser: string | null;
  reasoning_parser: string | null;
  enable_auto_tool_choice: boolean;
  trust_remote_code: boolean;
  chat_template_file: string | null;
  log_level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  log_file: string | null;
  no_log_file: boolean;
  auto_start: boolean;
  launchd_installed: boolean;
  created_at: string;
  updated_at: string;
}

export interface ServerProfileCreate {
  name: string;
  description?: string;
  model_path: string;
  model_type?: string;
  port: number;
  host?: string;
  context_length?: number;
  max_concurrency?: number;
  queue_timeout?: number;
  queue_size?: number;
  tool_call_parser?: string;
  reasoning_parser?: string;
  enable_auto_tool_choice?: boolean;
  trust_remote_code?: boolean;
  chat_template_file?: string;
  log_level?: string;
  log_file?: string;
  no_log_file?: boolean;
  auto_start?: boolean;
}

export interface RunningServer {
  profile_id: number;
  profile_name: string;
  pid: number;
  port: number;
  health_status: 'starting' | 'healthy' | 'unhealthy' | 'stopped';
  uptime_seconds: number;
  memory_mb: number;
}

export interface ModelSearchResult {
  model_id: string;
  author: string;
  downloads: number;
  likes: number;
  estimated_size_gb: number;
  tags: string[];
  is_downloaded: boolean;
  last_modified: string | null;
}

export interface LocalModel {
  model_id: string;
  local_path: string;
  size_bytes: number;
  size_gb: number;
}

export interface SystemMemory {
  total_gb: number;
  available_gb: number;
  used_gb: number;
  percent_used: number;
  mlx_recommended_gb: number;
}

export interface SystemInfo {
  os_version: string;
  chip: string;
  memory_gb: number;
  python_version: string;
  mlx_version: string;
  mlx_openai_server_version: string;
}

export interface DownloadProgress {
  status: 'starting' | 'downloading' | 'completed' | 'failed';
  model_id: string;
  progress: number;
  downloaded_bytes?: number;
  total_bytes?: number;
  speed_mbps?: number;
  error?: string;
}

export interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'starting';
  response_time_ms?: number;
  model_loaded?: boolean;
  error?: string;
}

export interface LaunchdStatus {
  installed: boolean;
  running: boolean;
  pid?: number;
  label: string;
}
```

---

## Appendix B: Development Commands

```bash
# Backend development
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 10242

# Frontend development
cd frontend
npm install
npm run dev

# Run both (from root)
./scripts/dev.sh

# Run tests
cd backend && pytest
cd frontend && npm run test

# Build for production
./scripts/build.sh

# Type checking
cd backend && mypy app
cd frontend && npm run check

# Linting
cd backend && ruff check app
cd frontend && npm run lint
```

---

**End of Document**

*This document is intended as a complete specification for implementation. For questions or clarifications, refer to the external references or create an issue in the project repository.*
