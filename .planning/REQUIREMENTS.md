# Requirements: MLX Model Manager v1.2

**Defined:** 2026-01-27 (revised)
**Core Value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.

## v1.2 Vision: MLX Unified Server

**Pivot:** Instead of building adapters/proxies for existing backends (mlx-openai-server, vLLM-MLX), we're building our own high-performance MLX inference server that:

- Serves multiple models simultaneously with dynamic loading
- Achieves 2-4x throughput via continuous batching (proven by vLLM-MLX: 328→1112 tok/s)
- Exposes both OpenAI and Anthropic compatible REST APIs
- Leverages Apple Silicon unified memory for zero-copy operations

**Rationale:** Research confirmed feasibility — mlx-lm/mlx-vlm/mlx-embeddings are mature, vLLM-MLX demonstrated batching gains, and we gain full control over the inference stack.

## v1.2 Requirements

### Server Foundation (SRV)

- [ ] **SRV-01**: FastAPI + uvloop server skeleton with Pydantic v2 validation and LogFire observability
- [ ] **SRV-02**: Model pool manager loads/unloads models with LRU eviction and memory pressure monitoring
- [ ] **SRV-03**: Single model inference via mlx-lm (text generation) with SSE streaming
- [ ] **SRV-04**: Vision model support via mlx-vlm (VisionAddOn pattern from LM Studio)
- [ ] **SRV-05**: Embedding model support via mlx-embeddings for `/v1/embeddings` endpoint

### Continuous Batching (BATCH)

- [ ] **BATCH-01**: Continuous batching scheduler processes multiple requests per token generation step
- [ ] **BATCH-02**: Paged KV cache manager allocates fixed-size blocks with copy-on-write
- [ ] **BATCH-03**: Prefix caching shares KV blocks across requests with common prefixes
- [ ] **BATCH-04**: Priority queue supports request prioritization (high/normal/low)

### API Layer (API)

- [ ] **API-01**: OpenAI-compatible `/v1/chat/completions` endpoint with full parameter support
- [ ] **API-02**: OpenAI-compatible `/v1/completions` endpoint for legacy clients
- [ ] **API-03**: Anthropic-compatible `/v1/messages` endpoint with protocol translation
- [ ] **API-04**: Model listing endpoint `/v1/models` returns all hot + loadable models
- [ ] **API-05**: Management endpoints for model preload/unload and pool status

### Model Adapters (ADAPT)

- [ ] **ADAPT-01**: Abstract ModelAdapter protocol defines per-family handling (chat template, tool parsing, stop tokens)
- [ ] **ADAPT-02**: Llama family adapter (Llama 3.x, CodeLlama)
- [ ] **ADAPT-03**: Qwen family adapter (Qwen2, Qwen2.5, Qwen-VL, Qwen3)
- [ ] **ADAPT-04**: Mistral family adapter (Mistral, Mixtral)
- [ ] **ADAPT-05**: Gemma family adapter (Gemma, Gemma2, Gemma3)

### Cloud Fallback (CLOUD)

- [ ] **CLOUD-01**: OpenAI cloud backend for fallback routing when local unavailable
- [ ] **CLOUD-02**: Anthropic cloud backend with automatic format translation
- [ ] **CLOUD-03**: Model name → backend mapping stored in database
- [ ] **CLOUD-04**: Automatic failover: local backend failure routes to configured cloud

### Configuration & UI (CONF)

- [ ] **CONF-01**: API keys stored encrypted using AuthLib (existing auth infrastructure)
- [ ] **CONF-02**: Visual model pool configuration (max memory, eviction policy)
- [ ] **CONF-03**: Provider configuration UI for cloud API keys and base URLs
- [ ] **CONF-04**: Model routing rules UI (exact, prefix, regex patterns)

### Production Hardening (PROD)

- [ ] **PROD-01**: Pydantic LogFire integration for request tracing and LLM metrics
- [ ] **PROD-02**: Unified error responses with proper HTTP status codes
- [ ] **PROD-03**: Per-backend configurable timeouts (local: 15min default, cloud: 10min default)
- [ ] **PROD-04**: Request audit log: timestamp, model, backend, duration, status, tokens

## v2 Requirements (Deferred)

### Performance Optimization (v1.3+)

- **PERF-01**: Chunked prefill for improved throughput
- **PERF-02**: Speculative decoding with draft models
- **PERF-03**: Rust SSE encoder for frame serialization
- **PERF-04**: Rust request router for low-latency routing

### AI Proxy Agent (v1.3+)

- **AIPROXY-01**: LLM-based intelligent routing (orchestrator model routes requests)
- **AIPROXY-02**: Natural language routing rules
- **AIPROXY-03**: Model role configuration (Thinking Agent, Worker Agent, etc.)

### Chat History (Deferred)

- **CHAT-05**: Persist chat history to database
- **CHAT-06**: Chat history sidebar per server
- **CHAT-07**: Server-scoped chats
- **CHAT-08**: Create new / delete existing chats

## Out of Scope

| Feature | Reason |
|---------|--------|
| Raw MLX implementation | Use mlx-lm/mlx-vlm/mlx-embeddings wrappers |
| Model training/fine-tuning | Focus on inference serving |
| Multi-node clustering | Local-first, single Apple Silicon machine |
| Every LLM provider | Focus on OpenAI + Anthropic (extensible) |
| Windows/Linux support | macOS-specific (MLX, unified memory) |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SRV-01 | Phase 7 | Complete |
| SRV-02 | Phase 7 | Complete |
| SRV-03 | Phase 7 | Complete |
| SRV-04 | Phase 8 | Pending |
| SRV-05 | Phase 8 | Pending |
| BATCH-01 | Phase 9 | Pending |
| BATCH-02 | Phase 9 | Pending |
| BATCH-03 | Phase 9 | Pending |
| BATCH-04 | Phase 9 | Pending |
| API-01 | Phase 7 | Complete |
| API-02 | Phase 7 | Complete |
| API-03 | Phase 10 | Pending |
| API-04 | Phase 7 | Complete |
| API-05 | Phase 8 | Pending |
| ADAPT-01 | Phase 7 | Complete |
| ADAPT-02 | Phase 7 | Complete |
| ADAPT-03 | Phase 8 | Pending |
| ADAPT-04 | Phase 8 | Pending |
| ADAPT-05 | Phase 8 | Pending |
| CLOUD-01 | Phase 10 | Pending |
| CLOUD-02 | Phase 10 | Pending |
| CLOUD-03 | Phase 10 | Pending |
| CLOUD-04 | Phase 10 | Pending |
| CONF-01 | Phase 11 | Pending |
| CONF-02 | Phase 11 | Pending |
| CONF-03 | Phase 11 | Pending |
| CONF-04 | Phase 11 | Pending |
| PROD-01 | Phase 12 | Pending |
| PROD-02 | Phase 12 | Pending |
| PROD-03 | Phase 12 | Pending |
| PROD-04 | Phase 12 | Pending |

**Coverage:**
- v1.2 requirements: 28 total
- Mapped to phases: 28/28 (100%)
- Unmapped: 0

---
*Requirements revised: 2026-01-27*
*Previous version: adapter/proxy approach (deprecated)*
