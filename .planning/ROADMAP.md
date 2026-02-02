# Roadmap: MLX Model Manager v1.2

## Overview

Transform MLX Manager from a management UI for external servers (mlx-openai-server) into a unified inference platform with our own high-performance MLX server. This milestone delivers a production-ready multi-model server built directly on mlx-lm/mlx-vlm/mlx-embeddings with continuous batching for 2-4x throughput improvement, paged KV cache for memory efficiency, and dual OpenAI/Anthropic API support.

**Architecture:** FastAPI + uvloop server with Pydantic v2 validation, Pydantic LogFire observability, model pool manager with LRU eviction, continuous batching scheduler, and paged KV cache—all leveraging Apple Silicon unified memory.

## Milestones

- v1.1 UX & Auth - Phases 1-6 (shipped 2026-01-26)
- v1.2 MLX Unified Server - Phases 7-15 (in progress)

## Phases

<details>
<summary>v1.1 UX & Auth (Phases 1-6) - SHIPPED 2026-01-26</summary>

29 requirements delivered across 6 phases:
- Models Panel UX, Server Panel Redesign
- User Authentication (JWT, admin approval)
- Model Discovery (characteristics, badges)
- Chat Multimodal (images, video, MCP tools)
- Bug Fixes (7 stability issues)

Archive: `.planning/milestones/v1.1-ROADMAP.md`

</details>

### v1.2 MLX Unified Server (In Progress)

**Milestone Goal:** Build our own high-performance MLX inference server with multi-model support, continuous batching, and dual API compatibility.

#### Phase 7: Foundation - Server Skeleton & Single Model Inference

**Goal**: FastAPI server skeleton with single model inference, OpenAI-compatible API, and SSE streaming

**Depends on**: Nothing (starts v1.2 milestone)

**Requirements**: SRV-01, SRV-02, SRV-03, API-01, API-02, API-04, ADAPT-01, ADAPT-02

**Success Criteria** (what must be TRUE):
1. FastAPI + uvloop server starts with Pydantic v2 request validation
2. Model pool manager loads one model via mlx-lm with memory tracking
3. `/v1/chat/completions` endpoint accepts OpenAI-format requests and returns responses
4. SSE streaming works for token-by-token response delivery
5. Llama family adapter handles chat template and stop tokens
6. `/v1/models` endpoint lists loaded models
7. Pydantic LogFire captures request spans (basic setup)

**Plans**: 7 plans in 3 waves

Plans:
- [x] 07-01-PLAN.md - Server foundation (package, config, FastAPI app, LogFire)
- [x] 07-02-PLAN.md - OpenAI schemas & /v1/models endpoint
- [x] 07-03-PLAN.md - Model pool manager with memory tracking
- [x] 07-04-PLAN.md - Model adapters (protocol + Llama adapter)
- [x] 07-05-PLAN.md - /v1/chat/completions with SSE streaming
- [x] 07-06-PLAN.md - /v1/completions endpoint (legacy API)
- [x] 07-07-PLAN.md - Fix MLX Metal thread affinity in inference (gap closure)

#### Phase 8: Multi-Model & Multimodal Support

**Goal**: Multi-model hot-swap with LRU eviction, vision model support, and additional model family adapters

**Depends on**: Phase 7

**Requirements**: SRV-04, SRV-05, API-05, ADAPT-03, ADAPT-04, ADAPT-05

**Success Criteria** (what must be TRUE):
1. Model pool manager supports multiple hot models with configurable memory limit
2. LRU eviction unloads least-recently-used models when memory pressure detected
3. Vision models load via mlx-vlm and generate responses using mlx_vlm.generate()
4. Embedding models load via mlx-embeddings for `/v1/embeddings` endpoint
5. Qwen, Mistral, and Gemma adapters handle their respective model families
6. Admin endpoints allow explicit model preload/unload

**Plans**: 7 plans in 3 waves

Plans:
- [x] 08-01-PLAN.md - Model pool LRU eviction and multi-model support
- [x] 08-02-PLAN.md - Qwen, Mistral, Gemma model family adapters
- [x] 08-03-PLAN.md - Model type detection and vision infrastructure
- [x] 08-04-PLAN.md - Vision model inference with chat API integration
- [x] 08-05-PLAN.md - /v1/embeddings endpoint with mlx-embeddings
- [x] 08-06-PLAN.md - Admin endpoints for model preload/unload
- [x] 08-07-PLAN.md - Fix vision model adapter compatibility (gap closure)

#### Phase 9: Continuous Batching & Paged KV Cache

**Goal**: Implement continuous batching scheduler and paged KV cache for 2-4x throughput improvement

**Depends on**: Phase 8

**Requirements**: BATCH-01, BATCH-02, BATCH-03, BATCH-04

**Success Criteria** (what must be TRUE):
1. Continuous batching scheduler processes multiple requests per token generation step
2. Paged KV cache allocates fixed-size blocks (32 tokens) instead of contiguous memory
3. Block table maps logical -> physical blocks with dynamic allocation
4. Prefix caching shares KV blocks across requests with identical prefixes
5. Priority queue allows request prioritization (high/normal/low)
6. Benchmark shows measurable throughput improvement over single-request baseline

**Plans**: 8 plans in 5 waves

Plans:
- [x] 09-01-PLAN.md - Foundation types and priority queue (types, BatchRequest, PriorityQueueWithAging)
- [x] 09-02-PLAN.md - Block manager and block table (KVBlock, PagedBlockManager)
- [x] 09-03-PLAN.md - Prefix caching with hash-based matching (PrefixCache)
- [x] 09-04-PLAN.md - Continuous batching scheduler core (ContinuousBatchingScheduler)
- [x] 09-05-PLAN.md - Batch inference engine with MLX integration (BatchInferenceEngine)
- [x] 09-06-PLAN.md - API integration and scheduler management (SchedulerManager, chat routing)
- [x] 09-07-PLAN.md - Benchmark and documentation (throughput verification)
- [x] 09-08-PLAN.md - Wire BatchInferenceEngine in configure_scheduler (gap closure)

#### Phase 10: Dual Protocol & Cloud Fallback

**Goal**: Anthropic API compatibility and cloud backend fallback for reliability

**Depends on**: Phase 9

**Requirements**: API-03, CLOUD-01, CLOUD-02, CLOUD-03, CLOUD-04

**Success Criteria** (what must be TRUE):
1. `/v1/messages` endpoint accepts Anthropic-format requests with protocol translation
2. Streaming works in Anthropic SSE format (event: content_block_delta, etc.)
3. OpenAI cloud backend routes requests when configured (httpx.AsyncClient)
4. Anthropic cloud backend routes with automatic OpenAI -> Anthropic translation
5. Model -> backend mapping stored in database (local model A, cloud model B)
6. Automatic failover: local failure triggers cloud fallback if configured

**Plans**: 9 plans in 4 waves

Plans:
- [x] 10-01-PLAN.md — Anthropic Messages API schemas
- [x] 10-02-PLAN.md — Database models for backend routing
- [x] 10-03-PLAN.md — Protocol translator service
- [x] 10-04-PLAN.md — Cloud backend client with retries
- [x] 10-05-PLAN.md — /v1/messages endpoint with Anthropic SSE
- [x] 10-06-PLAN.md — OpenAI cloud backend
- [x] 10-07-PLAN.md — Anthropic cloud backend with translation
- [x] 10-08-PLAN.md — Backend router with failover
- [x] 10-09-PLAN.md — Wire router into chat endpoint

#### Phase 11: Configuration UI

**Goal**: Visual configuration for model pool, cloud providers, and routing rules

**Depends on**: Phase 10

**Requirements**: CONF-01, CONF-02, CONF-03, CONF-04

**Success Criteria** (what must be TRUE):
1. API keys stored encrypted using existing AuthLib infrastructure
2. Model pool configuration UI (max memory, eviction policy, preload list)
3. Provider configuration UI for OpenAI/Anthropic API keys and base URLs
4. Model routing rules UI supports exact match, prefix, and regex patterns
5. Configuration changes apply without server restart

**Plans**: 6 plans in 3 waves

Plans:
- [x] 11-01-PLAN.md — Backend encryption service and settings API
- [x] 11-02-PLAN.md — Settings page and provider configuration UI
- [x] 11-03-PLAN.md — Model pool settings UI
- [x] 11-04-PLAN.md — Routing rules UI with drag-drop
- [x] 11-05-PLAN.md — Integration and navbar link
- [x] 11-06-PLAN.md — UAT bug fixes (gap closure)

#### Phase 12: Production Hardening

**Goal**: Observability, error handling, timeouts, and audit logging for production deployment

**Depends on**: Phase 11

**Requirements**: PROD-01, PROD-02, PROD-03, PROD-04

**Success Criteria** (what must be TRUE):
1. LogFire integration captures full request lifecycle with LLM token metrics
2. Unified error responses with consistent format and proper HTTP status codes
3. Per-backend configurable timeouts (local default: 15min, cloud default: 10min)
4. Request audit log captures: timestamp, model, backend type, duration, status, token count
5. Admin panel displays request logs with filtering by model, backend, status, time range

**Plans**: 7 plans in 4 waves

Plans:
- [x] 12-01-PLAN.md — LogFire integration (expand to both apps, HTTPX, SQLAlchemy, LLM)
- [x] 12-02-PLAN.md — RFC 7807 error responses with request_id
- [x] 12-03-PLAN.md — Per-endpoint configurable timeouts
- [x] 12-04-PLAN.md — Audit logging with background writes
- [x] 12-05-PLAN.md — Admin panel for audit logs with WebSocket
- [x] 12-06-PLAN.md — Timeout configuration UI
- [x] 12-07-PLAN.md — CLI benchmarks and PERFORMANCE.md

#### Phase 13: MLX Server Integration

**Goal**: Embed MLX Server into MLX Manager — remove legacy mlx-openai-server code, wire Chat UI to embedded server, connect Settings to live server behavior

**Depends on**: Phase 12

**Requirements**: INT-01 (Embedded Server), INT-02 (Legacy Removal), INT-03 (Chat Integration), INT-04 (Settings Integration)

**Success Criteria** (what must be TRUE):
1. MLX Server routes mounted directly into MLX Manager FastAPI app (single process, single port)
2. All mlx-openai-server code removed (server_manager.py, parser_options.py, command_builder.py)
3. Chat UI sends requests to embedded `/v1/chat/completions` endpoint
4. Model Pool Settings actually control the embedded server's model pool
5. Cloud Provider Settings connect to the embedded server's routing
6. Audit logs populate when using Chat UI
7. No references to mlx-openai-server remain in codebase

**Plans**: 5 plans in 3 waves

Plans:
- [ ] 13-01-PLAN.md — Mount MLX Server as sub-application of MLX Manager
- [ ] 13-02-PLAN.md — Remove legacy mlx-openai-server subprocess management code
- [ ] 13-03-PLAN.md — Wire Chat UI to embedded MLX Server
- [ ] 13-04-PLAN.md — Wire Settings to embedded MLX Server configuration
- [ ] 13-05-PLAN.md — Verify complete integration and update tests

#### Phase 14: Model Adapter Enhancements

**Goal**: Achieve feature parity with mlx-openai-server, mlx-omni-server, and vllm-mlx by implementing tool calling, reasoning mode, message converters, structured output, and LoRA support

**Depends on**: Phase 13

**Requirements**: ADAPT-06 (Tool Calling), ADAPT-07 (Reasoning Mode), ADAPT-08 (Message Converters), ADAPT-09 (Structured Output), ADAPT-10 (LoRA Adapters)

**Success Criteria** (what must be TRUE):
1. Tool/function calling works with model-specific parsers (Llama, Qwen, GLM4)
2. Reasoning/thinking mode parses `<think>`, `<reasoning>` tags and reasoning_content field
3. Message converters translate between OpenAI format and model-specific formats
4. Structured output validates responses against JSON Schema
5. LoRA adapters can be loaded alongside base models
6. All parsers integrated into chat completions endpoint

**Plans**: 6 plans in 3 waves

Plans:
- [x] 14-01-PLAN.md — Extended OpenAI schemas and ModelAdapter protocol
- [x] 14-02-PLAN.md — Tool call parsers (Llama, Qwen, GLM4)
- [x] 14-03-PLAN.md — Reasoning/thinking mode extraction
- [x] 14-04-PLAN.md — Structured output validation service
- [x] 14-05-PLAN.md — LoRA adapter loading support
- [x] 14-06-PLAN.md — Chat endpoint integration and tests
- [x] 14-07-PLAN.md — Unified ResponseProcessor (single-pass parsing, fix content cleanup)
- [x] 14-08-PLAN.md — StreamingProcessor (filter patterns during streaming)
- [x] 14-09-PLAN.md — Generic OpenAI/Anthropic-compatible providers

#### Phase 15: Code Cleanup & Integration Tests

**Goal**: Remove dead parser code, fix blocker bugs discovered during UAT, and create integration tests for ResponseProcessor to validate core inference works with all model families

**Depends on**: Phase 14

**Requirements**: CLEAN-01 (Dead Code Removal), CLEAN-02 (Bug Fixes), CLEAN-03 (Integration Tests)

**Success Criteria** (what must be TRUE):
1. Dead code removed: adapters/parsers/ folder, unused adapter methods (parse_tool_calls, extract_reasoning)
2. Database migration created for CloudCredential api_type and name columns
3. Qwen adapter handles enable_thinking exceptions properly (not just TypeError)
4. Streaming token logging changed from INFO to DEBUG level
5. Integration tests validate ResponseProcessor with real tokenizer outputs from each model family
6. Golden file tests cover tool calling and thinking extraction patterns for Qwen, Llama, GLM4

**Plans**: TBD (run /gsd:plan-phase 15 to break down)

Plans:
- [ ] 15-01-PLAN.md — TBD
- [ ] 15-02-PLAN.md — TBD
- [ ] 15-03-PLAN.md — TBD

## Progress

**Execution Order:** Phases execute in numeric order: 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 7. Foundation | v1.2 | 7/7 | Complete | 2026-01-28 |
| 8. Multi-Model | v1.2 | 7/7 | Complete | 2026-01-28 |
| 9. Batching | v1.2 | 8/8 | Complete | 2026-01-29 |
| 10. Dual Protocol | v1.2 | 9/9 | Complete | 2026-01-29 |
| 11. Configuration | v1.2 | 6/6 | Complete | 2026-01-30 |
| 12. Hardening | v1.2 | 7/7 | Complete | 2026-01-31 |
| 13. Integration | v1.2 | 0/5 | Planned | - |
| 14. Adapters | v1.2 | 9/9 | Complete | 2026-02-02 |
| 15. Cleanup & Tests | v1.2 | 0/3 | Not Started | - |

## Technical Architecture

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
|  |                    Protocol Translator                         |  |
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
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    Paged KV Cache Manager                      |  |
|  |  Block Pool | Block Tables | Prefix Sharing | Copy-on-Write    |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    Model Adapters (per family)                 |  |
|  |  Llama | Qwen | Mistral | Gemma | GLM4 | VisionAddOn           |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    MLX Libraries                               |  |
|  |  mlx-lm | mlx-vlm | mlx-embeddings | MLX Core                  |  |
|  +---------------------------------------------------------------+  |
|                                                                      |
|  +---------------------------------------------------------------+  |
|  |                    Observability (Pydantic LogFire)            |  |
|  |  Request Tracing | LLM Metrics | SQLite Spans | Alerts         |  |
|  +---------------------------------------------------------------+  |
+----------------------------------------------------------------------+
```

## Key Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| mlx-lm | latest | Text generation |
| mlx-vlm | latest | Vision-language models |
| mlx-embeddings | latest | Text embeddings |
| FastAPI | 0.115+ | HTTP server |
| uvloop | 0.19+ | 2-4x async performance |
| Pydantic | 2.x | Rust-powered validation |
| logfire | latest | Observability |
| httpx | 0.27+ | Cloud backend clients |
| httpx-retries | 0.4+ | Retry transport for cloud backends |
| jsonschema | 4.x | JSON Schema validation |

---
*Roadmap revised: 2026-02-01*
*Previous version: adapter/proxy approach (deprecated)*
