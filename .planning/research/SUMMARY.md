# Project Research Summary

**Project:** MLX Model Manager v1.2 - Unified API Gateway
**Domain:** LLM API Gateway / Proxy with Local-First Routing
**Researched:** 2026-01-26
**Confidence:** HIGH

## Executive Summary

The unified API gateway transforms MLX Manager from a local server management tool into an intelligent routing layer that seamlessly bridges local Apple Silicon inference (via mlx-openai-server) with cloud providers (OpenAI, Anthropic). Research shows this is a well-established domain in 2026, with clear patterns for gateway architecture, protocol translation, and multi-backend routing. The recommended approach leverages FastAPI's existing infrastructure with httpx.AsyncClient for proxying, adds an adapter pattern for backend abstraction, and stores routing configuration in SQLite for dynamic management.

The primary differentiator is **local-first with cloud fallback** — while most gateways focus on cloud-to-cloud routing, MLX Manager can optimize for free local inference while providing cloud reliability. The architecture integrates cleanly with existing components: the gateway router delegates to the ServerManager for on-demand model loading, reuses the health checker for backend availability, and extends the database schema for model routing configuration. Critical technology decisions: use httpx (already in dependencies) over specialized gateway libraries, adopt cryptography.fernet for API key encryption, integrate the official Anthropic SDK, and defer vLLM-MLX support due to experimental maturity.

Key risks center on process management (orphaned subprocesses), concurrency (race conditions in auto-start), and streaming reliability (timeout cascades, buffering). Prevention strategies include process group management, startup state machines with async locks, SSE heartbeats, and client disconnection detection. The research identified 14 critical pitfalls with actionable prevention strategies mapped to implementation phases.

## Key Findings

### Recommended Stack

The gateway extends the existing FastAPI + SQLModel stack with minimal new dependencies. No specialized gateway framework is required — httpx.AsyncClient (already at v0.28.0+) provides connection pooling and async streaming for proxying to both local mlx-openai-server instances and cloud APIs. Two new dependencies address specific needs: cryptography (v46.0.0+) for Fernet symmetric encryption of API keys at rest, and the official Anthropic SDK (v0.76.0) for type-safe Anthropic API integration with built-in retry logic and streaming support.

**Core technologies:**
- **httpx.AsyncClient**: HTTP proxying with connection pooling and streaming — reuse existing dependency, zero framework overhead
- **cryptography.fernet**: Encrypt API keys in SQLite with AES-128-CBC + HMAC — battle-tested, simple API, no external key management required
- **anthropic SDK (v0.76.0)**: Official Anthropic client with async support — maintained by Anthropic, automatic API versioning, native streaming
- **FastAPI lifespan**: Manage HTTP client lifecycle — initialize singleton clients at startup, proper cleanup on shutdown
- **Adapter pattern**: Abstract backend differences — each adapter (MLX, OpenAI, Anthropic) translates protocols independently

**Rejected alternatives:**
- **vLLM-MLX**: Too experimental (74 commits vs 703 for mlx-openai-server), defer to v1.3+ when mature
- **fastapi-proxy-lib**: Unnecessary abstraction over httpx, direct usage clearer and more maintainable
- **SQLCipher**: Overkill when only API keys need encryption, application-level encryption sufficient

### Expected Features

LLM API gateways have well-established expectations in 2026. Table stakes include OpenAI-compatible `/v1/chat/completions` endpoint, streaming SSE support, model name routing, request/response format translation, and proper error propagation. Missing any of these makes the product feel incomplete — tools like LiteLLM and OpenRouter define the standard.

**Must have (table stakes):**
- OpenAI-compatible `/v1/chat/completions` endpoint — industry standard, all client libraries expect this format
- Streaming response support (SSE) — required for real-time chat UX, must emit `data: [DONE]` properly
- Model name routing — resolve model identifier to backend (local profile, cloud provider)
- Request/response format translation — convert between OpenAI ↔ Anthropic ↔ native formats
- API key authentication — secure access control with Bearer token
- Health check endpoint — monitor gateway and backend availability
- Request logging — audit trail for debugging (request ID, model, latency, tokens, cost)

**Should have (competitive):**
- On-demand local model auto-start — zero-config: request arrives → model loads → inference begins (UNIQUE to mlx-manager)
- Local-first with cloud fallback — use Apple Silicon when available, cloud for reliability (UNIQUE value proposition)
- Model auto-discovery — detect available models from running servers, reduce manual configuration
- Visual routing configuration — UI for mapping model names to backends (better UX than YAML editing)
- Cost tracking per backend — show local (free) vs cloud (paid) usage breakdown

**Defer (v2+):**
- Anthropic `/v1/messages` endpoint — can route through OpenAI format initially
- Cache-aware routing — route repeat prompts to same backend for KV cache hits (45% latency improvement possible, but very high complexity)
- Intelligent failover — automatic provider switching on failure (medium complexity, nice-to-have for reliability)

### Architecture Approach

The gateway integrates as a new router + adapter service layer within MLX Manager's existing architecture. A new `/api/gateway` router accepts OpenAI/Anthropic-compatible requests, resolves model names to backend configurations (stored in SQLite `gateway_model_routes` table), and delegates to backend-specific adapters. The adapter pattern provides a uniform interface while handling protocol differences: MLXLocalAdapter integrates with the existing ServerManager for on-demand model loading, OpenAIAdapter and AnthropicAdapter proxy to cloud APIs with format translation, and VLLMMLXAdapter supports external vLLM-MLX servers (deferred for v1.2).

**Major components:**
1. **Gateway Router** (`routers/gateway.py`) — FastAPI router that accepts `/v1/chat/completions` requests, authenticates via existing JWT, resolves model to backend, and streams responses
2. **Router Service** (`services/gateway/router_service.py`) — Model name resolution with fallback chain: database routes → profile lookup → pattern matching (gpt-* → OpenAI, claude-* → Anthropic) → default backend
3. **Adapter Layer** (`services/gateway/adapter_*.py`) — Backend abstraction with base adapter interface, MLX local adapter (integrates ServerManager for auto-start), cloud adapters (OpenAI, Anthropic with format translation)
4. **Format Translator** (`services/gateway/format_translator.py`) — Protocol translation between OpenAI and Anthropic message formats (system message handling, tool calling, streaming differences)
5. **Database Extensions** — New tables: `gateway_model_routes` (model name → backend mapping), `gateway_requests` (audit log with latency/token tracking)

**Integration points:**
- **ServerManager**: MLXLocalAdapter calls `start_server()` for on-demand loading, checks `is_running()` before routing
- **HealthChecker**: Adapters query backend health before routing, extend with startup grace periods to avoid health check storms
- **Database**: Extends existing async session pattern, adds routing configuration and request audit log tables

**Key architectural decision:** Gateway lives in-process as FastAPI router (not separate microservice) to reuse ServerManager singleton, share auth/database/config via FastAPI Depends, and maintain simple deployment model.

### Critical Pitfalls

Research identified 14 pitfalls across process management, API compatibility, security, and performance. The top 5 require prevention in Phase 1-2:

1. **Orphaned Subprocess Cleanup** — When gateway crashes, mlx-openai-server processes become orphaned and consume resources indefinitely. Prevention: start subprocesses with `start_new_session=True`, write PIDs to files for recovery, scan/kill orphans on startup, implement heartbeat monitoring.

2. **Race Conditions in On-Demand Loading** — Concurrent requests to same model trigger duplicate startup attempts, causing port conflicts and wasted resources. Prevention: per-profile async locks (`asyncio.Lock`), startup state machine (stopped → starting → running), request queuing during startup with 60s timeout and cleanup on failure.

3. **Streaming Timeout Cascades** — Long LLM responses get terminated by intermediate timeouts (nginx 60s, API Gateway 30s), causing incomplete responses. Prevention: SSE heartbeats every 15s during idle, explicit timeout hierarchy (gateway→backend 15min, client configurable 10min default), graceful degradation with `X-Generation-Incomplete` header.

4. **API Key Leakage in Logs** — Credentials appear in logs, error messages, database dumps. Prevention: structured logging with redaction filter (`RedactAPIKeysFilter` regex for api_key patterns), environment variable masking (show `***SET***`/`***UNSET***`), database encryption with Fernet, sanitize error responses before returning to client.

5. **OpenAI vs Anthropic Format Mismatches** — Requests fail when forwarding OpenAI format to Anthropic backend without translation (system message handling, streaming terminator differences, tool calling format). Prevention: request transformation layer (concatenate system messages, extract to separate parameter), response transformation (add `[DONE]` terminator for OpenAI compatibility), feature compatibility matrix with capability-based routing.

**Additional critical pitfalls:** Model name routing ambiguity (prevent with explicit configuration + prefix namespacing), existing ServerManager lifecycle interference (server ownership tagging), background task cleanup (extend existing cancel pattern), health check storms during startup (startup grace periods), streaming response buffering (explicit StreamingResponse type), insecure credential storage (macOS Keychain integration), SSRF vulnerabilities (URL allowlist + IP blocklist), connection pool exhaustion (singleton AsyncClient with limits), memory leaks in streaming (client disconnection detection).

## Implications for Roadmap

Based on research, gateway development should follow a 5-phase approach that builds incrementally from foundation (core routing) through local integration (on-demand loading) to cloud backends (multi-provider) and production hardening (observability, failover). This order minimizes risk by validating patterns with the simplest backend (local MLX) before adding cloud complexity.

### Phase 1: Foundation - Core Gateway Router
**Rationale:** Establish gateway infrastructure and prove integration with existing ServerManager before adding multi-backend complexity. Start with single backend type (local MLX) to validate adapter pattern, routing logic, and streaming without cloud API dependencies.

**Delivers:** Basic `/v1/chat/completions` endpoint that routes to local mlx-openai-server instances, proves adapter pattern works, establishes database schema for routing configuration.

**Addresses:** Table stakes features (OpenAI endpoint, streaming, model routing), critical security (API key encryption, SSRF validation), foundational pitfalls (subprocess cleanup, background task management, server ownership model).

**Avoids:** Pitfalls 1, 4, 6, 7, 11, 12 (critical security and process management issues must be solved here).

**Components:**
- `routers/gateway.py` with `/v1/chat/completions` endpoint
- `services/gateway/adapter_base.py` abstract interface
- `services/gateway/adapter_mlx.py` local MLX adapter
- `services/gateway/router_service.py` model name resolution (profile lookup only)
- Database migration for `gateway_model_routes` table
- Orphaned process cleanup with PID tracking
- API key encryption with Fernet in new `cloud_providers` table
- SSRF validation for backend URLs

### Phase 2: On-Demand Model Loading
**Rationale:** Core differentiator for mlx-manager. Once basic routing works, add auto-start capability that triggers `ServerManager.start_server()` when request arrives for stopped model. Requires solving race conditions and coordinating with existing health checker.

**Delivers:** Gateway automatically starts mlx-openai-server on first request, queues concurrent requests until healthy, prevents duplicate startups.

**Uses:** Existing ServerManager (start_server, is_running), existing HealthChecker (extend with startup grace periods)

**Implements:** MLXLocalAdapter integration with ServerManager, startup state machine with per-profile async locks, request queuing during startup, health check coordination.

**Avoids:** Pitfalls 2, 8 (race conditions and health check storms are critical for auto-start reliability).

**Components:**
- Startup locks in ServerManager (`_startup_locks: dict[int, asyncio.Lock]`)
- Request queuing mechanism (queue requests during "starting" state)
- Health checker grace periods (disable checks for 60s after start)
- Startup timeout with cleanup (kill process if not healthy in 60s)

### Phase 3: Cloud Backend Integration
**Rationale:** Add cloud providers (OpenAI, Anthropic) to demonstrate multi-backend routing and protocol translation. Builds on proven adapter pattern from Phase 1-2. This phase unlocks "local-first with cloud fallback" value proposition.

**Delivers:** Route to OpenAI and Anthropic cloud APIs, translate between OpenAI and Anthropic message formats, unified endpoint for local + cloud backends.

**Uses:** anthropic SDK (v0.76.0), httpx.AsyncClient for OpenAI, format translator for protocol differences.

**Implements:** Cloud adapters (OpenAI, Anthropic), format translation layer, streaming compatibility (add [DONE] terminator for Anthropic), connection pooling configuration.

**Avoids:** Pitfalls 3, 9, 10, 13 (streaming timeout cascades, format mismatches, buffering, connection exhaustion).

**Components:**
- `services/gateway/adapter_openai.py` cloud adapter
- `services/gateway/adapter_anthropic.py` with SDK integration
- `services/gateway/format_translator.py` OpenAI ↔ Anthropic translation
- `/v1/messages` Anthropic-compatible endpoint (optional)
- Connection pooling with singleton AsyncClient (100 connections cloud, 10-20 local)
- SSE heartbeats every 15s to prevent timeout cascades
- Explicit StreamingResponse type to prevent buffering

### Phase 4: Model Configuration UI
**Rationale:** Core routing works, now add management layer for non-technical users. Database-driven routing established in Phase 1, this adds CRUD UI on top.

**Delivers:** Frontend components for managing model routes (table view, add/edit forms), backend CRUD endpoints for `gateway_model_routes`.

**Implements:** Visual routing configuration (differentiator feature), model auto-discovery from running servers, frontend integration with gateway API.

**Components:**
- Frontend: Gateway routes table component
- Frontend: Add/edit route form with backend selection
- Backend: GET/POST/PUT/DELETE `/api/gateway/routes`
- Model auto-discovery endpoint: GET `/api/gateway/available-models`

### Phase 5: Production Hardening
**Rationale:** Core functionality complete, add observability, reliability, and robustness features for production use.

**Delivers:** Request audit logging, automatic failover to alternative backends, cost tracking per provider, memory leak prevention.

**Implements:** Audit log persisted in `gateway_requests` table, priority-based fallback routing, client disconnection detection in streaming, memory monitoring.

**Avoids:** Pitfalls 14 (memory leaks in streaming), plus improved handling of pitfalls 3, 5 (timeout hierarchy, routing ambiguity resolution).

**Components:**
- `gateway_requests` table with audit log schema
- Priority-based fallback routing (query routes by priority order)
- Client disconnection detection in streaming generators
- Cost tracking integration (token counting, pricing table)
- Frontend: Request logs page with filtering
- Memory monitoring and alerts

### Phase Ordering Rationale

- **Phase 1 before 2-5:** Foundation must be solid (adapter pattern, database schema, security) before building features on top
- **Phase 2 before 3:** Prove auto-start with local backends (no API key complexity) before adding cloud
- **Phase 3 before 4:** Core routing must work before adding management UI
- **Phase 5 last:** Observability and reliability enhancements require complete system to monitor
- **vLLM-MLX deferred:** Too experimental (74 commits, no production patterns), revisit in v1.3+ when mature

This order avoids pitfalls by solving subprocess management (Phase 1) before on-demand loading (Phase 2), and solving protocol translation (Phase 3) before exposing to users (Phase 4).

### Research Flags

Phases with well-documented patterns (skip additional research):
- **Phase 1:** FastAPI routing, adapter pattern, subprocess management — established patterns with official docs
- **Phase 3:** OpenAI/Anthropic API compatibility — official SDKs and docs provide complete guidance
- **Phase 4:** Frontend CRUD patterns — standard SvelteKit table/form patterns

Phases likely needing deeper research during planning:
- **Phase 2:** Race condition prevention in async startup — may need experimentation with lock timing and queue behavior to avoid deadlocks
- **Phase 5:** Cost tracking implementation — need to decide on pricing data source (hardcoded table vs API), token counting accuracy validation, and cost calculation methodology

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | httpx proxy pattern verified with FastAPI docs + Context7 + recent articles (2025-2026), Anthropic SDK official PyPI (v0.76.0 Jan 2026), Fernet encryption from official cryptography docs |
| Features | HIGH | OpenAI compatibility requirements from official docs + Vertex AI compatibility guide, gateway routing patterns from LiteLLM docs + OpenRouter architecture, streaming format from OpenAI reference |
| Architecture | HIGH | Adapter pattern well-established (Gang of Four, LiteLLM reference implementation), integration points verified by reading existing ServerManager/main.py source code, database patterns match existing SQLModel usage |
| Pitfalls | HIGH | Subprocess management from Python docs + orphanage library, race conditions from vLLM GitHub issues (23697, 25991), streaming timeouts from AWS/FastAPI docs, API compatibility from Anthropic official docs |

**Overall confidence:** HIGH

The research draws heavily from official documentation (Python subprocess, FastAPI, httpx, Anthropic SDK, OpenAI API), established libraries (cryptography.fernet, LiteLLM), and recent 2025-2026 articles confirming current best practices. The architecture leverages existing mlx-manager patterns (ServerManager singleton, async database sessions, FastAPI lifespan), reducing integration risk.

### Gaps to Address

- **vLLM-MLX maturity timeline:** Cannot predict when project will reach production readiness. Recommendation: Monitor GitHub activity quarterly, reconsider when commit count exceeds 300+ and community reports stable production deployments.

- **Model name alias resolution:** Users may want "gpt-4-latest" to resolve dynamically. Not blocking for v1.2 (users can configure explicit model names), but consider adding `gateway_model_aliases` table in v1.3+ if requested.

- **Tool/function calling translation complexity:** OpenAI and Anthropic have significantly different tool calling formats. Phase 3 should include translation layer, but may require deeper research if users need advanced tool features (nested tools, strict schema enforcement, parallel tool calls).

- **Streaming error recovery:** If backend fails mid-stream after client received partial response, can't switch to fallback. May need buffering strategy (defeats streaming purpose) or client-side retry logic. Defer to Phase 5 for user feedback on whether this scenario occurs frequently.

- **Cost tracking data source:** Should pricing be hardcoded table (stale when providers change rates) or fetched from provider APIs (additional dependencies, rate limiting)? Defer decision to Phase 5 implementation, start with hardcoded table for MVP.

## Sources

### Primary (HIGH confidence)
- **Stack Research:** FastAPI lifespan docs, httpx proxy patterns (Medium 2025, Context7), anthropic SDK GitHub (v0.76.0 Jan 2026), cryptography.fernet official docs, mlx-openai-server GitHub (703 commits), vLLM-MLX evaluation (210 stars, 74 commits, experimental stage)
- **Features Research:** OpenAI API streaming reference, Vertex AI OpenAI compatibility guide, LiteLLM router docs, OpenRouter architecture (SaaStr 2025), vLLM sleep mode docs, gateway benchmarks (Maxim 2025)
- **Architecture Research:** Gateway routing pattern (Azure Architecture Center), adapter pattern (Gang of Four, LiteLLM implementation, anthropic_adapter GitHub), vLLM-MLX OpenAI compatibility docs, lazy loading MCP patterns (2026)
- **Pitfalls Research:** Python subprocess docs, python-orphanage library, vLLM race condition issues (#23697, #25991), FastAPI SSE proxy discussions (#10701), AWS API Gateway streaming docs, Anthropic OpenAI SDK compatibility docs, OWASP SSRF prevention cheat sheet

### Secondary (MEDIUM confidence)
- Industry article on MLX dev vs vLLM prod backends (Medium Dec 2025) — recommends keeping them separate, confirms vLLM-MLX experimental status
- Red Hat llm-d intelligent routing paper (Jan 2026) — 45% latency improvement with cache-aware routing, concept solid but implementation details sparse
- Gateway comparison articles (Dev.to 2026, Agenta.ai 2025) — consensus on table stakes features, but specific implementations vary

### Tertiary (LOW confidence, needs validation)
- vllama on-demand loading project (GitHub, low stars) — concept proven but implementation not production-tested, use as reference only
- Gateway latency benchmarks (Maxim Bifrost 11µs, LiteLLM 50µs) — directionally correct but not Apple Silicon specific, validate during implementation

---
*Research completed: 2026-01-26*
*Ready for roadmap: yes*
