# Roadmap: MLX Model Manager v1.2

## Overview

Transform MLX Manager from a local server management tool into a unified API gateway that routes requests to local or cloud backends based on model name. This milestone delivers a production-ready proxy with backend abstraction (mlx-openai-server, vLLM-MLX, OpenAI, Anthropic), on-demand local model auto-start, secure API key storage, and visual configuration UI. The architecture extends existing components (ServerManager, health checker, database) with an adapter pattern for multi-backend routing, streaming support across all providers, and reliability features (error handling, timeouts, failover, audit logging).

## Milestones

- ✅ **v1.1 UX & Auth** - Phases 1-6 (shipped 2026-01-26)
- 🚧 **v1.2 Unified Gateway** - Phases 7-12 (in progress)

## Phases

<details>
<summary>✅ v1.1 UX & Auth (Phases 1-6) - SHIPPED 2026-01-26</summary>

29 requirements delivered across 6 phases:
- Models Panel UX, Server Panel Redesign
- User Authentication (JWT, admin approval)
- Model Discovery (characteristics, badges)
- Chat Multimodal (images, video, MCP tools)
- Bug Fixes (7 stability issues)

Archive: `.planning/milestones/v1.1-ROADMAP.md`

</details>

### 🚧 v1.2 Unified Gateway (In Progress)

**Milestone Goal:** Enable unified API access to local and cloud LLMs with intelligent routing, on-demand model loading, and production-grade reliability.

#### Phase 7: Foundation - Core Gateway Infrastructure

**Goal**: Establish gateway router, adapter pattern, and secure configuration foundation

**Depends on**: Nothing (starts v1.2 milestone)

**Requirements**: BACK-01, CONF-01

**Success Criteria** (what must be TRUE):
1. Gateway has abstract adapter interface defining uniform backend contract (start, stop, health, chat)
2. API keys stored encrypted in database using Fernet symmetric encryption
3. Gateway router resolves model names to backend configurations via database lookup
4. Local MLX adapter integrates with existing ServerManager for health checks and routing
5. Backend URLs validated against SSRF attacks (allowlist/blocklist)

**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

#### Phase 8: Local Gateway - OpenAI-Compatible Routing

**Goal**: Enable OpenAI-compatible chat completions endpoint routing to local mlx-openai-server instances with on-demand auto-start

**Depends on**: Phase 7

**Requirements**: GATE-01, GATE-02, GATE-03, BACK-02, GATE-04

**Success Criteria** (what must be TRUE):
1. User can send OpenAI-formatted request to `/v1/chat/completions` and receive response from local MLX model
2. Gateway streams responses via SSE (server-sent events) without buffering entire response
3. Model name in request determines which MLX server profile handles the request
4. When request arrives for stopped model, gateway automatically starts the server and queues request until healthy
5. Concurrent requests to same stopped model queue without triggering duplicate startups (race condition prevention)

**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD

#### Phase 9: Cloud Backends - OpenAI & Anthropic Integration

**Goal**: Route requests to OpenAI and Anthropic cloud APIs with protocol translation and streaming compatibility

**Depends on**: Phase 8

**Requirements**: BACK-04, BACK-05, GATE-05

**Success Criteria** (what must be TRUE):
1. User can route requests to OpenAI cloud API by configuring model name → OpenAI backend mapping
2. User can route requests to Anthropic cloud API with automatic OpenAI → Anthropic format translation
3. Gateway exposes Anthropic-native `/v1/messages` endpoint for clients using Anthropic SDK
4. Streaming responses work identically across local MLX, OpenAI cloud, and Anthropic cloud backends
5. Connection pooling prevents exhaustion when routing high-volume traffic to cloud APIs

**Plans**: TBD

Plans:
- [ ] 09-01: TBD
- [ ] 09-02: TBD

#### Phase 10: vLLM-MLX Integration

**Goal**: Support vLLM-MLX as alternative local backend for production workloads

**Depends on**: Phase 9

**Requirements**: BACK-03

**Success Criteria** (what must be TRUE):
1. User can configure vLLM-MLX server as backend for model routing
2. vLLM-MLX adapter handles protocol differences from mlx-openai-server transparently
3. Gateway health checks detect vLLM-MLX availability before routing requests

**Plans**: TBD

Plans:
- [ ] 10-01: TBD

#### Phase 11: Configuration UI - Visual Model Routing

**Goal**: Provide frontend UI for managing model → backend mappings, API keys, and routing rules

**Depends on**: Phase 10

**Requirements**: CONF-02, CONF-03, CONF-04, CONF-05

**Success Criteria** (what must be TRUE):
1. User can view all model → backend route mappings in table view showing model pattern, backend type, priority
2. User can add/edit/delete routes via form UI without editing database directly
3. User can configure cloud provider API keys and base URLs through settings panel (encrypted on save)
4. Routing rules support exact match ("gpt-4"), prefix match ("gpt-*"), and regex patterns
5. Model auto-discovery shows available models from running servers to simplify route creation

**Plans**: TBD

Plans:
- [ ] 11-01: TBD
- [ ] 11-02: TBD

#### Phase 12: Production Hardening - Reliability & Observability

**Goal**: Add error handling, timeouts, failover, and audit logging for production deployment

**Depends on**: Phase 11

**Requirements**: RELI-01, RELI-02, RELI-03, RELI-04

**Success Criteria** (what must be TRUE):
1. Gateway returns consistent error responses across all backends (unified format, proper HTTP status codes)
2. Each backend has configurable timeout with safe defaults (local: 15min, cloud: 10min)
3. When local backend fails or is unavailable, gateway automatically routes to configured cloud fallback
4. All gateway requests logged to audit table capturing: timestamp, model, backend, duration, status, token count
5. Admin panel displays request logs with filtering by model, backend, status, and time range

**Plans**: TBD

Plans:
- [ ] 12-01: TBD
- [ ] 12-02: TBD

## Progress

**Execution Order:** Phases execute in numeric order: 7 → 8 → 9 → 10 → 11 → 12

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 7. Foundation | v1.2 | 0/TBD | Not started | - |
| 8. Local Gateway | v1.2 | 0/TBD | Not started | - |
| 9. Cloud Backends | v1.2 | 0/TBD | Not started | - |
| 10. vLLM-MLX | v1.2 | 0/TBD | Not started | - |
| 11. Configuration UI | v1.2 | 0/TBD | Not started | - |
| 12. Hardening | v1.2 | 0/TBD | Not started | - |

---
*Roadmap created: 2026-01-26*
*Last updated: 2026-01-26*
