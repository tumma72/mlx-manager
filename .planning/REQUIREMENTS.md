# Requirements: MLX Model Manager v1.2

**Defined:** 2026-01-26
**Core Value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.

## v1.2 Requirements

Requirements for v1.2 release: Unified API Gateway. Each maps to roadmap phases.

### Gateway Core

- [ ] **GATE-01**: OpenAI-compatible `/v1/chat/completions` endpoint accepts requests and routes to appropriate backend
- [ ] **GATE-02**: Streaming response support (SSE) for chat completions across all backends
- [ ] **GATE-03**: Model name in request determines which backend receives the request (routing logic)
- [ ] **GATE-04**: Local models auto-start on-demand when request arrives for model not currently running
- [ ] **GATE-05**: Anthropic-compatible `/v1/messages` endpoint with message format translation

### Backend Adapters

- [ ] **BACK-01**: Abstract adapter interface defines contract for all backends (start, stop, health, chat)
- [ ] **BACK-02**: mlx-openai-server adapter integrates existing server management with gateway
- [ ] **BACK-03**: vLLM-MLX adapter supports vLLM-MLX as alternative local backend
- [ ] **BACK-04**: OpenAI cloud adapter routes requests to OpenAI API with proper authentication
- [ ] **BACK-05**: Anthropic cloud adapter routes requests to Anthropic API with format translation

### Configuration

- [ ] **CONF-01**: API keys stored encrypted in database (Fernet encryption with env-based master key)
- [ ] **CONF-02**: Model → backend mapping stored in database (which model name routes where)
- [ ] **CONF-03**: Provider configuration UI for managing API keys and base URLs
- [ ] **CONF-04**: Visual route configuration UI for model → backend mappings
- [ ] **CONF-05**: Routing rules support: exact match, prefix match, regex patterns

### Reliability

- [ ] **RELI-01**: Unified error handling returns consistent error responses across all backends
- [ ] **RELI-02**: Configurable timeout per backend (local vs cloud may need different defaults)
- [ ] **RELI-03**: Automatic failover: if local backend fails/unavailable, fallback to configured cloud backend
- [ ] **RELI-04**: Request audit log captures: timestamp, model, backend, duration, status, tokens (if available)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### AI Proxy Agent (v1.3+)

- **AIPROXY-01**: LLM-based intelligent routing (orchestrator model routes requests based on criteria)
- **AIPROXY-02**: Natural language routing rules (e.g., "use Thinking Agent for brainstorming")
- **AIPROXY-03**: Model role configuration (Thinking Agent, Worker Agent, etc.)

### Chat History (Deferred)

- **CHAT-05**: Persist chat history to database
- **CHAT-06**: Chat history sidebar per server
- **CHAT-07**: Server-scoped chats (switching server loads relevant history)
- **CHAT-08**: Create new / delete existing chats

### Advanced Gateway (v1.3+)

- **ADVGATE-01**: Cost tracking and token usage per provider
- **ADVGATE-02**: Rate limiting per provider
- **ADVGATE-03**: Cache-aware intelligent routing
- **ADVGATE-04**: Load balancing across multiple instances of same model

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Every LLM provider | Focus on OpenAI + Anthropic (most common), extensible for future |
| Custom routing DSL | Natural language routing deferred to v1.3 AI Proxy |
| Multi-node clustering | Local-first tool, single machine deployment |
| Usage billing/invoicing | Out of scope for local dev tool |
| OAuth2 for gateway | JWT auth from v1.1 extends to gateway endpoints |

## Traceability

Which phases cover which requirements. Updated by create-roadmap.

| Requirement | Phase | Status |
|-------------|-------|--------|
| GATE-01 | TBD | Pending |
| GATE-02 | TBD | Pending |
| GATE-03 | TBD | Pending |
| GATE-04 | TBD | Pending |
| GATE-05 | TBD | Pending |
| BACK-01 | TBD | Pending |
| BACK-02 | TBD | Pending |
| BACK-03 | TBD | Pending |
| BACK-04 | TBD | Pending |
| BACK-05 | TBD | Pending |
| CONF-01 | TBD | Pending |
| CONF-02 | TBD | Pending |
| CONF-03 | TBD | Pending |
| CONF-04 | TBD | Pending |
| CONF-05 | TBD | Pending |
| RELI-01 | TBD | Pending |
| RELI-02 | TBD | Pending |
| RELI-03 | TBD | Pending |
| RELI-04 | TBD | Pending |

**Coverage:**
- v1.2 requirements: 19 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 19

---
*Requirements defined: 2026-01-26*
*Last updated: 2026-01-26 after initial definition*
