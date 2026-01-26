# Architecture Research: Unified API Gateway

**Domain:** API Gateway for Multi-Backend LLM Routing
**Researched:** 2026-01-26
**Confidence:** HIGH

## Executive Summary

A unified API gateway integrates into mlx-manager's existing layered architecture as a **new router + adapter service layer**. The gateway accepts OpenAI/Anthropic-compatible requests, routes to appropriate backends (local MLX servers, vLLM-MLX, cloud APIs), and translates request/response formats as needed. Integration leverages existing `server_manager` for on-demand model loading and extends the database schema for model routing configuration.

**Key architectural decision:** Gateway lives as a new `/api/gateway` router that proxies to backends, with adapter classes handling protocol translation. Model-to-backend mappings stored in SQLite for persistence.

## Integration Points with Existing Architecture

### 1. Router Layer (NEW: `gateway_router.py`)

**Location:** `backend/mlx_manager/routers/gateway.py`

**Responsibilities:**
- Accept requests at `/api/gateway/v1/chat/completions` (OpenAI-compatible)
- Accept requests at `/api/gateway/v1/messages` (Anthropic-compatible)
- Resolve model name to backend configuration
- Delegate to appropriate adapter service
- Stream responses back to client

**Integration with existing:**
- Mounted in `main.py` alongside existing routers (`auth_router`, `chat_router`, etc.)
- Uses `Depends(get_db)` for database session injection (existing pattern)
- Uses `Depends(get_current_user)` for authentication (existing pattern)

**Pattern match:** Follows existing router pattern (`routers/chat.py` proxies to mlx-openai-server)

### 2. Service Layer (NEW: Adapter Services)

**Location:** `backend/mlx_manager/services/gateway/`

**Structure:**
```
services/gateway/
├── __init__.py
├── adapter_base.py      # Abstract base adapter
├── adapter_mlx.py       # MLX local server adapter
├── adapter_vllm.py      # vLLM-MLX adapter
├── adapter_openai.py    # OpenAI cloud adapter
├── adapter_anthropic.py # Anthropic cloud adapter
├── router_service.py    # Model name → backend resolution
└── format_translator.py # Request/response translation
```

**Integration with existing services:**

| Existing Service | Integration Point | Purpose |
|-----------------|-------------------|---------|
| `server_manager` | Adapter calls `server_manager.start_server()` | Auto-start local MLX models on-demand |
| `health_checker` | Adapter checks backend health before routing | Failover to alternative backend if unhealthy |
| `hf_client` | Gateway UI could trigger downloads | Ensure model exists before creating route |

**Pattern match:** Singleton services instantiated at module level (like `server_manager`, `hf_client`)

### 3. Database Layer (EXTENDED)

**New Tables:**

```python
class GatewayModelRoute(SQLModel, table=True):
    """Model name to backend routing configuration."""
    __tablename__ = "gateway_model_routes"

    id: int | None = Field(default=None, primary_key=True)
    model_name: str = Field(unique=True, index=True)  # e.g., "gpt-4o", "claude-3.5-sonnet"
    backend_type: str  # "mlx_local", "vllm_mlx", "openai", "anthropic"
    backend_config: str  # JSON: {"profile_id": 1} or {"api_key": "...", "base_url": "..."}
    priority: int = Field(default=0)  # For fallback ordering
    enabled: bool = Field(default=True)
    created_at: datetime
    updated_at: datetime

class GatewayRequest(SQLModel, table=True):
    """Request audit log for observability."""
    __tablename__ = "gateway_requests"

    id: int | None = Field(default=None, primary_key=True)
    model_name: str
    backend_type: str
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    latency_ms: float
    status: str  # "success", "error"
    error_message: str | None = None
    created_at: datetime
```

**Integration with existing:**
- Uses existing `get_session()` async session factory
- Uses existing `init_db()` migration pattern (SQLModel creates tables on startup)
- Extends existing `ServerProfile` usage (referenced via `profile_id` in config)

### 4. Configuration (EXTENDED)

**New Settings in `config.py`:**

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Gateway configuration
    gateway_enabled: bool = True
    gateway_default_backend: str = "mlx_local"  # Fallback if no route found
    gateway_request_timeout: int = 300  # seconds
    gateway_enable_audit_log: bool = True

    # Cloud provider credentials (optional)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
```

**Integration:** Extends existing `Settings` class with `MLX_MANAGER_GATEWAY_` prefix for env vars

## New Components Design

### Component 1: Router Service (Model Name Resolution)

**Purpose:** Map incoming model name to backend configuration

**Algorithm:**
```python
async def resolve_backend(model_name: str) -> BackendConfig:
    # 1. Query gateway_model_routes for exact match
    route = await db.get(GatewayModelRoute, model_name=model_name, enabled=True)
    if route:
        return BackendConfig.from_route(route)

    # 2. Check if model_name matches a local profile's model_path
    profiles = await db.query(ServerProfile, model_path__contains=model_name)
    if profiles:
        return BackendConfig(type="mlx_local", profile_id=profiles[0].id)

    # 3. Fuzzy match against known patterns (gpt-* → openai, claude-* → anthropic)
    if model_name.startswith("gpt-"):
        return BackendConfig(type="openai", model=model_name)
    elif model_name.startswith("claude-"):
        return BackendConfig(type="anthropic", model=model_name)

    # 4. Fallback to default backend from config
    return BackendConfig(type=settings.gateway_default_backend)
```

**Confidence:** HIGH - Pattern used by LiteLLM, Conduit, LLM-Gateway

### Component 2: Adapter Base Class

**Purpose:** Abstract interface for backend adapters

**Interface:**
```python
class BackendAdapter(ABC):
    """Abstract adapter for LLM backends."""

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None] | dict:
        """Send chat completion request to backend."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        pass

    @abstractmethod
    def translate_request(self, request: dict) -> dict:
        """Translate gateway request to backend format."""
        pass

    @abstractmethod
    def translate_response(self, response: dict) -> dict:
        """Translate backend response to gateway format."""
        pass
```

**Pattern:** Adapter design pattern (Gang of Four) for protocol translation

**Confidence:** HIGH - Standard pattern, used by anthropic_adapter, LiteLLM

### Component 3: MLX Local Adapter

**Purpose:** Proxy to existing mlx-openai-server instances

**Key Integration:**
```python
class MLXLocalAdapter(BackendAdapter):
    def __init__(self, server_manager: ServerManager, profile: ServerProfile):
        self.server_manager = server_manager
        self.profile = profile

    async def chat_completion(self, model: str, messages: list[dict], stream: bool, **kwargs):
        # 1. Check if server is running
        if not self.server_manager.is_running(self.profile.id):
            # 2. Auto-start server (on-demand loading)
            await self.server_manager.start_server(self.profile)
            # 3. Wait for health check
            await self._wait_for_ready()

        # 4. Proxy request to mlx-openai-server
        url = f"http://{self.profile.host}:{self.profile.port}/v1/chat/completions"
        async with httpx.AsyncClient() as client:
            # ... proxy logic similar to existing chat_router.py ...
```

**On-Demand Loading:** Reuses existing `server_manager.start_server()` - no new process management logic needed

**Confidence:** HIGH - Pattern already exists in `routers/chat.py`

### Component 4: Format Translator

**Purpose:** Translate between OpenAI and Anthropic message formats

**Example Translation (OpenAI → Anthropic):**
```python
def openai_to_anthropic(messages: list[dict]) -> dict:
    """
    OpenAI: [{"role": "user", "content": "Hi"}]
    Anthropic: {"messages": [{"role": "user", "content": "Hi"}]}
    """
    # Extract system message if present
    system = None
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            anthropic_messages.append(msg)

    result = {"messages": anthropic_messages}
    if system:
        result["system"] = system
    return result
```

**Confidence:** HIGH - Well-documented format differences, anthropic_adapter reference implementation

### Component 5: vLLM-MLX Adapter

**Purpose:** Route to external vLLM-MLX servers (OpenAI-compatible)

**Key Finding:** vLLM-MLX maintains OpenAI API compatibility, so adapter is mostly pass-through

```python
class VLLMMLXAdapter(BackendAdapter):
    def __init__(self, base_url: str):
        self.base_url = base_url  # e.g., "http://localhost:8000"

    async def chat_completion(self, model: str, messages: list[dict], stream: bool, **kwargs):
        # vLLM-MLX is OpenAI-compatible, pass through
        url = f"{self.base_url}/v1/chat/completions"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={
                "model": model,
                "messages": messages,
                "stream": stream,
                **kwargs
            })
            # Stream or return response
```

**Confidence:** HIGH - vLLM-MLX documented as OpenAI-compatible (https://github.com/waybarrios/vllm-mlx)

## Data Flow

### Scenario 1: Request to Local MLX Model (Auto-Start)

```
1. Client → POST /api/gateway/v1/chat/completions
   Body: {"model": "mlx-community/Llama-3.2-1B-Instruct-4bit", "messages": [...]}

2. gateway_router.py
   ├─ Authenticate via get_current_user
   ├─ Parse request (OpenAI format)
   └─ Call router_service.resolve_backend("mlx-community/Llama-3.2-1B-Instruct-4bit")

3. router_service.resolve_backend()
   ├─ Query gateway_model_routes (no match)
   ├─ Query ServerProfile WHERE model_path LIKE '%Llama-3.2-1B-Instruct%' → profile_id=1
   └─ Return BackendConfig(type="mlx_local", profile_id=1)

4. gateway_router.py
   └─ Instantiate MLXLocalAdapter(server_manager, profile)

5. MLXLocalAdapter.chat_completion()
   ├─ Check server_manager.is_running(profile_id=1) → False
   ├─ Call server_manager.start_server(profile) → PID 12345 (existing logic)
   ├─ Wait for health check (existing health_checker)
   ├─ Proxy to http://127.0.0.1:10240/v1/chat/completions
   └─ Stream response back

6. gateway_router.py → StreamingResponse → Client
```

**Key Points:**
- Reuses existing `server_manager.start_server()` (no new code)
- Profile lookup by `model_path` (existing DB query pattern)
- On-demand loading: Server started only when first request arrives

### Scenario 2: Request to Cloud Provider (OpenAI)

```
1. Client → POST /api/gateway/v1/chat/completions
   Body: {"model": "gpt-4o", "messages": [...]}

2. gateway_router.py → router_service.resolve_backend("gpt-4o")

3. router_service.resolve_backend()
   ├─ Query gateway_model_routes WHERE model_name="gpt-4o" → route found
   │  backend_type="openai", backend_config={"api_key": "sk-..."}
   └─ Return BackendConfig(type="openai", api_key="sk-...")

4. gateway_router.py
   └─ Instantiate OpenAIAdapter(api_key="sk-...")

5. OpenAIAdapter.chat_completion()
   ├─ Proxy to https://api.openai.com/v1/chat/completions
   └─ Stream response back

6. gateway_router.py → StreamingResponse → Client
```

### Scenario 3: Request with Anthropic Format

```
1. Client → POST /api/gateway/v1/messages
   Body: {"model": "claude-3.5-sonnet", "messages": [...], "system": "You are..."}

2. gateway_router.py
   ├─ Detect Anthropic format endpoint
   └─ router_service.resolve_backend("claude-3.5-sonnet")

3. router_service → BackendConfig(type="anthropic")

4. gateway_router.py → AnthropicAdapter

5. AnthropicAdapter.chat_completion()
   ├─ Request already in Anthropic format, pass through
   ├─ Proxy to https://api.anthropic.com/v1/messages
   └─ Stream response back

6. gateway_router.py → StreamingResponse → Client
```

## Suggested Build Order

### Phase 1: Foundation (Core Gateway Router)
**Goal:** Basic routing infrastructure, single backend type

**Components:**
- `routers/gateway.py` - Basic router with `/v1/chat/completions` endpoint
- `services/gateway/adapter_base.py` - Abstract adapter interface
- `services/gateway/adapter_mlx.py` - MLX local adapter (reuse server_manager)
- `services/gateway/router_service.py` - Simple model name resolution (profile lookup only)
- Database migration - Add `gateway_model_routes` table

**Why first:** Proves integration with existing `server_manager`, establishes patterns

**Deliverable:** Can route `gpt-4o` → local MLX server if profile exists with matching model_path

**Validation:**
- Start local MLX server via gateway proxy
- Gateway auto-starts stopped server on request

### Phase 2: Multi-Backend Support
**Goal:** Add cloud provider adapters

**Components:**
- `services/gateway/adapter_openai.py` - OpenAI cloud adapter
- `services/gateway/adapter_anthropic.py` - Anthropic cloud adapter
- `services/gateway/format_translator.py` - Protocol translation
- `/v1/messages` endpoint for Anthropic format
- Config - Add `openai_api_key`, `anthropic_api_key` settings

**Why second:** Cloud adapters are simpler (no process management), establishes protocol translation

**Deliverable:** Route `gpt-4o` → OpenAI, `claude-3.5-sonnet` → Anthropic

**Validation:**
- Gateway forwards cloud requests correctly
- Format translation works for both directions

### Phase 3: vLLM-MLX Support
**Goal:** External vLLM-MLX server adapter

**Components:**
- `services/gateway/adapter_vllm.py` - vLLM-MLX adapter
- Extended `GatewayModelRoute.backend_config` for base_url

**Why third:** Requires external server setup for testing, simpler than Phase 1/2

**Deliverable:** Route to external vLLM-MLX instances

**Validation:**
- Gateway routes to external vLLM-MLX server
- OpenAI compatibility maintained

### Phase 4: Model Configuration UI
**Goal:** Frontend for managing model routes

**Components:**
- Frontend: Gateway routes table component
- Frontend: Add/edit route form
- Backend: CRUD endpoints for `gateway_model_routes`

**Why fourth:** Core routing works, now add management layer

**Deliverable:** Users can create model routes via UI

**Validation:**
- Add route via UI
- Request routed to configured backend

### Phase 5: Observability & Fallback
**Goal:** Request logging, backend health checks, failover

**Components:**
- `gateway_requests` table for audit log
- Priority-based fallback routing
- Health check integration with `health_checker`
- Frontend: Request logs page

**Why last:** Requires complete routing system to log

**Deliverable:** Production-ready gateway with monitoring

**Validation:**
- Failed backend triggers fallback
- Request logs persisted and viewable

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT (API Consumer)                    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ POST /api/gateway/v1/chat/completions
                                 │      /api/gateway/v1/messages
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GATEWAY ROUTER (NEW)                          │
│  Location: routers/gateway.py                                    │
│  ├─ Authenticate (Depends(get_current_user))                     │
│  ├─ Parse request (OpenAI/Anthropic format)                      │
│  ├─ Resolve backend via RouterService                            │
│  └─ Proxy via Adapter                                            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
        ┌─────────────────────┐  ┌─────────────────────┐
        │  RouterService (NEW) │  │ FormatTranslator    │
        │  Model → Backend     │  │ OpenAI ↔ Anthropic  │
        └──────────┬───────────┘  └─────────────────────┘
                   │
                   │ Query model routes & profiles
                   ▼
        ┌─────────────────────────────────────┐
        │       DATABASE (EXTENDED)            │
        │  ├─ gateway_model_routes (NEW)       │
        │  ├─ server_profiles (EXISTING)       │
        │  └─ gateway_requests (NEW - logs)    │
        └─────────────────────────────────────┘
                   │
                   │ Returns BackendConfig
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTER LAYER (NEW)                           │
│  Location: services/gateway/*.py                                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  BackendAdapter (Abstract Base)                          │   │
│  │  ├─ chat_completion()                                    │   │
│  │  ├─ health_check()                                       │   │
│  │  ├─ translate_request()                                  │   │
│  │  └─ translate_response()                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │             │              │              │            │
│         ▼             ▼              ▼              ▼            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │   MLX    │  │  vLLM-   │  │  OpenAI  │  │Anthropic │        │
│  │  Local   │  │   MLX    │  │  Cloud   │  │  Cloud   │        │
│  └────┬─────┘  └──────────┘  └──────────┘  └──────────┘        │
└───────┼─────────────────────────────────────────────────────────┘
        │
        │ Integrates with existing services
        ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXISTING SERVICES (REUSED)                          │
│  ├─ server_manager (start_server, is_running)  ←── MLX adapter  │
│  ├─ health_checker (check_health)              ←── All adapters │
│  └─ hf_client (download_model)                 ←── Gateway UI   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Spawns/manages processes
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND SERVERS                             │
│  ├─ mlx-openai-server (subprocess) - EXISTING                    │
│  ├─ vLLM-MLX (external server) - NEW                            │
│  ├─ OpenAI API (https://api.openai.com) - NEW                   │
│  └─ Anthropic API (https://api.anthropic.com) - NEW             │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture Patterns Applied

### 1. Adapter Pattern (PRIMARY)
**What:** Adapters translate between gateway interface and backend-specific protocols
**Why:** Each backend (MLX, vLLM-MLX, OpenAI, Anthropic) has different APIs
**Source:** Gang of Four, LiteLLM architecture, anthropic_adapter

### 2. Strategy Pattern
**What:** Router selects adapter dynamically based on model name
**Why:** Backend selection changes per request without changing router code
**Source:** Gateway routing in production LLM gateways

### 3. Proxy Pattern
**What:** Gateway router proxies requests to backend servers
**Why:** Client sees single interface, gateway handles backend complexity
**Source:** Existing `chat_router.py` already uses this pattern

### 4. Backend for Frontend (BFF)
**What:** Gateway provides unified interface customized for client needs
**Why:** Different clients (CLI, UI, API) use same gateway, routing logic centralized
**Source:** API gateway design patterns, production gateways like Portkey

### 5. Lazy Loading (On-Demand Model Loading)
**What:** MLX servers started only when first request arrives
**Why:** Saves memory, allows routing to many models without keeping all loaded
**Source:** MCP server lazy loading patterns (2026), production optimization

## Critical Design Decisions

### Decision 1: Router vs Separate Service
**Choice:** Gateway as FastAPI router in same process
**Alternatives:**
- Separate microservice (own process/port)
- Nginx/Traefik upstream routing

**Rationale:**
- ✅ Reuses existing `server_manager` singleton (same process required)
- ✅ Shares auth, database session, config (FastAPI `Depends`)
- ✅ Simple deployment (no new service to manage)
- ✅ Consistent with existing architecture (layered monolith)
- ❌ Cannot scale gateway independently (acceptable for local/small deployments)

**Confidence:** HIGH - Fits existing architecture

### Decision 2: Configuration Storage (Database vs Config File)
**Choice:** SQLite `gateway_model_routes` table
**Alternatives:**
- YAML/JSON config file
- Hardcoded routing rules

**Rationale:**
- ✅ Dynamic updates via API (no restart required)
- ✅ UI can manage routes (CRUD operations)
- ✅ Consistent with existing patterns (profiles stored in DB)
- ✅ Supports priority/fallback (complex routing)
- ❌ Slightly more complex than config file (acceptable tradeoff)

**Confidence:** HIGH - Standard pattern for dynamic config

### Decision 3: On-Demand Loading vs Pre-Start
**Choice:** Auto-start servers on first request
**Alternatives:**
- Pre-start all configured servers at gateway startup
- Require manual start before routing

**Rationale:**
- ✅ Memory efficient (only load needed models)
- ✅ Supports large model catalogs (dozens of routes)
- ✅ Follows lazy loading best practices (2026 MCP patterns)
- ❌ First request has ~2s latency (model loading time)
- ✅ Subsequent requests fast (server stays warm)

**Confidence:** MEDIUM-HIGH - Tradeoff depends on use case, but matches modern patterns

### Decision 4: Protocol Translation Layer
**Choice:** Dedicated `format_translator.py` service
**Alternatives:**
- Inline translation in each adapter
- Client-side translation

**Rationale:**
- ✅ Single source of truth for format rules
- ✅ Testable in isolation
- ✅ Reusable across adapters
- ✅ Follows separation of concerns

**Confidence:** HIGH - Clean architecture principle

## Anti-Patterns to Avoid

### Anti-Pattern 1: Adapter Logic in Router
**What:** Putting backend-specific code in `gateway_router.py`
**Why bad:** Router becomes bloated, violates single responsibility, hard to test
**Instead:** Router delegates to adapters, stays thin

### Anti-Pattern 2: Synchronous Proxy
**What:** Using `requests` library for backend calls
**Why bad:** Blocks event loop, kills FastAPI performance under load
**Instead:** Use `httpx.AsyncClient` (already used in `chat_router.py`)

### Anti-Pattern 3: Hardcoded Model Routes
**What:** `if model == "gpt-4o": route_to_openai()`
**Why bad:** Not scalable, requires code changes for new models
**Instead:** Database-driven routing with fallback logic

### Anti-Pattern 4: No Health Checks
**What:** Routing to backend without checking availability
**Why bad:** Requests fail instead of falling back
**Instead:** Integrate with `health_checker`, use priority-based fallback

### Anti-Pattern 5: Missing Observability
**What:** No logging of gateway requests
**Why bad:** Cannot debug routing issues or measure latency
**Instead:** Audit log in `gateway_requests` table, include latency tracking

## Scalability Considerations

| Concern | At 1 Local Model | At 5 Local Models | At 20+ Models (Mixed Backends) |
|---------|------------------|-------------------|-------------------------------|
| **Memory** | ~4GB (one server) | ~20GB if all running | On-demand loading: Only loaded models consume RAM |
| **Routing Logic** | O(1) DB lookup | O(1) DB lookup | O(1) indexed query, no performance impact |
| **Process Management** | Existing `server_manager` | Existing `server_manager` | May need process pool limits (config: `max_concurrent_servers`) |
| **Request Latency** | ~50ms proxy overhead | ~50ms proxy overhead | ~50ms + health check (~20ms) if failover |
| **Database Writes** | Audit log per request (~1ms) | Same | Consider batch writes or async logger |

**Bottleneck Analysis:**
- **Phase 1-3:** Process management (MLX servers limited by RAM)
- **Phase 4-5:** Database writes for audit log (solution: async queue, batch insert)

**Optimization Strategies:**
- Keep warm pool of N most-used models (configurable)
- TTL-based server shutdown (stop idle servers after 10min)
- Cache routing decisions (model → backend) with TTL

## Open Questions & Future Research

### Question 1: Tool Calling / Function Calling
**Context:** OpenAI and Anthropic have different tool calling formats
**Research needed:** How to translate tool definitions and responses?
**Impact:** May need extended format translator in Phase 2
**Priority:** Medium - Depends on use case

### Question 2: Streaming Error Handling
**Context:** If backend fails mid-stream, how to handle partial response?
**Research needed:** Can we switch to fallback backend after streaming starts?
**Impact:** May need buffering strategy or client-side retry
**Priority:** High - Affects reliability

### Question 3: Model Alias Resolution
**Context:** Users may want "gpt-4o-latest" to resolve to specific model version
**Research needed:** Should gateway maintain model alias table?
**Impact:** New table `gateway_model_aliases` or extend routing config
**Priority:** Low - Can defer to Phase 5+

### Question 4: Cost Tracking
**Context:** Cloud API calls cost money, users need visibility
**Research needed:** Should gateway track token usage and cost?
**Impact:** Extend `gateway_requests` with cost fields, add pricing config
**Priority:** Medium - Nice to have for production

## Sources

**API Gateway Patterns:**
- [Microservices Pattern: API Gateway](https://microservices.io/patterns/apigateway.html)
- [Gateway Routing pattern - Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/patterns/gateway-routing)
- [API Gateway Pattern - AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/modernization-integrating-microservices/api-gateway-pattern.html)

**LLM Gateway Implementations:**
- [LiteLLM: Python SDK for 100+ LLM APIs](https://github.com/BerriAI/litellm)
- [LLM-API-Key-Proxy: Universal LLM Gateway](https://github.com/Mirrowel/LLM-API-Key-Proxy)
- [Conduit: Unified API gateway for multiple LLM providers](https://github.com/nickna/Conduit)
- [Top LLM Gateways 2025](https://agenta.ai/blog/top-llm-gateways)

**Adapter Pattern & Protocol Translation:**
- [Anthropic Adapter: OpenAI to Anthropic translation](https://github.com/abhiram1809/anthropic_adapter)
- [Connecting Claude Code to Local LLMs](https://medium.com/@michael.hannecke/connecting-claude-code-to-local-llms-two-practical-approaches-faa07f474b0f)

**vLLM & MLX Compatibility:**
- [vLLM-MLX: OpenAI-compatible server for Apple Silicon](https://github.com/waybarrios/vllm-mlx)
- [OpenAI-Compatible Server - vLLM](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)

**On-Demand Loading & Lazy Loading:**
- [Feature Request: Lazy Loading for MCP Servers](https://github.com/anthropics/claude-code/issues/7336)
- [lazy-mcp-preload | MCP Servers](https://lobehub.com/mcp/iamsamuelrodda-lazy-mcp-preload)

**FastAPI Proxy Patterns:**
- [Behind a Proxy - FastAPI](https://fastapi.tiangolo.com/advanced/behind-a-proxy/)
- [fastapi-proxy-lib · PyPI](https://pypi.org/project/fastapi-proxy-lib/)

---

**Research Complete:** 2026-01-26
**Confidence:** HIGH (Core patterns), MEDIUM (vLLM-MLX integration specifics)
**Next Steps:** Proceed to roadmap creation with phase structure from "Suggested Build Order"
