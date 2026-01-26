# Feature Landscape: Unified API Gateway for LLM Inference

**Domain:** LLM API Gateway / Proxy
**Researched:** 2026-01-26
**Confidence:** HIGH

## Executive Summary

Unified API gateways for LLM inference have become foundational infrastructure in 2026, with clear patterns emerging for model routing, format translation, and multi-provider abstraction. The table stakes are OpenAI-compatible endpoints with streaming support, while differentiators focus on intelligent routing, cost optimization, and local-first capabilities.

For mlx-manager v1.2, the unique value proposition is **local-first with cloud fallback** — most gateways target cloud-to-cloud routing, while mlx-manager can provide seamless local Apple Silicon inference with optional cloud overflow.

## Table Stakes

Features users expect from any LLM API gateway. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| **OpenAI-compatible `/v1/chat/completions`** | Industry standard; tools expect this format | Medium | None | LiteLLM, OpenRouter, all major gateways support this |
| **Streaming response support (SSE)** | Required for real-time UX in chat apps | Medium | None | Must emit `data: [DONE]` properly per OpenAI spec |
| **Model name routing** | Gateway's core purpose: route based on model identifier | Medium | Backend registry | Pattern: `provider/model-name` or custom mapping |
| **Request/response format translation** | Convert between OpenAI ↔ Anthropic ↔ provider formats | High | Backend adapters | Critical: preserves tool calls, streaming, multimodal |
| **Error handling & status codes** | Propagate provider errors with context | Low | None | Must distinguish 401 (auth), 429 (rate limit), 5xx (provider down) |
| **API key authentication** | Secure access control | Low | Existing auth | Bearer token in `Authorization` header |
| **Health check endpoint** | Monitor gateway availability | Low | None | `/health` or `/v1/health` returns provider status |
| **Request logging** | Audit trail for debugging | Medium | Logging service | Log request ID, model, latency, tokens, cost |

**Source confidence:** HIGH — Verified via LiteLLM docs, OpenRouter architecture, Gateway API benchmarks

## Differentiators

Features that set mlx-manager apart from other gateways. Not expected, but highly valued.

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **On-demand local model auto-start** | Zero-config: request arrives → model loads → inference begins | High | `server_manager.py`, profile registry | UNIQUE: Most gateways assume backends are always running |
| **Local-first with cloud fallback** | Use local Apple Silicon when available, fallback to cloud | Medium | Routing logic, cloud adapters | UNIQUE: Optimizes cost while ensuring reliability |
| **Unified OpenAI + Anthropic endpoints** | Single gateway exposes both `/v1/chat/completions` AND `/v1/messages` | Medium | Dual format translators | Most gateways pick one format; mlx-manager can offer both |
| **Model auto-discovery from running servers** | Detect available models from mlx-openai-server and vLLM-MLX instances | Medium | Server health checker, model introspection | Reduces manual configuration |
| **Visual model routing configuration** | UI for mapping model names to backends (no config files) | Medium | Frontend + routing API | Better DX than editing YAML/JSON |
| **Cost tracking per backend** | Show local (free) vs cloud (paid) usage with cost breakdown | High | Request logging, provider pricing table | Helps users understand savings from local inference |
| **Intelligent cache-aware routing** | Route repeat prompts to same backend for KV cache hits | Very High | Request hashing, llm-d-style routing | 45% latency improvement possible (Red Hat research) |
| **Automatic provider failover** | If OpenAI fails, retry with Anthropic; if local OOM, use cloud | Medium | Retry logic, health checks | Industry standard but critical for reliability |

**Source confidence:** MEDIUM-HIGH — LiteLLM features verified (HIGH), local-first patterns inferred from vLLM sleep mode and vllama project (MEDIUM)

## Anti-Features

Features to explicitly NOT build. Common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Building custom LLM router logic from scratch** | Gateway vendors like LiteLLM, OpenRouter have 3+ years of edge case handling | Use LiteLLM as library or study their adapter patterns; focus on local-first differentiation |
| **Supporting every possible LLM provider** | Maintenance burden explodes; most users need 2-4 providers | Focus on: mlx-openai-server, vLLM-MLX, OpenAI, Anthropic. Add others based on user demand |
| **Implementing gateway AS a Python package dependency** | Creates version conflicts, bloats user environments | Keep as standalone service; expose HTTP API |
| **Synchronous model loading** | Blocks API response for 30+ seconds while model loads | Use async task queue; return 202 Accepted → poll for ready state |
| **Storing cloud API keys in database without encryption** | Security vulnerability | Use system keychain (macOS Keychain) or encrypt at rest with per-user keys |
| **Adding 50ms+ latency per request** | Python-based LiteLLM adds ~50µs; bad implementations add 50ms+ | Benchmark early; use async I/O; avoid unnecessary serialization |
| **Making gateway a single point of failure** | If gateway crashes, all inference stops | Ensure gateway restarts quickly (launchd service); document manual bypass (direct access to mlx-openai-server ports) |
| **Auto-scaling to expensive models** | Users surprised by $100 bills from GPT-4 usage | Require explicit user confirmation for cloud model routing; show cost estimates |
| **Complex prompt rewriting/modification** | Gateway shouldn't second-guess user intent; introduces unpredictability | Log prompts, provide analytics, but don't modify unless explicitly requested (guardrails feature) |
| **Trying to compete on gateway latency alone** | Maxim Bifrost (11µs) uses Go; Python can't match that | Focus on feature differentiation (local-first, auto-start, cost savings) not raw speed |

**Source confidence:** HIGH — Anti-patterns documented in LLM Gateway pitfalls, local LLM deployment mistakes, gateway benchmark comparisons

## Feature Complexity Matrix

Sorted by implementation complexity and value.

| Feature | Complexity | Value | Priority | Estimated Effort | Dependencies |
|---------|-----------|-------|----------|------------------|--------------|
| **OpenAI `/v1/chat/completions` endpoint** | Medium | Critical | P0 | 3-5 days | Backend adapter pattern |
| **Model name → backend routing** | Medium | Critical | P0 | 2-3 days | Profile registry |
| **Request format translation (OpenAI ↔ native)** | High | Critical | P0 | 5-7 days | Per-backend adapters |
| **Streaming SSE support** | Medium | Critical | P0 | 2-4 days | Async response handling |
| **API key auth for gateway** | Low | High | P0 | 1-2 days | Existing JWT system |
| **On-demand model auto-start** | High | High | P1 | 5-8 days | `server_manager.py` integration |
| **Health check endpoint** | Low | High | P1 | 1 day | None |
| **Error handling & propagation** | Low | High | P1 | 2-3 days | Exception mapping |
| **Cloud provider adapters (OpenAI, Anthropic)** | Medium | High | P1 | 4-6 days | HTTP client, format translation |
| **Request logging** | Medium | High | P1 | 2-3 days | Logging service |
| **Model auto-discovery** | Medium | Medium | P2 | 3-4 days | Health checker, model introspection |
| **Visual routing configuration UI** | Medium | Medium | P2 | 4-5 days | Frontend + API |
| **Anthropic `/v1/messages` endpoint** | Medium | Medium | P2 | 3-4 days | Anthropic format translator |
| **Cost tracking** | High | Medium | P3 | 5-7 days | Usage logging, pricing table |
| **Automatic failover** | Medium | Low | P3 | 3-4 days | Retry logic |
| **Cache-aware routing** | Very High | Low | P4/Deferred | 10-15 days | Request hashing, llm-d-style scheduler |

**Total estimated effort (P0-P1):** 28-48 days

## Feature Dependencies

Dependency graph showing what must be built first:

```
Foundation Layer (Phase 1):
├─ Backend adapter pattern (interface + base class)
├─ Model registry (maps model names to backends)
└─ Request/response schemas (OpenAI format)

Core Gateway (Phase 2):
├─ OpenAI `/v1/chat/completions` endpoint
│  ├─ Requires: Backend adapter pattern
│  ├─ Requires: Model registry
│  └─ Requires: Request format translation
├─ Streaming SSE support
│  └─ Requires: OpenAI endpoint
└─ Error handling
   └─ Requires: OpenAI endpoint

Local Backend Integration (Phase 3):
├─ mlx-openai-server adapter
│  └─ Requires: Backend adapter pattern
├─ vLLM-MLX adapter
│  └─ Requires: Backend adapter pattern
└─ On-demand auto-start
   ├─ Requires: mlx-openai-server adapter
   └─ Requires: server_manager.py

Cloud Backend Integration (Phase 4):
├─ OpenAI cloud adapter
│  ├─ Requires: Backend adapter pattern
│  └─ Requires: API key storage
└─ Anthropic cloud adapter
   ├─ Requires: Backend adapter pattern
   └─ Requires: API key storage

Enhancements (Phase 5+):
├─ Visual routing UI
│  └─ Requires: Core gateway
├─ Cost tracking
│  └─ Requires: Request logging
└─ Automatic failover
   └─ Requires: Multiple backend adapters
```

## MVP Recommendation

For mlx-manager v1.2 MVP, prioritize:

### Must Have (Minimum Viable Gateway)
1. **OpenAI-compatible `/v1/chat/completions` endpoint** — Industry standard
2. **Model name → backend routing** — Core functionality
3. **Streaming support** — Expected for chat UX
4. **mlx-openai-server adapter** — Leverages existing infrastructure
5. **OpenAI cloud adapter** — Demonstrates local + cloud capability
6. **API key auth** — Secure access
7. **Request logging** — Debugging and monitoring

### Should Have (Differentiation)
8. **On-demand model auto-start** — Unique value for local-first
9. **Model auto-discovery** — Reduces manual config
10. **Error handling with context** — Production-ready

### Nice to Have (Defer to v1.3+)
- Anthropic `/v1/messages` endpoint (can route through OpenAI format initially)
- Visual routing configuration UI (start with API + env vars)
- Cost tracking (add after MVP validates demand)
- Cache-aware routing (optimization for v2.0)

## Integration with Existing mlx-manager Features

| Existing Feature | Gateway Integration Point | Notes |
|-----------------|--------------------------|-------|
| **Server profiles** | Gateway treats each profile as a backend | Profile already has `name`, `model_id`, `port` — perfect for routing |
| **Server lifecycle (start/stop)** | On-demand auto-start calls `server_manager.py` | Reuse existing process management |
| **Health checker** | Gateway queries health to know if backend is ready | Extend to check `/v1/models` endpoint |
| **User authentication** | Gateway requires valid JWT token | Reuse existing JWT middleware |
| **Model download** | Gateway can trigger download if model not found locally | Async workflow: 404 → download → retry |
| **Chat interface** | Chat can switch from direct server access to gateway proxy | Transparent to user; improves reliability |

## Model Name Routing Patterns

Research shows two dominant patterns in 2026:

### Pattern 1: Provider Prefix (OpenRouter, LiteLLM)
```
gpt-4-turbo → openai/gpt-4-turbo
claude-3-opus → anthropic/claude-3-opus-20240229
mlx-community/Qwen2.5-7B → local/mlx-community/Qwen2.5-7B
```

### Pattern 2: Custom Aliases (mlx-manager approach)
```
User configures: "my-fast-model" → Profile "Qwen 7B" → mlx-openai-server on port 10240
User configures: "my-smart-model" → OpenAI "gpt-4-turbo"
```

**Recommendation:** Use Pattern 2 for better UX — users define memorable aliases mapped to backends. Gateway resolves:
1. Check local profile registry
2. If no match, check cloud provider patterns (gpt-* → OpenAI, claude-* → Anthropic)
3. If no match, return 404 with helpful message

## OpenAI API Compatibility Expectations

Based on 2026 standards, OpenAI-compatible gateways must support:

| Feature | Required | Notes |
|---------|----------|-------|
| **POST /v1/chat/completions** | Yes | Core endpoint |
| **Streaming (stream=true)** | Yes | SSE format with `data:` prefix |
| **Function/tool calling** | Yes | `tools` parameter + `tool_calls` in response |
| **Vision (image inputs)** | Recommended | `content` array with `type: "image_url"` |
| **Multimodal (image + text)** | Recommended | Already supported in mlx-manager chat UI |
| **System messages** | Yes | Role-based message format |
| **Temperature, top_p, max_tokens** | Yes | Standard sampling parameters |
| **Stop sequences** | Recommended | `stop` parameter |
| **Seed for reproducibility** | Optional | `seed` parameter (not all backends support) |
| **Response format (JSON mode)** | Optional | `response_format: {type: "json_object"}` |

**Compatibility testing:** Use OpenAI Python SDK against gateway to verify 100% compatibility.

## Anthropic API Compatibility Expectations

If supporting `/v1/messages`:

| Feature | Required | Notes |
|---------|----------|-------|
| **POST /v1/messages** | Yes | Core endpoint |
| **Streaming** | Yes | SSE format similar to OpenAI |
| **Tool use** | Yes | `tools` parameter + `tool_use` content blocks |
| **Vision** | Yes | `image` content type in messages |
| **System prompts** | Yes | Separate `system` parameter (not in messages array) |
| **Max tokens** | Yes | Required parameter (no default) |
| **Thinking models** | Optional | Extended reasoning support |

**Translation complexity:** Anthropic's message format differs from OpenAI:
- OpenAI uses `role: system/user/assistant`
- Anthropic uses `system` parameter + `messages` array (user/assistant only)
- Tool call format differs significantly

## On-Demand Model Loading Patterns

Research shows two approaches in 2026:

### Approach 1: vLLM Sleep Mode (Fast Resume)
- Level 1: Offload weights to CPU RAM, discard KV cache (90% GPU freed)
- Level 2: Discard weights entirely (for different model)
- Resume latency: ~1-2 seconds for Level 1

### Approach 2: vllama On-Demand Loading (Full Lifecycle)
- Model loaded on first request (cold start: 30-60 seconds)
- Auto-unload after 5 minutes idle
- Optimized for multi-model scenarios with limited VRAM

**Recommendation for mlx-manager:**
- **Phase 1 (v1.2):** Auto-start server if not running (cold start acceptable)
- **Phase 2 (v1.3+):** Implement warm server pool with vLLM sleep mode for faster resume

Implementation:
```
Request arrives for model "my-fast-model"
→ Resolve to Profile ID 3 (Qwen 7B on port 10240)
→ Check health: server not running
→ Call server_manager.start_server(profile_id=3)
→ Wait for health check success (poll /health every 1s, timeout 60s)
→ Proxy request to localhost:10240/v1/chat/completions
→ Stream response back to client
```

## Security Considerations

| Concern | Mitigation | Priority |
|---------|-----------|----------|
| **Cloud API keys in database** | Encrypt at rest; use macOS Keychain for storage | P0 |
| **Gateway API key exposure** | Require JWT auth; rate limit per user | P0 |
| **Prompt injection via model names** | Validate model names against allowlist; sanitize inputs | P1 |
| **SSRF via custom backend URLs** | Restrict to localhost + configured cloud endpoints | P1 |
| **Cost overruns** | Per-user budgets; require confirmation for cloud routing | P2 |

## Performance Benchmarks (2026 Context)

Gateway latency expectations:

| Gateway Type | Overhead | Notes |
|--------------|----------|-------|
| **Go-based (Bifrost)** | 11µs | Industry leading |
| **Python async (LiteLLM)** | ~50µs | Acceptable for most use cases |
| **Python sync (bad implementation)** | 50ms+ | Avoid this |

**Target for mlx-manager:** <1ms gateway overhead (Python async should achieve this easily with proper implementation)

## Research Confidence Assessment

| Area | Confidence | Sources |
|------|-----------|---------|
| **OpenAI compatibility requirements** | HIGH | Official OpenAI docs, Vertex AI compatibility guide |
| **Gateway routing patterns** | HIGH | LiteLLM docs, OpenRouter architecture, Gateway benchmarks |
| **Streaming format** | HIGH | OpenAI streaming events reference |
| **On-demand loading** | MEDIUM | vLLM sleep mode docs (HIGH), vllama project (MEDIUM) |
| **Cost tracking** | MEDIUM | Inferred from gateway comparison articles |
| **Cache-aware routing** | MEDIUM | Red Hat llm-d research paper (HIGH for concept, MEDIUM for implementation details) |

## Sources

### Core Gateway Patterns
- [App of the Week: OpenRouter — The Universal API for All Your LLMs](https://www.saastr.com/app-of-the-week-openrouter-the-universal-api-for-all-your-llms/)
- [Best LLM Gateways in 2025: Features, Benchmarks, and Builder's Guide](https://www.getmaxim.ai/articles/best-llm-gateways-in-2025-features-benchmarks-and-builders-guide/)
- [What is LLM Gateway ? How Does It Work ?](https://www.truefoundry.com/blog/llm-gateway)
- [Top 5 LLM Gateways in 2026: A Deep-Dive Comparison](https://dev.to/varshithvhegde/top-5-llm-gateways-in-2026-a-deep-dive-comparison-for-production-teams-34d2)
- [Top LLM Gateways 2025](https://agenta.ai/blog/top-llm-gateways)

### OpenAI API Compatibility
- [OpenAI-Compatible Endpoints | liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)
- [Function calling | OpenAI API](https://platform.openai.com/docs/guides/function-calling)
- [Streaming API responses | OpenAI API](https://platform.openai.com/docs/guides/streaming-responses)
- [OpenAI compatibility | Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/openai)

### Format Translation
- [GitHub - BerriAI/litellm](https://github.com/BerriAI/litellm)
- [Anthropic | liteLLM](https://docs.litellm.ai/docs/providers/anthropic)
- [GitHub - looplj/axonhub](https://github.com/looplj/axonhub)

### Load Balancing & Routing
- [Router - Load Balancing | liteLLM](https://docs.litellm.ai/docs/routing)
- [Accelerate multi-turn LLM workloads with llm-d intelligent routing](https://developers.redhat.com/articles/2026/01/13/accelerate-multi-turn-workloads-llm-d)
- [Load balancing in multi-LLM setups](https://portkey.ai/blog/llm-load-balancing/)

### On-Demand Model Loading
- [Sleep Mode - vLLM](https://docs.vllm.ai/en/latest/features/sleep_mode/)
- [vLLM Quickstart](https://www.glukhov.org/post/2026/01/vllm-quickstart/)
- [GitHub - erkkimon/vllama](https://github.com/erkkimon/vllama)

### Security & Best Practices
- [API Gateway Security Best Practices for 2026](https://www.practical-devsecops.com/api-gateway-security-best-practices/)
- [Understanding Core API Gateway Features](https://api7.ai/learning-center/api-gateway-guide/core-api-gateway-features)

### Anti-Patterns & Pitfalls
- [Common mistakes in local LLM deployments](https://sebastianpdw.medium.com/common-mistakes-in-local-llm-deployments-03e7d574256b)
- [AI Systems Engineering Patterns](https://blog.alexewerlof.com/p/ai-systems-engineering-patterns)

---

**Last Updated:** 2026-01-26
**Next Review:** After MVP implementation (gather user feedback on feature priorities)
