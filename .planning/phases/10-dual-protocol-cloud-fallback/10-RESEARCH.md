# Phase 10: Dual Protocol & Cloud Fallback - Research

**Researched:** 2026-01-29
**Domain:** Protocol translation, cloud backend routing, automatic failover
**Confidence:** HIGH

## Summary

This phase implements Anthropic API compatibility via a `/v1/messages` endpoint and cloud backend fallback routing. The existing codebase has a solid foundation with OpenAI-compatible endpoints, SSE streaming via `sse-starlette`, and Pydantic v2 schemas. The Anthropic Messages API differs significantly in message structure (content blocks vs simple content) and streaming format (typed events vs `data:` only).

Key research findings:
1. **Protocol translation is well-understood** - mlx-omni-server and LiteLLM demonstrate proven patterns for bidirectional OpenAI/Anthropic conversion
2. **SSE streaming differs significantly** - Anthropic uses named event types (`event: content_block_delta`) while OpenAI uses only `data:` lines
3. **Cloud routing requires httpx with retries** - Use `httpx-retries` or `tenacity` for exponential backoff; consider `pybreaker` for circuit breaker
4. **Database schema extension is straightforward** - Add `BackendMapping` table for model-to-backend routing configuration

**Primary recommendation:** Build a `ProtocolTranslator` service with bidirectional conversion, use `httpx.AsyncClient` with retry transport for cloud backends, and store backend mappings in SQLite alongside existing profile data.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| httpx | 0.28+ | Async HTTP client for cloud backends | Already in project; native async, HTTP/2 |
| httpx-retries | 0.4+ | Retry transport with exponential backoff | Clean integration with httpx.AsyncClient |
| sse-starlette | 2.0+ | Server-sent events for streaming | Already in project; supports named events |
| pydantic | 2.0+ | Request/response validation | Already used; Rust core for speed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pybreaker | 1.0+ | Circuit breaker pattern | When cloud backend fails repeatedly |
| tenacity | 8.0+ | Advanced retry policies | If httpx-retries insufficient |
| cryptography | 41+ | API key encryption | For secure storage of cloud API keys |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| httpx-retries | tenacity | tenacity more flexible but requires manual httpx integration |
| pybreaker | circuitbreaker | circuitbreaker uses decorators; pybreaker has Redis state storage |
| Custom SSE | anthropic-sdk | SDK adds dependency; custom gives full control |

**Installation:**
```bash
pip install httpx-retries pybreaker
# Note: httpx, sse-starlette, pydantic already in project
```

## Architecture Patterns

### Recommended Project Structure
```
mlx_manager/mlx_server/
├── api/v1/
│   ├── chat.py           # Existing OpenAI chat endpoint
│   ├── messages.py       # NEW: Anthropic messages endpoint
│   └── ...
├── schemas/
│   ├── openai.py         # Existing OpenAI schemas
│   └── anthropic.py      # NEW: Anthropic request/response schemas
├── services/
│   ├── protocol.py       # NEW: Protocol translator
│   ├── cloud/
│   │   ├── __init__.py
│   │   ├── client.py     # NEW: Cloud backend client with retries
│   │   ├── openai.py     # NEW: OpenAI cloud backend
│   │   ├── anthropic.py  # NEW: Anthropic cloud backend
│   │   └── router.py     # NEW: Backend router with failover
│   └── ...
└── ...
```

### Pattern 1: Protocol Translator (Bidirectional)
**What:** Service that converts between OpenAI and Anthropic formats
**When to use:** Any endpoint accepting one format but needing the other
**Example:**
```python
# Source: Pattern from LiteLLM and mlx-omni-server
from pydantic import BaseModel

class ProtocolTranslator:
    """Bidirectional translation between OpenAI and Anthropic formats."""

    def anthropic_to_internal(
        self, request: AnthropicMessagesRequest
    ) -> InternalRequest:
        """Convert Anthropic Messages API to internal format."""
        messages = []

        # Handle system prompt (Anthropic has separate field)
        if request.system:
            if isinstance(request.system, str):
                messages.append({"role": "system", "content": request.system})
            else:
                # Array of TextBlockParam
                text = " ".join(b.text for b in request.system)
                messages.append({"role": "system", "content": text})

        # Convert content blocks to simple content
        for msg in request.messages:
            content = self._extract_text_content(msg.content)
            messages.append({"role": msg.role, "content": content})

        return InternalRequest(
            model=request.model,
            messages=messages,
            max_tokens=request.max_tokens,  # Required in Anthropic
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            stop=request.stop_sequences,
        )

    def internal_to_anthropic_event(
        self, token: str, index: int, request_id: str
    ) -> str:
        """Convert internal token to Anthropic SSE event."""
        delta = {"type": "text_delta", "text": token}
        data = {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        }
        return f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"
```

### Pattern 2: Cloud Backend Client with Retries
**What:** httpx AsyncClient with automatic retry and exponential backoff
**When to use:** All cloud API calls (OpenAI, Anthropic)
**Example:**
```python
# Source: httpx-retries documentation
from httpx_retries import Retry, RetryTransport
import httpx

class CloudBackendClient:
    """Cloud backend client with retry and circuit breaker."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            # Retry on these status codes
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=RetryTransport(retry=retry),
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def chat_completion(
        self, request: dict, stream: bool = False
    ) -> AsyncGenerator[str, None] | dict:
        """Send chat completion request to cloud backend."""
        if stream:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=request
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        yield line
        else:
            response = await self._client.post(
                "/v1/chat/completions", json=request
            )
            response.raise_for_status()
            return response.json()
```

### Pattern 3: Backend Router with Failover
**What:** Routes requests to appropriate backend with automatic failover
**When to use:** All inference requests that may need cloud fallback
**Example:**
```python
# Source: Resilience patterns for APIs
from pybreaker import CircuitBreaker

class BackendRouter:
    """Routes requests with automatic failover."""

    def __init__(self, db_session):
        self._db = db_session
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    async def route_request(
        self, model_id: str, request: InternalRequest
    ) -> AsyncGenerator | dict:
        """Route to local or cloud based on mapping and availability."""
        mapping = await self._get_backend_mapping(model_id)

        # Try local first if configured
        if mapping.backend_type == "local":
            try:
                return await self._local_inference(request)
            except (MemoryError, RuntimeError) as e:
                if mapping.fallback_backend:
                    logger.warning(f"Local failed, falling back: {e}")
                    return await self._cloud_inference(
                        mapping.fallback_backend, request
                    )
                raise

        # Cloud backend
        return await self._cloud_inference(mapping.backend_type, request)
```

### Anti-Patterns to Avoid
- **Hardcoded API keys:** Never embed keys in code; use encrypted database storage or environment variables
- **Synchronous cloud calls:** Always use async httpx; blocking calls degrade server performance
- **No retry on transient failures:** Cloud APIs have rate limits and transient errors; always retry with backoff
- **Mixing SSE formats:** Anthropic and OpenAI SSE differ; don't mix `event:` lines with bare `data:`

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP retries | Custom retry loop | httpx-retries | Handles backoff, jitter, status codes correctly |
| Circuit breaker | Simple counter | pybreaker | Proper state machine, configurable thresholds |
| API key encryption | Simple base64 | AuthLib/cryptography | Secure key derivation, proper encryption |
| SSE parsing | String splitting | httpx-sse | Handles edge cases, multi-byte chars |
| Anthropic schemas | Manual dicts | Pydantic models | Validation, type safety, IDE support |

**Key insight:** Cloud API integration has many edge cases (rate limits, partial responses, timeouts). Using established libraries prevents subtle bugs that appear under load.

## Common Pitfalls

### Pitfall 1: SSE Event Format Mismatch
**What goes wrong:** Anthropic clients fail to parse OpenAI-style SSE
**Why it happens:** OpenAI uses only `data: {...}` lines; Anthropic requires `event: type\ndata: {...}`
**How to avoid:** Use separate event generators for each protocol; test with actual Anthropic SDK
**Warning signs:** Streaming works with curl but not with Anthropic Python SDK

### Pitfall 2: Missing max_tokens in Anthropic Requests
**What goes wrong:** 400 error on Anthropic endpoint
**Why it happens:** Anthropic requires `max_tokens`; OpenAI makes it optional
**How to avoid:** Schema validation with `max_tokens: int` (no default, required field)
**Warning signs:** Works with OpenAI SDK, fails with Anthropic SDK

### Pitfall 3: Content Block vs String Content
**What goes wrong:** Parsing errors when content is array vs string
**Why it happens:** Anthropic allows `content: string | ContentBlock[]`; code assumes one type
**How to avoid:** Always normalize content to internal format before processing
**Warning signs:** Text messages work but multimodal messages fail

### Pitfall 4: Cloud API Key Leakage
**What goes wrong:** API keys exposed in logs or error messages
**Why it happens:** Keys included in request headers, logged on error
**How to avoid:** Use dedicated secrets manager; mask keys in logs; never log full request headers
**Warning signs:** Keys visible in debug output or error responses

### Pitfall 5: Circuit Breaker Shared State
**What goes wrong:** One user's failures affect all users
**Why it happens:** Single circuit breaker instance shared across all requests
**How to avoid:** Per-backend circuit breakers; consider per-model if needed
**Warning signs:** Cloud fallback stops working for everyone when one user has issues

### Pitfall 6: Stop Reason Translation
**What goes wrong:** Incorrect finish reason in translated responses
**Why it happens:** OpenAI uses `stop`, Anthropic uses `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`
**How to avoid:** Explicit mapping table for stop reasons
**Warning signs:** Client interprets response as incomplete when it's actually done

## Code Examples

Verified patterns from official sources:

### Anthropic Messages Request Schema
```python
# Source: https://platform.claude.com/docs/en/api/messages
from pydantic import BaseModel, Field
from typing import Literal

class TextBlockParam(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageBlockParam(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSource

class MessageParam(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[TextBlockParam | ImageBlockParam]

class AnthropicMessagesRequest(BaseModel):
    model: str
    max_tokens: int = Field(ge=1)  # Required in Anthropic
    messages: list[MessageParam]
    system: str | list[TextBlockParam] | None = None
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: dict | None = None
```

### Anthropic Messages Response Schema
```python
# Source: https://platform.claude.com/docs/en/api/messages
class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[TextBlock]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    stop_sequence: str | None = None
    usage: Usage
```

### Anthropic SSE Streaming Events
```python
# Source: https://platform.claude.com/docs/en/api/messages-streaming
from sse_starlette.sse import EventSourceResponse

async def anthropic_stream_generator(
    request_id: str,
    model: str,
    token_stream: AsyncGenerator[str, None],
) -> AsyncGenerator[dict, None]:
    """Generate Anthropic-format SSE events."""

    # 1. message_start
    yield {
        "event": "message_start",
        "data": json.dumps({
            "type": "message_start",
            "message": {
                "id": request_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 1},
            },
        }),
    }

    # 2. content_block_start
    yield {
        "event": "content_block_start",
        "data": json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
    }

    # 3. content_block_delta events (main content)
    output_tokens = 0
    async for token in token_stream:
        output_tokens += 1
        yield {
            "event": "content_block_delta",
            "data": json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": token},
            }),
        }

    # 4. content_block_stop
    yield {
        "event": "content_block_stop",
        "data": json.dumps({
            "type": "content_block_stop",
            "index": 0,
        }),
    }

    # 5. message_delta (final usage)
    yield {
        "event": "message_delta",
        "data": json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }),
    }

    # 6. message_stop
    yield {
        "event": "message_stop",
        "data": json.dumps({"type": "message_stop"}),
    }
```

### Backend Mapping Database Model
```python
# Source: Project patterns from models.py
from sqlmodel import SQLModel, Field
from datetime import UTC, datetime
from enum import Enum

class BackendType(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class BackendMapping(SQLModel, table=True):
    """Maps model names to backends with fallback configuration."""

    __tablename__ = "backend_mappings"

    id: int | None = Field(default=None, primary_key=True)
    model_pattern: str = Field(index=True)  # e.g., "gpt-*" or exact name
    backend_type: BackendType
    backend_model: str | None = None  # Override model name for cloud
    fallback_backend: BackendType | None = None
    priority: int = Field(default=0)  # Higher = checked first
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
```

### Cloud Backend Credentials (Encrypted)
```python
# Source: AuthLib pattern from project
from sqlmodel import SQLModel, Field
from datetime import UTC, datetime

class CloudCredential(SQLModel, table=True):
    """Encrypted cloud API credentials."""

    __tablename__ = "cloud_credentials"

    id: int | None = Field(default=None, primary_key=True)
    backend_type: BackendType = Field(unique=True)
    encrypted_api_key: str  # Encrypted with AuthLib
    base_url: str | None = None  # Override default API URL
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual retry loops | httpx-retries transport | 2024 | Cleaner code, proper backoff |
| Synchronous requests | httpx.AsyncClient | 2022 | Non-blocking cloud calls |
| String concatenation SSE | sse-starlette EventSourceResponse | 2023 | Proper event naming |
| Hardcoded cloud URLs | Configurable base URLs | 2025 | Supports Azure OpenAI, proxies |

**Deprecated/outdated:**
- `requests` library: Use `httpx` for async support
- Manual SSE formatting: Use `sse-starlette` for proper event handling
- Synchronous circuit breakers: Use `purgatory` or async-compatible `pybreaker`

## Open Questions

Things that couldn't be fully resolved:

1. **Cost tracking data source**
   - What we know: Need to track costs per request for cloud backends
   - What's unclear: Hardcoded pricing table vs API fetch vs token-based estimation
   - Recommendation: Start with hardcoded pricing table; add API fetch later if needed

2. **Tool use translation complexity**
   - What we know: Both APIs support tools but formats differ significantly
   - What's unclear: Whether to support full tool translation or text-only initially
   - Recommendation: Implement text streaming first; add tool support in follow-up

3. **Extended thinking support**
   - What we know: Anthropic has `thinking` parameter for chain-of-thought
   - What's unclear: How to represent this in OpenAI format for cloud fallback
   - Recommendation: Support in Anthropic endpoint; skip in OpenAI translation

## Sources

### Primary (HIGH confidence)
- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages) - Complete request/response spec
- [Anthropic Streaming](https://platform.claude.com/docs/en/api/messages-streaming) - SSE event types and format
- [httpx-retries GitHub](https://github.com/will-ockmore/httpx-retries) - Retry transport API

### Secondary (MEDIUM confidence)
- [mlx-omni-server GitHub](https://github.com/madroidmaq/mlx-omni-server) - Anthropic adapter patterns
- [LiteLLM Anthropic](https://docs.litellm.ai/docs/providers/anthropic) - Format translation patterns
- [pybreaker PyPI](https://pypi.org/project/pybreaker/) - Circuit breaker implementation

### Tertiary (LOW confidence)
- Web search results on circuit breaker patterns - Needs validation with actual load testing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Libraries already in project or well-documented
- Architecture: HIGH - Patterns proven in mlx-omni-server, LiteLLM
- Pitfalls: HIGH - Documented in official API docs and community issues
- Cloud fallback: MEDIUM - Patterns established but need load testing

**Research date:** 2026-01-29
**Valid until:** 2026-02-28 (30 days - stable domain, well-documented APIs)
