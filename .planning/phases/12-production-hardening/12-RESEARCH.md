# Phase 12: Production Hardening - Research

**Researched:** 2026-01-30
**Domain:** Observability, Error Handling, Timeouts, Audit Logging
**Confidence:** HIGH

## Summary

This phase implements production hardening features for the MLX Manager: Pydantic LogFire integration for observability, unified RFC 7807-style error responses, per-endpoint configurable timeouts, and request audit logging with an admin panel.

LogFire is already partially integrated (v3.0.0+ as a dependency with basic FastAPI instrumentation in the MLX server). This research confirms the approach for expanding that integration and implementing the remaining requirements. The project already uses SQLModel/SQLAlchemy with aiosqlite, httpx for HTTP clients, and sse-starlette for streaming - all have LogFire instrumentation available.

Key findings:
- LogFire v4.21.0 (Jan 2026) provides native instrumentation for FastAPI, HTTPX, SQLAlchemy, OpenAI, and Anthropic
- RFC 7807/9457 Problem Details is the standard for structured API errors - use `fastapi-rfc7807` middleware
- Timeouts require custom implementation using `asyncio.wait_for()` since FastAPI has no built-in per-endpoint timeouts
- Audit logging should use background tasks for non-blocking database writes

**Primary recommendation:** Expand existing LogFire integration to both apps, add RFC 7807 error middleware, implement timeout wrapper decorator, and add audit log model with background writes.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| logfire[fastapi] | ^3.0.0 (already installed) | Full-stack observability | Pydantic-native, OpenTelemetry-based, LLM token tracking |
| fastapi-rfc7807 | ^1.4.0+ | RFC 7807 error responses | Automatic exception translation to Problem Details |
| asyncio.wait_for | stdlib | Per-endpoint timeouts | Python native, clean cancellation semantics |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| logfire[httpx] | (via logfire) | HTTP client instrumentation | Trace outbound cloud API calls |
| logfire[sqlalchemy] | (via logfire) | Database query tracing | Trace SQLModel/SQLAlchemy queries |
| logfire.instrument_openai | (via logfire) | OpenAI API tracing | Track cloud fallback requests with tokens |
| logfire.instrument_anthropic | (via logfire) | Anthropic API tracing | Track cloud fallback requests with tokens |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| fastapi-rfc7807 | Custom exception handlers | More control but more code to maintain |
| SQLite audit log | LogFire only | LogFire has 30-day retention limit for free tier; local gives full control |
| WebSocket for live updates | SSE polling | WebSocket has better browser support limits (6 connections per domain for SSE/HTTP1.1) |

**Installation:**
```bash
pip install 'logfire[fastapi,httpx,sqlalchemy]' fastapi-rfc7807
```

Note: `logfire[fastapi]` already in pyproject.toml - add `httpx` and `sqlalchemy` extras.

## Architecture Patterns

### Recommended Project Structure
```
backend/mlx_manager/
├── observability/           # NEW: Observability configuration
│   ├── __init__.py
│   └── logfire_config.py   # LogFire setup, instrumentation calls
├── errors/                  # NEW: Error handling
│   ├── __init__.py
│   ├── problem_details.py  # RFC 7807 response models
│   └── handlers.py         # Exception handlers
├── middleware/              # NEW: Middleware components
│   ├── __init__.py
│   ├── timeout.py          # Timeout middleware/decorators
│   └── audit.py            # Audit logging middleware
├── models.py               # Add AuditLog model
└── main.py                 # Configure LogFire, error handlers, middleware
```

### Pattern 1: LogFire Configuration (Centralized)
**What:** Single configuration point for all LogFire instrumentation
**When to use:** At application startup, before creating FastAPI app
**Example:**
```python
# Source: https://logfire.pydantic.dev/docs/reference/configuration/
import logfire
from mlx_manager.config import settings

def configure_logfire() -> None:
    """Configure LogFire with all instrumentations."""
    logfire.configure(
        service_name="mlx-manager",
        service_version=settings.version,
        environment=settings.environment,  # "development" | "production"
        send_to_logfire="if-token-present",  # Only send if LOGFIRE_TOKEN set
    )

    # Instrument integrations AFTER configure()
    # Note: Each integration called exactly once
    logfire.instrument_httpx()  # All httpx clients
    logfire.instrument_sqlalchemy(engine=engine)  # Database queries
```

### Pattern 2: RFC 7807 Problem Details Response
**What:** Structured error response format per RFC 7807/9457
**When to use:** All API error responses
**Example:**
```python
# Source: https://datatracker.ietf.org/doc/html/rfc7807
from pydantic import BaseModel

class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details response."""
    type: str = "about:blank"  # URI reference to problem type
    title: str                 # Human-readable summary (not occurrence-specific)
    status: int                # HTTP status code
    detail: str | None = None  # Occurrence-specific explanation
    instance: str | None = None  # URI for specific occurrence
    request_id: str            # Always include for log correlation (CONTEXT.md decision)

# Example error response:
{
    "type": "https://mlx-manager.dev/errors/model-not-found",
    "title": "Model Not Found",
    "status": 404,
    "detail": "Model 'mlx-community/Llama-3.2-3B' is not loaded",
    "request_id": "req_abc123xyz"
}
```

### Pattern 3: Per-Endpoint Timeout Decorator
**What:** Configurable timeouts per endpoint type using asyncio.wait_for
**When to use:** Long-running inference endpoints (chat, completions, embeddings)
**Example:**
```python
# Source: https://sentry.io/answers/make-long-running-tasks-time-out-in-fastapi/
import asyncio
from functools import wraps
from typing import TypeVar, ParamSpec, Callable, Awaitable

P = ParamSpec('P')
T = TypeVar('T')

# Timeout settings from CONTEXT.md decisions
TIMEOUT_CHAT = 15 * 60      # 15 minutes
TIMEOUT_COMPLETIONS = 10 * 60  # 10 minutes
TIMEOUT_EMBEDDINGS = 2 * 60   # 2 minutes

def with_timeout(seconds: float):
    """Decorator to add timeout to async endpoint."""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutHTTPException(
                    timeout_seconds=seconds,
                    detail=f"Request timed out after {seconds} seconds"
                )
        return wrapper
    return decorator

# Usage:
@router.post("/v1/chat/completions")
@with_timeout(TIMEOUT_CHAT)
async def create_chat_completion(request: ChatCompletionRequest):
    ...
```

### Pattern 4: Audit Log with Background Write
**What:** Non-blocking audit log writes using FastAPI BackgroundTasks
**When to use:** After request completes, before returning response
**Example:**
```python
# Source: https://sgdatasolutions.dk/blog_posts/api-audit-logging/
from datetime import datetime, UTC
from sqlmodel import SQLModel, Field

class AuditLog(SQLModel, table=True):
    """Request audit log entry - metadata only, no prompt content."""
    __tablename__ = "audit_logs"

    id: int | None = Field(default=None, primary_key=True)
    request_id: str = Field(index=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str
    backend_type: str  # "local" | "openai" | "anthropic"
    endpoint: str      # "/v1/chat/completions", etc.
    duration_ms: int
    status: str        # "success" | "error" | "timeout"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error_type: str | None = None  # HTTPException class name if error

async def write_audit_log(log_entry: AuditLog) -> None:
    """Background task to write audit log."""
    async with get_session() as session:
        session.add(log_entry)
        await session.commit()
```

### Pattern 5: WebSocket Live Updates for Admin Panel
**What:** Real-time audit log streaming via WebSocket
**When to use:** Admin panel for live request monitoring
**Example:**
```python
# Source: MDN WebSocket API
from fastapi import WebSocket
from collections import deque

# In-memory buffer for recent logs (WebSocket broadcast)
recent_logs: deque[dict] = deque(maxlen=100)
connected_clients: set[WebSocket] = set()

@router.websocket("/ws/audit-logs")
async def audit_log_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        # Send recent logs on connect
        for log in recent_logs:
            await websocket.send_json(log)
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)

async def broadcast_audit_log(log: dict) -> None:
    """Broadcast new log entry to all connected clients."""
    recent_logs.append(log)
    for client in list(connected_clients):
        try:
            await client.send_json(log)
        except:
            connected_clients.discard(client)
```

### Anti-Patterns to Avoid
- **Synchronous audit writes:** Never block request response to write audit logs - use BackgroundTasks
- **Storing prompt content:** NEVER log request/response content - privacy requirement from CONTEXT.md
- **Global timeout middleware:** Per-endpoint timeouts are required; global middleware cannot differentiate
- **Exposing internal errors:** Never include stack traces or internal details in error responses

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Request tracing | Custom request ID generation | LogFire span IDs | Automatic correlation, distributed tracing support |
| Error response format | Custom JSON structure | fastapi-rfc7807 | Standard format, automatic exception translation |
| HTTP client tracing | Manual timing code | logfire.instrument_httpx() | Automatic span creation, header correlation |
| Database query logging | Manual logging | logfire.instrument_sqlalchemy() | Query timing, parameter capture, async support |
| LLM token counting | Manual parsing | logfire.instrument_openai/anthropic() | Automatic extraction from API responses |

**Key insight:** LogFire provides most observability features out of the box. The primary custom work is audit logging (for local persistence beyond LogFire's retention) and timeout handling (no standard solution exists).

## Common Pitfalls

### Pitfall 1: LogFire Not Configured Before Instrumentation
**What goes wrong:** Instrumentation calls before `logfire.configure()` emit warnings and don't log
**Why it happens:** Initialization order matters; configure must come first
**How to avoid:** Call `logfire.configure()` at top of main.py, before any imports that might use instrumented libraries
**Warning signs:** LogFire warnings in logs about "not configured"

### Pitfall 2: Timeout Not Cancelling Task Properly
**What goes wrong:** asyncio.wait_for raises TimeoutError but background task continues
**Why it happens:** The task is cancelled but may have non-cancellable operations
**How to avoid:** Ensure all long-running operations are cancellation-aware; use try/except asyncio.CancelledError
**Warning signs:** Resource leaks, GPU memory not freed after timeout

### Pitfall 3: SSE Connection Limits (HTTP/1.1)
**What goes wrong:** Browser can only open 6 SSE connections per domain
**Why it happens:** HTTP/1.1 per-domain connection limit affects EventSource
**How to avoid:** Use WebSocket for admin panel live updates (not subject to same limit); SSE is fine for single inference streams
**Warning signs:** "net::ERR_INSUFFICIENT_RESOURCES" in browser, connections hanging

### Pitfall 4: Audit Log Query Performance
**What goes wrong:** Slow queries on large audit_logs table
**Why it happens:** No indexes on frequently filtered columns
**How to avoid:** Add indexes on timestamp, model, backend_type, status; implement 30-day retention cleanup
**Warning signs:** Slow admin panel load times, database file growing indefinitely

### Pitfall 5: Streaming Errors Not Reaching Client
**What goes wrong:** Error occurs mid-stream but client doesn't see it
**Why it happens:** SSE events already sent; can't change response status after starting
**How to avoid:** Send error as final SSE event with "error" event type, then close connection
**Warning signs:** Client hangs waiting for more data after server error

## Code Examples

Verified patterns from official sources:

### LogFire FastAPI Setup
```python
# Source: https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/
import logfire
from fastapi import FastAPI

# Configure FIRST
logfire.configure(
    service_name="mlx-manager",
    send_to_logfire="if-token-present",
)

app = FastAPI()

# Instrument AFTER configure
logfire.instrument_fastapi(app)
# Optional: Extra spans for argument parsing and endpoint execution
# logfire.instrument_fastapi(app, extra_spans=True)
```

### LogFire HTTPX Instrumentation
```python
# Source: https://logfire.pydantic.dev/docs/integrations/http-clients/httpx/
import logfire
import httpx

logfire.configure()
logfire.instrument_httpx()  # Instruments ALL httpx clients globally

# Or instrument specific client:
# client = httpx.AsyncClient()
# logfire.instrument_httpx(client)
```

### LogFire OpenAI/Anthropic Instrumentation
```python
# Source: https://logfire.pydantic.dev/docs/integrations/llms/openai/
import logfire

logfire.configure()
logfire.instrument_openai()  # Global instrumentation
logfire.instrument_anthropic()  # Global instrumentation

# Captures: request duration, token usage, exceptions, streaming spans
```

### LogFire SQLAlchemy with SQLModel
```python
# Source: https://logfire.pydantic.dev/docs/integrations/databases/sqlalchemy/
from sqlalchemy.ext.asyncio import create_async_engine
import logfire

logfire.configure()
engine = create_async_engine("sqlite+aiosqlite:///./data.db")
logfire.instrument_sqlalchemy(engine=engine)

# Note: Works with SQLModel since it's built on SQLAlchemy
```

### RFC 7807 Error Handler
```python
# Source: https://github.com/vapor-ware/fastapi-rfc7807
from fastapi import FastAPI
from fastapi_rfc7807 import middleware

app = FastAPI()
middleware.register(app)

# All exceptions now return RFC 7807 format:
# {
#   "exc_type": "ValueError",
#   "type": "about:blank",
#   "title": "Unexpected Server Error",
#   "status": 500,
#   "detail": "something went wrong"
# }
```

### SSE Error Event Pattern
```python
# Source: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
import json
from sse_starlette.sse import EventSourceResponse

async def generate_with_error_handling():
    try:
        async for chunk in generate_stream():
            yield {"data": json.dumps(chunk)}
        yield {"data": "[DONE]"}
    except Exception as e:
        # Send error as special event type
        error_event = {
            "type": "error",
            "title": "Generation Failed",
            "detail": str(e),
            "request_id": request_id
        }
        yield {"event": "error", "data": json.dumps(error_event)}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| RFC 7807 | RFC 9457 | 2024 | RFC 9457 obsoletes 7807 but maintains compatibility; same structure |
| Prometheus + custom spans | Pydantic LogFire | 2024 | Unified observability with LLM-native metrics |
| Manual request logging | LogFire automatic instrumentation | 2024 | Less code, automatic correlation |
| Global timeouts | Per-endpoint timeouts | Best practice | Different operations need different limits |

**Deprecated/outdated:**
- RFC 7807: Superseded by RFC 9457 but fully compatible; use "RFC 9457 Problem Details"
- `asyncio.exceptions.TimeoutError`: Python 3.11+ uses `TimeoutError` directly

## Open Questions

Things that couldn't be fully resolved:

1. **LogFire Free Tier Retention**
   - What we know: LogFire has data retention limits (30 days typical for free tier)
   - What's unclear: Exact retention for paid tiers
   - Recommendation: Keep local audit log as source of truth with 30-day retention

2. **aiosqlite Direct Instrumentation**
   - What we know: LogFire has `instrument_sqlite3()` for sync and `instrument_sqlalchemy()` for async
   - What's unclear: Whether aiosqlite is directly instrumentable or relies on SQLAlchemy
   - Recommendation: Use `instrument_sqlalchemy(engine=async_engine)` for SQLModel

3. **Streaming Timeout Behavior**
   - What we know: `asyncio.wait_for` cancels tasks after timeout
   - What's unclear: How to handle partial streaming response on timeout
   - Recommendation: Per CONTEXT.md, discard partial responses on timeout (timeout = error)

## Sources

### Primary (HIGH confidence)
- [Pydantic LogFire FastAPI Integration](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/) - FastAPI setup, instrumentation options
- [Pydantic LogFire HTTPX Integration](https://logfire.pydantic.dev/docs/integrations/http-clients/httpx/) - HTTP client tracing
- [Pydantic LogFire SQLAlchemy Integration](https://logfire.pydantic.dev/docs/integrations/databases/sqlalchemy/) - Database query tracing
- [Pydantic LogFire OpenAI Integration](https://logfire.pydantic.dev/docs/integrations/llms/openai/) - LLM token tracking
- [Pydantic LogFire Configuration](https://logfire.pydantic.dev/docs/reference/configuration/) - Environment variables, setup
- [RFC 7807/9457 IETF Specification](https://datatracker.ietf.org/doc/html/rfc7807.html) - Problem Details format
- [fastapi-rfc7807 GitHub](https://github.com/vapor-ware/fastapi-rfc7807) - RFC 7807 middleware
- [Sentry: FastAPI Timeouts](https://sentry.io/answers/make-long-running-tasks-time-out-in-fastapi/) - asyncio.wait_for pattern
- [MDN: Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) - SSE error handling

### Secondary (MEDIUM confidence)
- [FastAPI Audit Logging Best Practices](https://sgdatasolutions.dk/blog_posts/api-audit-logging/) - Background task pattern
- [FastAPI Middleware Patterns 2026](https://johal.in/fastapi-middleware-patterns-custom-logging-metrics-and-error-handling-2026-2/) - Middleware design
- [fastapi-audit-log PyPI](https://pypi.org/project/fastapi-audit-log/) - Existing audit package (reference, not recommendation)

### Tertiary (LOW confidence)
- WebSearch results for timeout patterns - Community implementations vary
- Performance overhead claims (~5%) - Not independently verified

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified via official documentation
- Architecture: HIGH - Patterns follow official examples and community standards
- Pitfalls: MEDIUM - Based on official docs and common issues; some specific to this project
- Timeout handling: HIGH - Python stdlib, well-documented behavior

**Research date:** 2026-01-30
**Valid until:** 2026-02-28 (30 days - LogFire evolving but stable core)
