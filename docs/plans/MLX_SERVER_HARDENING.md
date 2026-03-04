# MLX Server Hardening & Performance Plan

**Version:** 1.0
**Date:** 2026-03-03
**Status:** ALL TIERS COMPLETE (P0+P1+P2+P3)
**Scope:** `backend/mlx_manager/mlx_server/` (~77 files, ~280KB)

---

## Executive Summary

A pre-release review of the `mlx_server` component identified 21 improvements across security,
performance, architecture, and code quality. The issues range from OOM-inducing input validation
gaps and a path traversal vulnerability in static file serving, to hot-path performance problems
like O(n²) string concatenation and 100ms polling latency per streaming token.

Items are organized into 4 priority tiers:

- **P0 — Critical (3 items):** Security flaws and correctness bugs that must be fixed before
  any public or multi-tenant deployment.
- **P1 — High (5 items):** Performance hardening and administrative security that significantly
  improve production reliability.
- **P2 — Medium (7 items):** Operational improvements (rate limiting, metrics, graceful shutdown)
  that are important but not blocking.
- **P3 — Low (6 items):** Quality-of-life and developer experience improvements.

All new behaviors are opt-in via config or have zero-impact defaults to preserve backward
compatibility.

---

## Table of Contents

1. [P0 — Critical](#p0--critical-3-items)
   - [P0-1: Unbounded request body sizes](#p0-1-unbounded-request-body-sizes)
   - [P0-2: Static file path traversal](#p0-2-static-file-path-traversal)
   - [P0-3: Blocking tokenizer encode](#p0-3-blocking-tokenizer-encode)
2. [P1 — High](#p1--high-5-items)
   - [P1-1: Streaming poll interval](#p1-1-streaming-poll-interval)
   - [P1-2: Admin endpoint auth](#p1-2-admin-endpoint-auth)
   - [P1-3: O(n²) string concatenation](#p1-3-on-string-concatenation)
   - [P1-4: KV cache memory tracking](#p1-4-kv-cache-memory-tracking)
   - [P1-5: DRY router patterns](#p1-5-dry-router-patterns)
3. [P2 — Medium](#p2--medium-7-items)
4. [P3 — Low](#p3--low-6-items)
5. [Implementation Order](#implementation-order)
6. [Key Files Reference](#key-files-reference)

---

## P0 — Critical (3 items)

### P0-1: Unbounded Request Body Sizes

**Priority:** CRITICAL | **Effort:** Small | **Risk:** Low
**Files:** `mlx_server/schemas/openai.py`, `mlx_server/schemas/anthropic.py`

#### Problem

All list fields in request schemas (`messages`, `tools`, `stop`, `content`, `input`) accept
unbounded arrays. A malicious or misconfigured client can send millions of entries to trigger
out-of-memory conditions on the server host. There is no defense at the Pydantic validation
layer — the fields are plain `list[...]` with no upper bound.

#### Fix

Add `Field(max_length=N)` constraints to all request-facing list fields. Proposed limits:

| Field | Schema | Max |
|-------|--------|-----|
| `messages` | Both | 1024 |
| `tools` | Both | 256 |
| `stop` / `stop_sequences` | Both | 16 |
| `tool_calls` | OpenAI | 128 |
| `content` blocks | Anthropic | 256 |
| `prompt` (list variant) | OpenAI | 32 |
| `input` (embeddings list) | OpenAI | 2048 |
| `system` (list variant) | Anthropic | 64 |

Additionally:

- Add `le=128000` upper bound on Anthropic `max_tokens` (matches Claude API limit).
- Add a `model_validator` to `ChatCompletionRequest` that rejects `json_schema` objects
  exceeding 50KB when serialized. Deeply nested schemas can cause quadratic parse times even
  within the Pydantic validation step.

#### Example

```python
# Before
messages: list[ChatMessage]

# After
from pydantic import Field
messages: list[ChatMessage] = Field(max_length=1024)
```

#### Validation

- Unit test: requests with 1025 messages return 422 Unprocessable Entity.
- Unit test: requests with a `json_schema` > 50KB return 422.
- All existing schema tests pass unchanged.

---

### P0-2: Static File Path Traversal

**Priority:** CRITICAL | **Effort:** Trivial | **Risk:** Very low
**File:** `main.py` (`serve_spa()`)

#### Problem

The SPA catch-all route constructs a filesystem path directly from the URL:

```python
file_path = STATIC_DIR / full_path
```

There is no verification that the resolved path stays within `STATIC_DIR`. A request for
`GET /../../etc/passwd` resolves to a path outside the static directory. On some deployment
configurations this could read arbitrary files accessible to the process.

#### Fix

Add a path containment check before any filesystem access:

```python
async def serve_spa(full_path: str):
    file_path = STATIC_DIR / full_path
    resolved = file_path.resolve()
    if not resolved.is_relative_to(STATIC_DIR.resolve()):
        raise HTTPException(status_code=404)
    if resolved.is_file():
        return FileResponse(resolved)
    return FileResponse(STATIC_DIR / "index.html")
```

`Path.is_relative_to()` is available in Python 3.9+. If the project targets an earlier
version, use `str(resolved).startswith(str(STATIC_DIR.resolve()))` as the fallback check.

#### Validation

- Unit test: `GET /../../etc/passwd` returns 404.
- Unit test: `GET /%2e%2e%2fetc%2fpasswd` (URL-encoded) returns 404 (Starlette decodes before
  routing).
- Existing static file serving tests pass unchanged.

---

### P0-3: Blocking Tokenizer Encode

**Priority:** CRITICAL | **Effort:** Small | **Risk:** Low
**File:** `mlx_server/services/inference.py`

#### Problem

Five `tokenizer.encode()` calls run synchronously on the async event loop inside
`_complete_chat_ir()` and `_generate_raw_completion()`. For large inputs (long system prompts,
many-message threads, large image captions), this blocks the event loop and prevents other
in-flight requests from making progress.

Additionally, calls 2, 3, and 5 use the raw tokenizer reference rather than `actual_tokenizer`,
which is the unwrapped form. This is inconsistent with calls 1 and 4 and may silently produce
incorrect token counts for models where `tokenizer` is a wrapper object.

#### Fix

Wrap all five calls in `run_in_executor` to move them off the event loop:

```python
# Before
token_count = len(tokenizer.encode(text))

# After
loop = asyncio.get_event_loop()
token_count = len(
    await loop.run_in_executor(None, actual_tokenizer.encode, text)
)
```

Apply `actual_tokenizer` consistently across all five call sites. Consider extracting a local
helper to avoid repeating the executor boilerplate:

```python
async def _encode(tokenizer, text: str) -> list[int]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, tokenizer.encode, text)
```

#### Validation

- Existing inference unit tests pass unchanged.
- Manual test: send a request with a 50K-token system prompt; verify event loop latency (e.g.,
  via a concurrent health-check request) does not spike.
- Verify `actual_tokenizer` is available at all five call sites (it should be — add an
  assertion if there is any doubt).

---

## P1 — High (5 items)

### P1-1: Streaming Poll Interval

**Priority:** HIGH | **Effort:** Medium | **Risk:** Medium
**File:** `mlx_server/utils/metal.py`

#### Problem

`stream_from_metal_thread()` polls for new tokens using `queue.get(timeout=0.1)`. This adds
up to 100ms of latency per token when the queue drains between tokens. For models generating
5–10 tokens/second, this poll interval can double or triple perceived streaming latency.

#### Fix

Replace busy-polling with `asyncio.Event`-based signaling:

```python
# Producer (Metal thread): after each queue.put(), signal the consumer
event.set()

# Consumer (async loop): await the event before draining the queue
await asyncio.wait_for(event.wait(), timeout=0.5)
event.clear()
while not queue.empty():
    yield queue.get_nowait()
```

Because `asyncio.Event` is not thread-safe, the producer must call `loop.call_soon_threadsafe(event.set)`
rather than `event.set()` directly.

This delivers near-zero latency token forwarding — the consumer wakes as soon as a token is
available rather than after a 100ms sleep. The 0.5s timeout fallback handles edge cases where
the signal is missed.

#### Design Note

This is a non-trivial refactor because `metal.py` currently owns the queue/thread lifecycle
and `stream_from_metal_thread()` is called from multiple routers. Validate carefully that
the event object lifetime is correctly scoped per-request.

#### Validation

- Streaming latency benchmark: time-to-first-token and inter-token latency should both
  decrease measurably.
- All existing streaming unit tests pass.
- No test should need to account for the 100ms poll floor (remove any `asyncio.sleep(0.1)`
  workarounds in tests).

---

### P1-2: Admin Endpoint Auth

**Priority:** HIGH | **Effort:** Small | **Risk:** Low
**Files:** `mlx_server/config.py`, `mlx_server/api/v1/admin.py`

#### Problem

All `/v1/admin/*` endpoints (model load/unload, audit log export, health details) are
unauthenticated. In any deployment where the MLX server is exposed beyond localhost — including
behind a reverse proxy without its own auth — any client can unload models, read audit logs,
or trigger expensive model loads.

#### Fix

Add an `MLX_SERVER_ADMIN_TOKEN` setting to `mlx_server/config.py`:

```python
class MLXServerSettings(BaseSettings):
    admin_token: str | None = Field(
        default=None,
        description=(
            "Bearer token required for /v1/admin/* endpoints. "
            "When None or empty, endpoints are open (backward compat)."
        ),
    )
```

Create a `verify_admin_token` FastAPI dependency:

```python
# mlx_server/api/dependencies.py
def verify_admin_token(
    authorization: str | None = Header(default=None),
    settings: MLXServerSettings = Depends(get_mlx_settings),
) -> None:
    if not settings.admin_token:
        return  # Token not configured — open access (backward compat)
    if authorization != f"Bearer {settings.admin_token}":
        raise HTTPException(status_code=401, detail="Invalid admin token")
```

Apply as a router-level dependency on the admin router:

```python
router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(verify_admin_token)],
)
```

#### Backward Compatibility

When `MLX_SERVER_ADMIN_TOKEN` is not set, the dependency is a no-op. Existing deployments
are unaffected.

#### Validation

- Unit test: admin endpoints return 401 when token is configured and request has no/wrong token.
- Unit test: admin endpoints return 200 when token is configured and request has correct token.
- Unit test: admin endpoints return 200 when no token is configured (backward compat).

---

### P1-3: O(n²) String Concatenation

**Priority:** HIGH | **Effort:** Small | **Risk:** Low
**File:** `mlx_server/services/response_processor.py`

#### Problem

Two hot-path attributes use `+=` string concatenation in the streaming loop:

```python
self._accumulated += token
self._yielded_content += to_yield
```

Python strings are immutable. Each `+=` allocates a new string of length n+k, copies all n
existing characters, then appends k new characters. Over a 10K-token response, this creates
O(n²) total work and O(n) peak allocation pressure, making long responses increasingly slow
to stream.

#### Fix

Replace both attributes with `list[str]` buffers. Add `@property` accessors that join lazily
only when the accumulated value is needed outside the tight token loop:

```python
# Initialization
self._accumulated_parts: list[str] = []
self._yielded_parts: list[str] = []

# In the hot path
self._accumulated_parts.append(token)
self._yielded_parts.append(to_yield)

# Accessor (called only at finalize time / pattern matching boundaries)
@property
def _accumulated(self) -> str:
    return "".join(self._accumulated_parts)

@property
def _yielded_content(self) -> str:
    return "".join(self._yielded_parts)
```

The key constraint is to ensure `_accumulated` and `_yielded_content` are not read on every
token iteration — they should only be joined when pattern matching or finalization requires
the full string. Audit all read sites before changing.

#### Validation

- All existing `response_processor` tests pass unchanged.
- Benchmark: measure streaming latency for a 5K-token and 20K-token response before and after.
  The 20K case should show the largest improvement.

---

### P1-4: KV Cache Memory Tracking

**Priority:** HIGH | **Effort:** Small | **Risk:** Low
**File:** `mlx_server/models/pool.py`

#### Problem

`_estimate_model_size()` applies a 5% overhead factor on top of model weight memory:

```python
return model_weights_gb * 1.05
```

KV cache allocations during inference can consume 20–40% of model weight memory, depending
on context length and batch size. With only a 5% buffer, the pool regularly over-commits
system memory, which triggers macOS memory pressure events and degrades performance for all
active models.

#### Fix

Increase the overhead factor from `1.05` to `1.25` as an immediate conservative correction:

```python
KV_CACHE_OVERHEAD_FACTOR = 1.25  # Accounts for KV cache (20-40% of weights)
return model_weights_gb * KV_CACHE_OVERHEAD_FACTOR
```

Additionally, when `mx.metal.get_active_memory()` is available (mlx >= 0.16), supplement the
static estimate with live memory tracking to make eviction decisions more accurate:

```python
try:
    live_bytes = mx.metal.get_active_memory()
    live_gb = live_bytes / (1024 ** 3)
    # Use the larger of static estimate and live measurement
    return max(static_estimate_gb, live_gb)
except AttributeError:
    return static_estimate_gb  # Fallback for older mlx versions
```

Add `DEBUG`-level logging of estimated vs. actual memory at load time to help tune the
factor over time:

```python
logger.debug(
    "Model %s: estimated %.2f GB, live %.2f GB",
    model_id, static_estimate_gb, live_gb,
)
```

#### Validation

- Load a model and verify it is not immediately evicted due to inaccurate size estimation.
- Verify the pool does not OOM when serving long-context requests with multiple models loaded.

---

### P1-5: DRY Router Patterns

**Priority:** HIGH | **Effort:** Medium | **Risk:** Low
**Files:** `mlx_server/api/v1/chat.py`, `mlx_server/api/v1/completions.py`,
`mlx_server/api/v1/messages.py`, `mlx_server/api/v1/embeddings.py`,
`mlx_server/utils/request_helpers.py` (new file)

#### Problem

Four patterns are duplicated across all inference routers:

1. **Timeout error events** — each router constructs an SSE error dict inline when a timeout
   fires.
2. **Streaming loops** — the `async for event in stream: yield format(event)` scaffold is
   repeated with minor variations.
3. **Audit tracking** — `audit_service.record(...)` is called with near-identical arguments
   in each router.
4. **Exception handlers** — `except InferenceError`, `except asyncio.TimeoutError`, and
   `except Exception` blocks are copy-pasted with the same logging and response patterns.

This creates a maintenance burden: a fix to one router's error handling rarely gets applied
to all four.

#### Fix

Extract shared helpers into `mlx_server/utils/request_helpers.py`:

```python
def timeout_error_event(timeout: float, request_id: str) -> dict:
    """Construct an SSE error event for timeout responses."""
    return {
        "error": {
            "type": "timeout",
            "message": f"Request timed out after {timeout}s",
            "code": "request_timeout",
        },
        "id": request_id,
    }

async def with_inference_timeout(
    coro: Awaitable[T],
    timeout: float,
    description: str,
) -> T:
    """Run coro with a timeout, raising TimeoutHTTPException on expiry."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutHTTPException(
            detail=f"{description} timed out after {timeout}s"
        )
```

The shared streaming event loop generator and unified exception handler pattern should be
designed to accept format callbacks, keeping router-specific serialization logic in the router
while the control flow lives in `request_helpers.py`.

#### Design Constraint

The streaming generator cannot be fully extracted because SSE formatting (OpenAI vs. Anthropic
wire format) differs per router. Extract only the control flow skeleton; inject format
callbacks as arguments.

#### Validation

- All existing router tests pass unchanged.
- No behavior change — pure structural refactor.
- Run the full test suite after extraction to catch any import or callback binding issues.

---

## P2 — Medium (7 items)

### P2-1: Rate Limiting

Add configurable per-IP and global rate limiting to protect against runaway clients.

- **Config:** `MLX_SERVER_RATE_LIMIT_RPM` (requests per minute per IP), default `0` (disabled).
- **Implementation:** `slowapi` middleware (wraps `limits` library) or a custom `asyncio`
  token-bucket middleware if `slowapi` introduces unacceptable overhead.
- **Headers:** Return `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After` on 429
  responses per standard practice.

---

### P2-2: Request ID Propagation

Add `X-Request-ID` header support to enable distributed tracing and log correlation.

- Generate a UUID v4 if the header is absent.
- Thread the request ID through: structured log fields, audit log records, and all error
  response bodies.
- Return `X-Request-ID` in every response so clients can correlate logs.

---

### P2-3: Structured Error Responses

Standardize all error responses to a consistent schema:

```json
{
  "error": {
    "type": "validation_error",
    "message": "messages field exceeds maximum length of 1024",
    "code": "field_max_length_exceeded"
  }
}
```

- Define an `ErrorCode` enum for programmatic handling by clients.
- Add a global exception handler in `main.py` that maps `RequestValidationError`,
  `InferenceError`, and unhandled exceptions to the standard format.
- Update OpenAPI spec with error response schemas on all endpoints.

---

### P2-4: Graceful Shutdown

Add connection draining on SIGTERM to avoid dropping in-flight requests during restarts.

- **Mechanism:** FastAPI lifespan handler catches SIGTERM, sets a `shutting_down` flag,
  and waits up to a configurable `MLX_SERVER_DRAIN_TIMEOUT` (default: 30s) for active
  requests to complete.
- **Model unload:** After drain, unload all models cleanly to release Metal memory before
  the process exits.
- **Health endpoint:** Return `503 Service Unavailable` for new requests once the drain flag
  is set, so load balancers stop routing to the instance.

---

### P2-5: Model Loading Progress

Add an SSE endpoint that clients can subscribe to for model loading progress:

- **Endpoint:** `GET /v1/admin/models/{model_id}/loading-progress`
- **Events:** `download_progress` (percent), `weights_loading` (percent), `adapter_init`,
  `ready`, `error`.
- Enables frontend progress bars and eliminates the need for clients to poll the model
  status endpoint during load.

---

### P2-6: Prometheus Metrics

Add an optional metrics endpoint for production observability.

- **Endpoint:** `GET /v1/admin/metrics` (plain text Prometheus format)
- **Gating:** Only enabled when `MLX_SERVER_METRICS_ENABLED=true`.
- **Metrics to track:**
  - `mlx_request_latency_seconds` (histogram by endpoint, model)
  - `mlx_token_throughput_total` (counter by model)
  - `mlx_model_load_duration_seconds` (histogram by model)
  - `mlx_model_memory_bytes` (gauge by model)
  - `mlx_pool_cache_hits_total` / `mlx_pool_cache_misses_total`
  - `mlx_active_requests` (gauge)

---

### P2-7: Input Validation Hardening

Strengthen validation before the inference path is entered:

- **Model field:** Validate the `model` field against the set of loaded or available models
  early in request handling (before allocating inference resources). Return 404 with a clear
  message rather than a cryptic internal error.
- **Multimodal content-type:** Validate `image_url.url` MIME type or base64 prefix for
  vision requests before attempting to decode.
- **Base64 image size:** Reject base64-encoded images exceeding a configurable size limit
  (default: 20MB decoded) before decoding to prevent memory spikes.

---

## P3 — Low (6 items)

### P3-1: OpenAPI Spec Enrichment

- Add response examples, detailed error response schemas, and parameter descriptions to all
  `mlx_server` endpoints.
- Verify the enriched spec can be used to generate typed client SDKs (e.g., via `openapi-generator`).

---

### P3-2: Audit Log Rotation

- Add rotation by database size: when the audit log exceeds `MLX_SERVER_AUDIT_MAX_MB`
  (default: 100MB), purge the oldest records to stay under the limit.
- Run `VACUUM` after purge to reclaim disk space.
- Configurable retention: `MLX_SERVER_AUDIT_RETENTION_DAYS` (default: 30).

---

### P3-3: Config Hot-Reload

- Watch the config file for changes using `watchfiles` or `inotify` and reload without
  requiring a restart.
- Support signal-based reload via SIGHUP for ops tooling compatibility.
- Exclude settings that cannot be changed at runtime (e.g., `database_path`, `port`) — log
  a warning if those are modified.

---

### P3-4: Connection Pooling for Embedded Mode

- When MLX Manager and MLX Server run in the same process (embedded mode), they currently
  open separate `aiosqlite` connection pools.
- Share a single pool to reduce connection overhead and eliminate any risk of lock contention
  on the SQLite file.
- Implement via a module-level singleton pool injected into both subsystems at startup.

---

### P3-5: Model Preload Warming

- Add optional startup model preloading: `MLX_SERVER_PRELOAD_MODEL` config value.
- After the server starts, load the specified model and run a configurable warmup prompt
  (`MLX_SERVER_WARMUP_PROMPT`, default: `"Hello"`).
- Reduces cold-start latency for the first real request, which is important for interactive
  use cases where the first user sees a multi-second delay.

---

### P3-6: Test Coverage Gaps

Current test coverage for `mlx_server` leaves several critical paths untested:

| Gap | Suggested Test |
|-----|----------------|
| Streaming timeout path | Mock a slow inference; verify SSE error event is sent and connection closes cleanly |
| Admin auth (`verify_admin_token`) | Test all three cases: no token configured, correct token, wrong token |
| Path traversal prevention | Request `GET /../../etc/passwd`; assert 404 |
| Request body size limits (P0-1) | Parameterized test for each bounded field at N and N+1 |
| `run_in_executor` tokenizer encode (P0-3) | Mock event loop to verify encode runs off-thread |

Target: 80% coverage for the `mlx_server` module overall.

---

## Implementation Order

```
P0-2 (Path Traversal)          ── Trivial, no deps, fix immediately
    │
P0-1 (Request Body Limits)     ── Small, no deps, fix immediately
    │
P0-3 (Blocking Tokenizer)      ── Small, no deps, fix immediately
    │
P1-3 (O(n²) Concatenation)    ── Small, no deps, high ROI
    │
P1-4 (KV Cache Overhead)       ── Small, no deps, immediate stability fix
    │
P1-2 (Admin Auth)              ── Small, requires config.py change first
    │
P1-5 (DRY Routers)             ── Medium, must be done before adding more routers
    │
P1-1 (Streaming Poll)          ── Medium, higher risk; do after other P1s are stable
    │
P2-7 (Input Validation)        ── Builds on P0-1 field limits
P2-2 (Request ID)              ── Needed before P2-6 metrics are useful
P2-3 (Structured Errors)       ── Depends on P2-2 for request ID in errors
P2-1 (Rate Limiting)           ── After P2-2 (per-IP needs request context)
P2-4 (Graceful Shutdown)        ── Independent, can be done any time
P2-5 (Loading Progress)        ── Independent, can be done any time
P2-6 (Prometheus Metrics)      ── After P2-2 (request ID correlation)
    │
P3-* (Low items)               ── In any order, after P0/P1/P2 are stable
```

P0 items should land in a single commit and be treated as a hotfix. P1 items get their own
commit each. P2 and P3 items may be batched by theme.

---

## Risk Assessment

| Item | Risk | Mitigation |
|------|------|------------|
| P0-1 (body limits) | Very low — additive validation | Verify no legitimate client sends > 1024 messages |
| P0-2 (path traversal) | Very low — one-line guard | Test URL-encoded variants |
| P0-3 (executor tokenize) | Low — same behavior, off-thread | Verify `actual_tokenizer` consistency across all 5 call sites |
| P1-1 (stream poll) | Medium — async/thread boundary | Careful event lifetime scoping; benchmark before/after |
| P1-2 (admin auth) | Very low — opt-in; backward compat by default | Ensure empty string and None both mean "open" |
| P1-3 (string buffers) | Low — mechanical substitution | Audit all `_accumulated` / `_yielded_content` read sites |
| P1-4 (KV overhead) | Low — increase only | Monitor that pool size estimates don't become too conservative |
| P1-5 (DRY routers) | Low — structural only | Run full test suite after extraction |
| P2-* | Low — all opt-in | Feature flags / config gates |
| P3-* | Very low — additive | No existing behavior changed |

---

## Key Files Reference

| File | Items |
|------|-------|
| `schemas/openai.py` | P0-1 |
| `schemas/anthropic.py` | P0-1 |
| `main.py` | P0-2 |
| `services/inference.py` | P0-3 |
| `utils/metal.py` | P1-1 |
| `config.py` | P1-2 |
| `api/v1/admin.py` | P1-2 |
| `services/response_processor.py` | P1-3 |
| `models/pool.py` | P1-4 |
| `api/v1/chat.py` | P1-5 |
| `api/v1/completions.py` | P1-5 |
| `api/v1/messages.py` | P1-5 |
| `api/v1/embeddings.py` | P1-5 |
| `utils/request_helpers.py` (new) | P1-5 |
