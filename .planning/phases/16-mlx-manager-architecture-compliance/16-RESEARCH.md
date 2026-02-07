# Phase 16: MLX Manager Architecture Compliance - Research

**Researched:** 2026-02-07
**Domain:** FastAPI auth dependencies, SSE/WebSocket query-param authentication, router decoupling, deprecated endpoint cleanup
**Confidence:** HIGH

## Summary

This phase addresses compliance gaps between the codebase and the authoritative `backend/mlx_manager/ARCHITECTURE.md` blueprint. The research investigated three requirement areas: (1) SSE/WebSocket query-parameter authentication (ARCH-01), (2) router decoupling from mlx_server internals (ARCH-02), and (3) housekeeping -- deprecated endpoint removal and JWT secret startup warning (ARCH-03).

The codebase has clear, measurable gaps. The SSE download progress endpoint (`/api/models/download/{taskId}/progress`) uses standard `get_current_user` (Bearer header), which **will not work** with browser EventSource since EventSource cannot send custom headers. The WebSocket audit log endpoint (`/api/system/ws/audit-logs`) accepts connections **without any authentication at all**. The frontend SSE download store already omits the token entirely (no `?token=` in the EventSource URL). The frontend WebSocket connection in `auditLogs.createWebSocket()` also omits the token. Two deprecated parser-options endpoints remain with "empty list" stubs. The JWT secret default placeholder exists but no startup warning is emitted.

**Primary recommendation:** Create a shared `get_current_user_from_token()` dependency in `dependencies.py` that extracts JWT from query parameters and feeds it into the existing `decode_token()` pipeline; apply it to SSE and WebSocket endpoints; update both frontend connection sites; add JWT secret warning to lifespan; remove deprecated endpoints and their tests.

## Standard Stack

This phase uses **only existing project dependencies** -- no new libraries are needed.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | 0.115+ | HTTP framework | Already in use; provides `WebSocket`, `Query`, `Depends` |
| authlib.jose | installed | JWT decode/validate | Already the project's JWT library per ARCHITECTURE.md |
| httpx | 0.27+ | Async test client | Already used for all backend tests |
| Starlette | (via FastAPI) | WebSocket, Query param extraction | Built-in, already available |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | installed | Test framework | Writing new query-param auth tests |
| loguru | installed | Logging | JWT secret startup warning |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual query param extraction | FastAPI `Query` dependency | Both work; manual is simpler for WebSocket since WS doesn't support full `Depends` chain with DB |
| Dedicated WS auth middleware | Per-endpoint extraction | Per-endpoint is simpler for 1-2 endpoints |

**Installation:**
```bash
# No new dependencies needed
```

## Architecture Patterns

### Recommended Approach: Shared Token Extraction Dependency

The ARCHITECTURE.md specifies a `get_current_user_from_token` dependency that handles query parameter extraction and reuses the same `decode_token()` pipeline. This should live in `dependencies.py` alongside the existing `get_current_user`.

### Pattern 1: Query Parameter Auth for SSE Endpoints
**What:** Extract JWT from `?token=<jwt>` query parameter, validate with existing `decode_token()`, look up user
**When to use:** SSE endpoints that use `StreamingResponse` (like download progress)
**Example:**
```python
# Source: ARCHITECTURE.md Section 4.3-4.4
from fastapi import Query, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

async def get_current_user_from_token(
    token: str = Query(..., description="JWT token for SSE/WS auth"),
    session: AsyncSession = Depends(get_db),
) -> User:
    """Get current user from query parameter token.

    Used for SSE and WebSocket endpoints where Authorization header
    is not available (EventSource/WebSocket APIs don't support custom headers).
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    email: str | None = payload.get("sub")
    if email is None:
        raise credentials_exception

    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None or user.status != UserStatus.APPROVED:
        raise credentials_exception

    return user
```

### Pattern 2: WebSocket Query Param Auth (Before Accept)
**What:** Extract and validate JWT from query params **before** calling `websocket.accept()`
**When to use:** WebSocket endpoints
**Example:**
```python
# Source: FastAPI WebSocket auth pattern + ARCHITECTURE.md Section 5.4
@router.websocket("/ws/audit-logs")
async def proxy_audit_log_stream(websocket: WebSocket) -> None:
    # Extract token from query params BEFORE accepting
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)  # Policy Violation
        return

    # Validate token using existing decode_token pipeline
    payload = decode_token(token)
    if payload is None:
        await websocket.close(code=1008)
        return

    email = payload.get("sub")
    if not email:
        await websocket.close(code=1008)
        return

    # Validate user exists and is approved (need manual DB session)
    # ... user lookup ...

    await websocket.accept()
    # ... proceed with proxying ...
```

**Note on WebSocket auth:** FastAPI's `Depends()` works with WebSocket endpoints but the dependency chain for DB session injection is more complex. The WebSocket handler cannot use `get_current_user_from_token` as a `Depends()` directly in the same ergonomic way as REST endpoints because WebSocket error handling differs (close with code instead of HTTPException). The recommended approach is to create a helper function that the WebSocket handler calls explicitly before accepting.

### Pattern 3: Frontend Token Injection
**What:** Pass auth token as query parameter in EventSource and WebSocket URLs
**When to use:** Frontend SSE/WS connections
**Example:**
```typescript
// Source: ARCHITECTURE.md Sections 8.3 and 8.4
// SSE with token
import { authStore } from '$lib/stores/auth.svelte';

const token = authStore.token;
const eventSource = new EventSource(
    `/api/models/download/${taskId}/progress?token=${token}`
);

// WebSocket with token
const token = authStore.token;
const wsUrl = `${protocol}//${host}/api/system/ws/audit-logs?token=${token}`;
const ws = new WebSocket(wsUrl);
```

### Pattern 4: JWT Secret Startup Warning
**What:** Check if JWT secret is the default placeholder during lifespan startup; log a warning
**When to use:** Application startup in `lifespan()` in `main.py`
**Example:**
```python
# Source: ARCHITECTURE.md Section 11, item 7
DEFAULT_JWT_SECRET = "CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check JWT secret
    if manager_settings.jwt_secret == DEFAULT_JWT_SECRET:
        logger.warning(
            "JWT secret is using the default placeholder! "
            "Set MLX_MANAGER_JWT_SECRET environment variable for production use."
        )
    # ... rest of startup ...
```

### Anti-Patterns to Avoid
- **Accepting WebSocket before auth validation:** Always validate the token BEFORE calling `websocket.accept()`. Accepting first leaks that the endpoint exists and wastes resources.
- **Creating separate token decode logic:** Reuse the existing `decode_token()` from `auth_service.py`. Don't create a parallel JWT validation path.
- **Using `auto_error=False` on OAuth2PasswordBearer for SSE:** This would make all REST endpoints optionally authenticated. Instead, create a separate dependency specifically for query-param auth.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JWT validation | Custom JWT parsing | `auth_service.decode_token()` | Already handles signature verification, expiry, error cases |
| User lookup from token | Inline DB query | Extract common logic from `get_current_user()` | Avoid duplicating user lookup + status check logic |
| WebSocket close codes | Arbitrary codes | `1008` (Policy Violation) | Standard WebSocket close code for auth failures |

**Key insight:** The entire auth pipeline already exists in `dependencies.py` (`get_current_user`) and `auth_service.py` (`decode_token`). The only new code needed is the query-parameter extraction layer that feeds into the same pipeline.

## Common Pitfalls

### Pitfall 1: EventSource Cannot Send Headers
**What goes wrong:** Using `Depends(get_current_user)` (which extracts from Authorization header) on an SSE endpoint. Browser EventSource API has no mechanism to set custom headers.
**Why it happens:** The current SSE download progress endpoint uses `Depends(get_current_user)` which works only because the frontend doesn't actually enforce auth on SSE connections currently. It silently fails or the endpoint is exposed without auth.
**How to avoid:** Use a separate `get_current_user_from_token` dependency that reads from query parameter instead.
**Warning signs:** SSE connections fail with 401 when frontend properly sends token as query param but backend expects header.

### Pitfall 2: WebSocket Auth After Accept
**What goes wrong:** Accepting the WebSocket connection, then checking auth, then closing if unauthorized.
**Why it happens:** Developers follow the "accept first, validate later" pattern from tutorials.
**How to avoid:** Check `websocket.query_params.get("token")` and validate BEFORE calling `websocket.accept()`. Close with code 1008 if invalid.
**Warning signs:** Unauthorized users can trigger WebSocket handshake completion before being rejected.

### Pitfall 3: Breaking Existing SSE Tests
**What goes wrong:** Adding required `token` query param to SSE endpoint breaks existing tests that don't pass it.
**Why it happens:** Tests use `auth_client` fixture with Bearer header but no query params.
**How to avoid:** Update tests to pass `?token=<jwt>` in the URL when testing SSE endpoints. The `auth_token` fixture already exists in conftest.py.
**Warning signs:** Test failures on download progress endpoint after adding query param auth.

### Pitfall 4: Router Decoupling Scope Creep
**What goes wrong:** Trying to eliminate ALL mlx_server imports from routers. The ARCHITECTURE.md actually permits imports of the "public Python API" (`get_model_pool()`, `generate_chat_completion()`, etc.).
**Why it happens:** Misreading "Routers import from services and models, never from mlx_server internals" as "no mlx_server imports at all."
**How to avoid:** The rule is about not importing from *internal implementation details*. `get_model_pool()`, `detect_model_type()`, `generate_chat_completion()`, `generate_vision_completion()`, `preprocess_images()`, `get_memory_usage()`, and `get_router()` are all **public API** functions. They are explicitly listed as valid integration points in ARCHITECTURE.md Section 3.
**Warning signs:** Refactoring chat.py or servers.py when those imports are already compliant.

### Pitfall 5: Removing detect-options Endpoint by Mistake
**What goes wrong:** Confusing `/api/models/detect-options/{model_id}` (which is in the ARCHITECTURE.md endpoint reference and still used) with the deprecated parser-options endpoints.
**Why it happens:** Both relate to "parser options" conceptually.
**How to avoid:** Only remove the TWO explicitly deprecated endpoints: `GET /api/models/available-parsers` and `GET /api/system/parser-options`. The `detect-options` endpoint is still functional and listed in ARCHITECTURE.md.
**Warning signs:** Frontend breaks because `detectOptions()` API call returns 404.

### Pitfall 6: Frontend Types Cleanup Incomplete
**What goes wrong:** Removing backend endpoints but leaving dead types/functions in frontend.
**Why it happens:** Only looking at backend files.
**How to avoid:** Also remove `ParserOptions` type from `types.ts`, `getAvailableParsers()` from `client.ts`, `parserOptions()` from `client.ts`, and their corresponding tests.
**Warning signs:** Dead code in frontend, unused exports.

## Code Examples

### Current State: SSE Endpoint (BROKEN for browser EventSource)
```python
# Source: backend/mlx_manager/routers/models.py lines 115-118
# This uses get_current_user which extracts from Authorization header
# EventSource API cannot send custom headers
@router.get("/download/{task_id}/progress")
async def get_download_progress(
    current_user: Annotated[User, Depends(get_current_user)],  # <-- Won't work with EventSource
    task_id: str,
):
```

### Current State: WebSocket Endpoint (NO AUTH)
```python
# Source: backend/mlx_manager/routers/system.py lines 297-298
# No authentication at all - anyone can connect
@router.websocket("/ws/audit-logs")
async def proxy_audit_log_stream(websocket: WebSocket) -> None:
    await websocket.accept()  # <-- Accepts without any auth check
```

### Current State: Frontend SSE (NO TOKEN)
```typescript
// Source: frontend/src/lib/stores/downloads.svelte.ts lines 146-148
// No token passed in URL
const eventSource = new EventSource(
    `/api/models/download/${taskId}/progress`,  // <-- Missing ?token=<jwt>
);
```

### Current State: Frontend WebSocket (NO TOKEN)
```typescript
// Source: frontend/src/lib/api/client.ts line 700
// No token passed in URL
return new WebSocket(`${protocol}//${host}/api/system/ws/audit-logs`);  // <-- Missing ?token=<jwt>
```

### Target State: Shared Query Param Auth Dependency
```python
# dependencies.py - new function alongside existing get_current_user
async def get_current_user_from_token(
    token: str = Query(..., description="JWT authentication token"),
    session: AsyncSession = Depends(get_db),
) -> User:
    """Authenticate user from query parameter token (SSE/WebSocket)."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception
    email: str | None = payload.get("sub")
    if email is None:
        raise credentials_exception
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None or user.status != UserStatus.APPROVED:
        raise credentials_exception
    return user
```

### Target State: SSE Endpoint with Query Param Auth
```python
@router.get("/download/{task_id}/progress")
async def get_download_progress(
    current_user: Annotated[User, Depends(get_current_user_from_token)],  # Query param
    task_id: str,
):
```

### Target State: WebSocket with Auth Before Accept
```python
@router.websocket("/ws/audit-logs")
async def proxy_audit_log_stream(websocket: WebSocket) -> None:
    # Validate auth BEFORE accepting connection
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return

    # Use shared decode pipeline
    user = await _validate_ws_token(token)
    if user is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    # ... proxy logic ...
```

### Target State: Frontend SSE with Token
```typescript
// downloads.svelte.ts
import { authStore } from '$lib/stores/auth.svelte';

private connectSSE(modelId: string, taskId: string): void {
    this.closeSSE(modelId);
    const token = authStore.token;
    const eventSource = new EventSource(
        `/api/models/download/${taskId}/progress?token=${token}`,
    );
    // ...
}
```

### Target State: Frontend WebSocket with Token
```typescript
// client.ts
import { authStore } from '$lib/stores/auth.svelte';

createWebSocket: (): WebSocket => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const token = authStore.token;
    return new WebSocket(
        `${protocol}//${host}/api/system/ws/audit-logs?token=${token}`
    );
},
```

## Inventory of Changes Required

### ARCH-01: SSE/WS Auth

| File | Change | Type |
|------|--------|------|
| `backend/mlx_manager/dependencies.py` | Add `get_current_user_from_token()` | New function |
| `backend/mlx_manager/routers/models.py` | SSE endpoint: swap `get_current_user` for `get_current_user_from_token` | Modify |
| `backend/mlx_manager/routers/system.py` | WebSocket: add token validation before accept | Modify |
| `frontend/src/lib/stores/downloads.svelte.ts` | Add `?token=${authStore.token}` to EventSource URL | Modify |
| `frontend/src/lib/api/client.ts` | Add `?token=${authStore.token}` to WebSocket URL | Modify |
| `frontend/src/lib/stores/downloads.svelte.test.ts` | Update EventSource URL expectations | Modify |
| `frontend/src/lib/api/client.test.ts` | Update WebSocket URL expectations | Modify |
| `backend/tests/` (new or existing) | Tests for query-param auth on SSE and WS endpoints | Add tests |

### ARCH-02: Router Decoupling (Assessment)

After analysis, the router imports from mlx_server are **already compliant** with ARCHITECTURE.md Section 3. The imports are all to the **public Python API** (`get_model_pool()`, `detect_model_type()`, `generate_chat_completion()`, etc.), which Section 3 explicitly permits as valid integration points. No router imports from truly internal implementation details (e.g., `_models`, private modules, internal data structures) except for `pool._models` access in `servers.py` -- this is the one violation that should be addressed by using `get_model_pool().get_loaded_models()` or adding a public method.

| File | Current Import | Compliant? | Action |
|------|---------------|------------|--------|
| `routers/chat.py` | `detect_model_type`, `ModelType`, `preprocess_images`, `generate_chat_completion`, `generate_vision_completion` | YES | Public API calls, no change needed |
| `routers/servers.py` | `get_model_pool` (top-level and inline) | YES | Public API, no change needed |
| `routers/servers.py` | `get_memory_usage` | YES | Public API, no change needed |
| `routers/servers.py` | `pool._models` (line 109, 191, 361) | **NO** | Access to private dict; add public method or use existing API |
| `routers/settings.py` | `get_model_pool`, `get_router` | YES | Public API, no change needed |

### ARCH-03: Housekeeping

| File | Change | Type |
|------|--------|------|
| `backend/mlx_manager/main.py` | Add JWT secret warning in `lifespan()` | Modify |
| `backend/mlx_manager/routers/models.py` | Remove `get_available_parsers()` endpoint | Delete code |
| `backend/mlx_manager/routers/system.py` | Remove `get_available_parser_options()` endpoint | Delete code |
| `frontend/src/lib/api/client.ts` | Remove `getAvailableParsers()` and `parserOptions()` | Delete code |
| `frontend/src/lib/api/types.ts` | Remove `ParserOptions` interface | Delete code |
| `frontend/src/lib/api/client.test.ts` | Remove parser-related tests | Delete code |
| `backend/tests/test_models.py` | Remove `test_get_available_parsers_returns_empty_list` | Delete code |
| `backend/tests/test_system.py` | Remove `test_get_parser_options_endpoint` | Delete code |
| `frontend/e2e/app.spec.ts` | Remove parser-options mock (line 184) | Delete code |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No SSE/WS auth | Query param JWT auth | ARCHITECTURE.md (current target) | Security compliance |
| Parser options endpoints (deprecated stubs) | Remove entirely | Phase 13 deprecated them; Phase 16 removes | Dead code cleanup |
| No JWT secret check | Startup warning for default | ARCHITECTURE.md Section 11 | Production safety |

**Deprecated/outdated:**
- `GET /api/models/available-parsers`: Returns empty list, artifact of mlx-openai-server era. Remove.
- `GET /api/system/parser-options`: Returns empty dict, artifact of mlx-openai-server era. Remove.
- `ParserOptions` type in frontend: Dead type, no longer needed. Remove.
- `get_parser_options()` / `find_parser_options()` in `fuzzy_matcher.py` and `model_detection.py`: Already deprecated stubs. Candidate for cleanup but lower priority since they're in utils, not routers.

## Open Questions

1. **Should `detect-options` endpoint also be removed?**
   - What we know: It's still in the ARCHITECTURE.md endpoint reference (Section 12.2). The frontend uses `detectOptions()`. The endpoint returns model family and recommended options (which are now empty).
   - What's unclear: Whether this endpoint provides value now that parser options are empty.
   - Recommendation: Keep it for now -- it still returns `model_family` and `is_downloaded` which are useful. It's in the ARCHITECTURE.md as a valid endpoint. If it should be removed, that's a separate decision.

2. **Should `pool._models` access in servers.py be addressed?**
   - What we know: `servers.py` accesses `pool._models` directly in 3 places (lines 109, 191, 361) to get loaded model metadata (size_gb, loaded_at, preloaded flag). This violates the "never import from mlx_server internals" rule.
   - What's unclear: Whether a public API method already exists or needs to be added to `ModelPoolManager`.
   - Recommendation: Add a `get_model_info(model_id)` public method to `ModelPoolManager` and use it instead of `pool._models` access. This is a minimal change that improves encapsulation.

3. **WebSocket testing with httpx AsyncClient**
   - What we know: httpx AsyncClient doesn't natively support WebSocket connections. Starlette's `TestClient` (sync) has `websocket_connect()` support. For async testing, we'd need either the sync TestClient or a separate WebSocket testing library.
   - What's unclear: The project currently has no WebSocket tests at all for the audit log proxy.
   - Recommendation: Use Starlette's `TestClient` (sync) for WebSocket auth tests since it supports `websocket_connect()` with query params. The auth validation itself is the important thing to test; the proxy behavior is secondary.

4. **Frontend import cycle: downloads.svelte.ts importing from auth.svelte.ts**
   - What we know: The downloads store needs to access `authStore.token` to include in EventSource URLs.
   - What's unclear: Whether this creates a circular import since both are stores.
   - Recommendation: This should be fine since stores are typically leaf modules. The auth store is initialized at module load time and the downloads store accesses it at connection time. No circular dependency concern.

## Sources

### Primary (HIGH confidence)
- `backend/mlx_manager/ARCHITECTURE.md` - Authoritative architecture blueprint, Sections 3, 4, 5, 8, 11, 12
- `backend/mlx_manager/dependencies.py` - Current auth dependency implementation
- `backend/mlx_manager/routers/models.py` - SSE endpoint current implementation
- `backend/mlx_manager/routers/system.py` - WebSocket endpoint current implementation
- `backend/mlx_manager/services/auth_service.py` - JWT decode_token() implementation
- `backend/mlx_manager/config.py` - JWT secret default value
- `backend/mlx_manager/main.py` - Lifespan handler
- `frontend/src/lib/stores/downloads.svelte.ts` - Frontend SSE connection
- `frontend/src/lib/api/client.ts` - Frontend WebSocket connection + parser API calls
- `frontend/src/lib/stores/auth.svelte.ts` - Frontend auth store with token access
- `backend/tests/conftest.py` - Test infrastructure and fixtures

### Secondary (MEDIUM confidence)
- [FastAPI WebSocket Authentication](https://fastapi.tiangolo.com/advanced/websockets/) - WebSocket query param patterns
- [FastAPI OAuth2 JWT Tutorial](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/) - JWT dependency patterns
- [WebSocket Testing with Starlette](https://www.starlette.io/testclient/) - TestClient WebSocket support

### Tertiary (LOW confidence)
- [Medium: Authenticating WebSocket Clients in FastAPI](https://hexshift.medium.com/authenticating-websocket-clients-in-fastapi-with-jwt-and-dependency-injection-d636d48fdf48) - Community patterns for WS auth

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All existing project dependencies, no new libraries needed
- Architecture: HIGH - ARCHITECTURE.md is the authoritative reference and was read in full; all current code was inspected
- Pitfalls: HIGH - Based on direct codebase analysis; every gap and anti-pattern was verified by reading source
- Code examples: HIGH - All examples derived from actual project code patterns

**Research date:** 2026-02-07
**Valid until:** 2026-03-07 (stable -- no external dependencies changing)
