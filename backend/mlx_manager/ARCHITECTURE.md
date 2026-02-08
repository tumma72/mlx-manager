# MLX Manager — Architecture Blueprint

This document defines the target architecture of the `mlx_manager` component: the
management UI and API layer for MLX Model Manager. It provides model browsing,
downloading, server profile management, user authentication, and cloud provider
configuration. It is the authoritative reference for how the component **should**
work. Deviations from this document are tracked separately as compliance issues.

The companion blueprint for the embedded inference server lives at
`mlx_server/ARCHITECTURE.md`.

---

## 1. Principles

1. **AuthLib as the single security library**: All cryptographic operations — JWT
   signing, JWE encryption, token validation — use `authlib.jose`. Password hashing
   uses `pwdlib[argon2]`. No other security libraries (pyjwt, cryptography,
   python-jose, passlib, bcrypt) are permitted as dependencies.
2. **Unified token transport**: Every real-time channel (REST, SSE, WebSocket) must
   authenticate the user. The token transport mechanism must work for all channel
   types without requiring protocol-specific workarounds in business logic.
3. **Thin routers, thick services**: Routers validate input, enforce auth, and
   format responses. Business logic lives in services. Routers never import from
   `mlx_server` internals directly.
4. **Singleton services with test isolation**: Long-lived services follow the
   `get_*()` / `reset_*()` pattern. Module-level instances are acceptable for
   stateless utility services (auth, encryption).
5. **Frontend as a stateless SPA**: The SvelteKit frontend stores only a JWT token
   and derived user state. All data is fetched on demand from the API. Route
   protection happens at both the frontend (layout guards) and backend
   (dependency injection).

---

## 2. Component Map

```
mlx_manager/
  main.py                   # FastAPI app, lifespan, CORS, static serving
  config.py                 # Settings (env: MLX_MANAGER_*)
  database.py               # Async SQLite (aiosqlite), migrations, session factory
  models.py                 # SQLModel entities + Pydantic request/response schemas
  types.py                  # TypedDict definitions for internal data structures
  dependencies.py           # FastAPI auth dependencies (get_current_user, get_admin_user)
  cli.py                    # Typer CLI entry point
  menubar.py                # macOS status bar app (rumps)
  logging_config.py         # Loguru setup + stdlib interception

  routers/                  # API endpoints (thin — validate, auth, dispatch, format)
    auth.py                 # POST /api/auth/login, /register, /me, user management
    profiles.py             # CRUD /api/profiles
    models.py               # /api/models/search, download, local, SSE progress
    servers.py              # /api/servers — model pool control
    chat.py                 # POST /api/chat/completions — SSE streaming
    settings.py             # Cloud providers, routing rules, pool config, timeouts
    system.py               # System info, launchd, audit log proxy, WebSocket
    mcp.py                  # MCP tool listing and execution

  services/                 # Business logic
    auth_service.py         # JWT create/decode (authlib.jose), password hash (pwdlib)
    encryption_service.py   # API key encrypt/decrypt (authlib JWE: A256KW+A256GCM)
    hf_client.py            # HuggingFace Hub integration (search, download, delete)
    hf_api.py               # Direct HF REST API for model search with sizing
    health_checker.py       # Background model pool health monitoring
    launchd.py              # macOS launchd plist generation and service control
    manager_launchd.py      # MLX Manager's own launchd service

  utils/                    # Shared utilities
    security.py             # Model path validation
    model_detection.py      # Extract model characteristics from config.json
    fuzzy_matcher.py        # Fuzzy search for model names

  observability/            # LogFire instrumentation
    logfire_config.py       # configure_logfire(), instrument_*() helpers

  mlx_server/               # Embedded inference server (see mlx_server/ARCHITECTURE.md)
  static/                   # Embedded frontend build (production only)
```

### Frontend Structure

```
frontend/src/
  routes/
    (public)/               # Unauthenticated routes
      login/+page.svelte    # Login form
    (protected)/            # Authenticated routes (layout guard)
      +layout.ts            # Auth validation on every navigation
      +page.svelte          # Dashboard
      models/               # Model browsing and download
      profiles/             # Profile CRUD
      servers/              # Model pool management
      settings/             # Cloud providers, routing rules, pool config
      chat/                 # Chat UI
      users/                # Admin user management

  lib/
    api/
      client.ts             # Fetch wrapper with auth header injection
      types.ts              # TypeScript API response types

    stores/                 # Svelte 5 runes-based state
      auth.svelte.ts        # JWT token + user state (localStorage)
      profiles.svelte.ts    # Server profiles
      models.svelte.ts      # Model search and local models
      downloads.svelte.ts   # Download progress (SSE connections)
      servers.svelte.ts     # Running servers / pool status
      settings.svelte.ts    # Cloud providers and routing rules
      system.svelte.ts      # System info

    components/             # UI components by feature domain
      layout/               # Shell, navigation, sidebar
      models/               # Model cards, download tiles
      profiles/             # Profile forms, cards
      servers/              # Server status, memory display
      settings/             # Provider forms, rule editor, audit logs
      ui/                   # Shared primitives (buttons, dialogs, inputs)
```

---

## 3. Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (SvelteKit SPA)                      │
│  Route guards validate auth on navigation. API client injects   │
│  Bearer token on every fetch. Stores manage local state.        │
│  SSE and WebSocket connections authenticate via token query     │
│  parameter.                                                     │
├─────────────────────────────────────────────────────────────────┤
│                     API / Router Layer                           │
│  Thin endpoints. Validate request schemas. Inject auth via      │
│  Depends(get_current_user). Dispatch to services. Format        │
│  responses. Never contain business logic.                       │
├─────────────────────────────────────────────────────────────────┤
│                      Service Layer                              │
│  Business logic singletons. Auth service (JWT, passwords).      │
│  Encryption service (API keys). HF client (model lifecycle).    │
│  Health checker (background monitoring). Launchd manager.       │
├─────────────────────────────────────────────────────────────────┤
│                     Data / Model Layer                           │
│  SQLModel entities (User, ServerProfile, Download,              │
│  CloudCredential, BackendMapping, ServerConfig, Setting).       │
│  Async SQLite sessions via aiosqlite.                           │
├─────────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                           │
│  Configuration (pydantic-settings). Logging (loguru).           │
│  Observability (LogFire). Static file serving.                  │
│  Embedded mlx_server mounted at /v1.                            │
└─────────────────────────────────────────────────────────────────┘
```

**Layer rules:**
- The frontend communicates only with the API layer via HTTP/SSE/WebSocket.
- Routers import from services and models, never from `mlx_server` internals.
- Services import from models and config, never from routers.
- The only integration point with `mlx_server` is through its public Python API
  (`get_model_pool()`, `generate_chat_completion()`, etc.) — called from routers
  or services that bridge the two components (chat, servers, system).

---

## 4. Authentication Architecture

### 4.1 Security Stack

| Concern             | Library               | Algorithm / Method              |
|---------------------|-----------------------|---------------------------------|
| Password hashing    | `pwdlib[argon2]`      | Argon2id (salted, constant-time)|
| JWT signing         | `authlib.jose`        | HS256 (HMAC-SHA256)             |
| API key encryption  | `authlib.jose` (JWE)  | A256KW + A256GCM                |
| Key derivation      | `hashlib`             | SHA-256 of `jwt_secret`         |

All cryptographic operations derive from a single secret: `MLX_MANAGER_JWT_SECRET`.

### 4.2 Token Lifecycle

```
         POST /api/auth/login
              │
              ▼
    Validate email + password (Argon2 verify)
              │
              ▼
    create_access_token(sub=email, exp=7d)
              │
              ▼
    Return { access_token, token_type: "bearer" }
              │
              ▼
    Frontend stores token in localStorage
              │
    ┌─────────┴──────────┐
    │                     │
    ▼                     ▼
  REST requests        SSE / WebSocket
  Authorization:       ?token=<jwt>
  Bearer <jwt>         (query parameter)
```

### 4.3 Token Transport — Dual Mode

The system supports two token transport mechanisms to accommodate all channel types:

| Channel    | Transport                          | Extraction                       |
|------------|------------------------------------|----------------------------------|
| REST API   | `Authorization: Bearer <token>`    | `OAuth2PasswordBearer` dependency|
| SSE        | `?token=<token>` query parameter   | Manual extraction in endpoint    |
| WebSocket  | `?token=<token>` query parameter   | Manual extraction on connect     |

**Why dual mode:** The browser's `EventSource` API and `WebSocket` constructor
cannot send custom HTTP headers. The `Authorization` header is only available in
`fetch()` calls. For SSE and WebSocket, the token must travel as a query parameter.

**Security considerations for query parameter tokens:**
- Tokens in query strings appear in server access logs and browser history.
- For a local-first application (localhost only), this is an acceptable trade-off.
- HTTPS in production deployments prevents token interception in transit.
- The same JWT is used regardless of transport — no separate token types needed.

### 4.4 Auth Dependency Chain

```python
# dependencies.py

OAuth2PasswordBearer(tokenUrl="/api/auth/login")
      │
      ▼
get_current_user(token)          # Decode JWT → query User → check APPROVED
      │
      ▼
get_admin_user(current_user)     # Check is_admin flag (403 if not)
```

For SSE/WebSocket endpoints, a parallel extraction function reads the token from
the query parameter and feeds it into the same `decode_token()` → user lookup
pipeline, ensuring identical validation regardless of transport.

### 4.5 User Lifecycle

```
Register (first user → admin + APPROVED)
    │
    ▼ (subsequent users)
PENDING ──admin approves──▶ APPROVED ──admin disables──▶ DISABLED
                                │
                                ▼
                        Can authenticate
                        Can access all non-admin endpoints
```

### 4.6 API Key Encryption

Cloud provider API keys are encrypted at rest using JWE:

```
plaintext API key
      │
      ▼
encrypt_api_key()   ──▶  JWE compact serialization (5 parts)
      │                       stored in CloudCredential.encrypted_api_key
      ▼
decrypt_api_key()   ◀──  read from DB when testing provider connection
```

The encryption key is derived deterministically: `SHA-256(jwt_secret)` → 256-bit
AES key. Each encryption produces a unique ciphertext (random IV per operation).

---

## 5. Request Flows

### 5.1 REST API (standard authenticated request)

```
Frontend fetch("/api/profiles")
    │  Authorization: Bearer <jwt>
    ▼
FastAPI middleware (CORS, LogFire)
    │
    ▼
profiles_router.list_profiles()
    │  Depends(get_current_user) → User
    │  Depends(get_db) → AsyncSession
    ▼
Database query → ServerProfile list
    │
    ▼
JSON response
```

### 5.2 SSE — Download Progress

```
Frontend: EventSource("/api/models/download/{taskId}/progress?token=<jwt>")
    │
    ▼
models_router.get_download_progress()
    │  Extract token from query param
    │  decode_token() → validate → User
    ▼
async generator:
    │  yield SSE events: {status, progress, downloaded_bytes, total_bytes}
    │  update DB periodically
    │  auto-cleanup after 60s
    ▼
EventSource.onmessage → update download store
```

### 5.3 SSE — Chat Streaming

```
Frontend: fetch("/api/chat/completions", { method: "POST", body, headers })
    │  Authorization: Bearer <jwt>
    ▼
chat_router.chat_completions()
    │  Depends(get_current_user)
    │  Detect model type (text vs vision)
    ▼
mlx_server.services.inference.generate_chat_completion()
    │  or mlx_server.services.vision.generate_vision_completion()
    ▼
StreamingResponse (text/event-stream):
    events: thinking, thinking_done, response, tool_call, tool_calls_done, done, error
```

Note: Chat streaming uses POST + fetch (not EventSource), so the Authorization
header works normally. No query parameter needed.

### 5.4 WebSocket — Audit Log Stream

```
Frontend: new WebSocket("ws://host/api/system/ws/audit-logs?token=<jwt>")
    │
    ▼
system_router.websocket_audit_logs()
    │  Extract token from query param
    │  decode_token() → validate → User
    ▼
Bidirectional proxy to mlx_server:
    ws://localhost:{port}/v1/admin/ws/audit-logs
    │
    ▼
Frontend receives: { type: "log", data: {...} }
```

### 5.5 Model Download (full lifecycle)

```
POST /api/models/download  { model_id }
    │  → creates task_id, registers cancel_event, starts background task
    │  → creates Download record in DB
    ▼
EventSource /api/models/download/{taskId}/progress?token=<jwt>
    │  → streams progress events
    ▼
POST /api/models/download/{downloadId}/pause
    │  → sets cancel_event, marks DB as paused
    ▼
POST /api/models/download/{downloadId}/resume
    │  → creates new task_id, starts new background task
    │  → frontend connects new EventSource
    ▼
POST /api/models/download/{downloadId}/cancel
    │  → sets cancel_event, cleans up partial files, marks DB
```

---

## 6. Service Layer

### 6.1 Service Inventory

| Service              | File                  | Pattern       | Responsibility                          |
|----------------------|-----------------------|---------------|-----------------------------------------|
| Auth Service         | `auth_service.py`     | Functions     | JWT create/decode, password hash/verify |
| Encryption Service   | `encryption_service.py`| Functions (cached) | API key JWE encrypt/decrypt       |
| HF Client            | `hf_client.py`        | Singleton     | Model search, download, delete, list    |
| HF API               | `hf_api.py`           | Functions     | Direct HF REST API for model sizing     |
| Health Checker        | `health_checker.py`   | Singleton     | Background model pool health monitoring |
| Launchd Manager       | `launchd.py`          | Singleton     | macOS launchd plist management          |
| Manager Launchd       | `manager_launchd.py`  | Functions     | MLX Manager's own launchd service       |

### 6.2 Auth Service Contract

```python
# auth_service.py — stateless functions, no class needed

hash_password(password: str) -> str
    # Argon2id hash with random salt

verify_password(plain_password: str, hashed_password: str) -> bool
    # Constant-time comparison

create_access_token(data: dict, expires_delta: timedelta | None) -> str
    # JWT with HS256 signing, default 7-day expiry

decode_token(token: str) -> dict | None
    # Validate signature + expiry, return payload or None
```

### 6.3 Encryption Service Contract

```python
# encryption_service.py — stateless functions with cached key derivation

encrypt_api_key(plain_key: str) -> str
    # JWE compact serialization (A256KW + A256GCM)

decrypt_api_key(encrypted_key: str) -> str
    # Decrypt or raise DecryptionError

clear_cache() -> None
    # Clear LRU-cached JWE key (for testing)
```

### 6.4 Download Infrastructure

The download system bridges async FastAPI with synchronous `huggingface_hub`:

```
POST /api/models/download
    │
    ▼
hf_client.download_model()     ← async generator
    │
    ├── dry_run in executor     ← get total size
    ├── snapshot_download()     ← in executor (blocking)
    │   └── CancellableProgress(tqdm)  ← checks cancel_event on each bar update
    ├── poll directory size     ← async loop every 1s
    └── yield DownloadStatus    ← progress events
```

Cancellation uses `threading.Event` + custom tqdm subclass that raises
`DownloadCancelledError` when the event is set.

---

## 7. Data Layer

### 7.1 Database Entities

| Entity           | Table              | Purpose                                      |
|------------------|--------------------|-----------------------------------------------|
| `User`           | `users`            | Email, hashed_password, is_admin, status      |
| `ServerProfile`  | `server_profiles`  | Model config (path, type, context, prompts)   |
| `Download`       | `downloads`        | Download tracking (status, bytes, timestamps) |
| `DownloadedModel`| `downloaded_models`| Cache of local model metadata                 |
| `CloudCredential`| `cloud_credentials`| Encrypted API keys for cloud backends         |
| `BackendMapping` | `backend_mappings` | Model pattern → cloud backend routing rules   |
| `ServerConfig`   | `server_config`    | Singleton pool configuration                  |
| `Setting`        | `settings`         | Key-value app settings                        |

### 7.2 Session Management

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except HTTPException:
            raise               # Don't log expected HTTP errors
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### 7.3 Schema Migrations

Migrations are additive-only (ALTER TABLE ADD COLUMN) due to SQLite limitations.
Run at startup in `migrate_schema()`. Columns are never dropped.

---

## 8. Frontend Architecture

### 8.1 Auth Flow

```
App Mount
    │
    ▼
AuthStore.initialize()         ← read token + user from localStorage
    │
    ▼
(protected)/+layout.ts load()  ← on every protected route navigation
    │
    ├── authStore.isAuthenticated?  ← token AND user present?
    │   └── NO → redirect /login
    │
    ├── auth.me()              ← validate token with backend
    │   └── 401 → clearAuth() → redirect /login
    │
    └── OK → render page
```

### 8.2 API Client

```typescript
// client.ts

function getAuthHeaders(): HeadersInit {
    const token = authStore.token;
    return {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
    };
}

function handleResponse(response: Response): void {
    if (response.status === 401) {
        authStore.clearAuth();
        window.location.href = "/login";
        throw new ApiError(401, "Session expired");
    }
}
```

### 8.3 SSE Connection with Auth

```typescript
// downloads.svelte.ts

private connectSSE(modelId: string, taskId: string): void {
    const token = authStore.token;
    const url = `/api/models/download/${taskId}/progress?token=${token}`;
    const eventSource = new EventSource(url);
    // ...
}
```

### 8.4 WebSocket Connection with Auth

```typescript
// AuditLogPanel.svelte

function connectWebSocket(): void {
    const token = authStore.token;
    const wsUrl = `${wsProtocol}://${host}/api/system/ws/audit-logs?token=${token}`;
    const ws = new WebSocket(wsUrl);
    // ...
}
```

### 8.5 Store Pattern

All stores use Svelte 5 runes (`$state`, `$derived`) and follow this pattern:

```typescript
class SomeStore {
    items = $state<Item[]>([]);
    loading = $state(false);
    error = $state<string | null>(null);

    get isEmpty(): boolean { return this.items.length === 0; }

    async load(): Promise<void> {
        this.loading = true;
        try {
            this.items = await api.getItems();
        } catch (e) {
            this.error = e.message;
        } finally {
            this.loading = false;
        }
    }
}

export const someStore = new SomeStore();
```

---

## 9. Application Lifecycle

### 9.1 Startup (Lifespan)

```
1. setup_logging()           ← loguru + stdlib interception
2. configure_logfire()       ← optional observability
3. init_db()                 ← create tables, run migrations
4. ModelPoolManager()        ← initialize model pool
5. recover_downloads()       ← resume interrupted downloads
6. health_checker.start()    ← begin background monitoring
7. Ready
```

### 9.2 Shutdown

```
1. cancel_download_tasks()   ← cancel running downloads
2. health_checker.stop()     ← stop monitoring
3. scheduler.shutdown()      ← stop batching (if enabled)
4. pool.cleanup()            ← unload all models
5. logger.complete()         ← flush pending logs
```

### 9.3 App Mounting

```
FastAPI app (port 10242)
    │
    ├── /api/auth/*         ← auth_router
    ├── /api/chat/*         ← chat_router
    ├── /api/mcp/*          ← mcp_router
    ├── /api/models/*       ← models_router
    ├── /api/profiles/*     ← profiles_router
    ├── /api/servers/*      ← servers_router
    ├── /api/settings/*     ← settings_router
    ├── /api/system/*       ← system_router
    ├── /v1/*               ← mlx_server (embedded sub-app)
    ├── /health             ← unauthenticated health check
    └── /*                  ← SPA static files (production)
```

---

## 10. Cross-Cutting Concerns

### 10.1 Observability

- **LogFire**: Wraps FastAPI, httpx, SQLAlchemy with automatic spans.
  Configured via `LOGFIRE_TOKEN` env var. Disabled when token absent.
- **Logging**: loguru with configurable level (`MLX_MANAGER_LOG_LEVEL`).
  Intercepts stdlib logging for unified output.

### 10.2 CORS

Development origins (localhost:5173, 5174, current port) are explicitly
allowed. `allow_credentials=True` enables cookie/auth header passthrough.

### 10.3 Error Handling

- **HTTPException**: FastAPI standard, used for all expected errors (401, 403,
  404, 409). Not logged as errors in the database session handler.
- **Unhandled exceptions**: Caught by FastAPI's default handler, logged by
  loguru, returned as 500.

### 10.4 Configuration

`Settings` via pydantic-settings (`MLX_MANAGER_` prefix):

| Setting             | Default                              | Purpose                    |
|---------------------|--------------------------------------|----------------------------|
| `jwt_secret`        | `"CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR"` | JWT signing + JWE key derivation |
| `jwt_algorithm`     | `"HS256"`                            | JWT signing algorithm      |
| `jwt_expire_days`   | `7`                                  | Token expiration           |
| `database_path`     | `~/.mlx-manager/mlx-manager.db`     | SQLite database location   |
| `hf_cache_path`     | `~/.cache/huggingface/hub`           | Model download cache       |
| `hf_organization`   | `None`                               | Filter to specific HF org  |
| `offline_mode`      | `False`                              | Disable HF network access  |
| `port`              | `10242`                              | HTTP server port           |
| `host`              | `"127.0.0.1"`                        | Bind address               |
| `log_level`         | `"INFO"`                             | Logging verbosity          |

---

## 11. Security Invariants

These properties must hold at all times:

1. **Every endpoint except `/api/auth/login`, `/api/auth/register`, and
   `/health` requires a valid JWT** — either via Authorization header or query
   parameter.
2. **Admin endpoints require `is_admin=True`** — checked via `get_admin_user()`.
3. **Only APPROVED users can authenticate** — PENDING and DISABLED users are
   rejected at the dependency level, not just at login.
4. **The last admin cannot be deleted or demoted** — prevents lockout.
5. **API keys are never stored in plaintext** — always JWE-encrypted with a key
   derived from `jwt_secret`.
6. **API keys are never returned to the frontend** — responses for cloud
   credentials omit the `encrypted_api_key` field.
7. **JWT secret must be overridden in production** — the default value is a
   placeholder that should trigger a warning or startup check.
8. **Password hashing uses Argon2id** — the current recommended algorithm,
   with salt and constant-time verification.

---

## 12. Endpoint Reference

### 12.1 Unprotected

| Method | Path                    | Purpose              |
|--------|-------------------------|----------------------|
| POST   | `/api/auth/register`    | Create user account  |
| POST   | `/api/auth/login`       | Exchange credentials for JWT |
| GET    | `/health`               | Health check         |

### 12.2 Authenticated (APPROVED user)

| Method | Path                                        | Transport | Notes                    |
|--------|---------------------------------------------|-----------|--------------------------|
| GET    | `/api/auth/me`                              | REST      |                          |
| GET    | `/api/profiles`                             | REST      |                          |
| GET    | `/api/profiles/{id}`                        | REST      |                          |
| POST   | `/api/profiles`                             | REST      |                          |
| PUT    | `/api/profiles/{id}`                        | REST      |                          |
| DELETE | `/api/profiles/{id}`                        | REST      |                          |
| POST   | `/api/profiles/{id}/duplicate`              | REST      |                          |
| GET    | `/api/models/search`                        | REST      |                          |
| GET    | `/api/models/local`                         | REST      |                          |
| POST   | `/api/models/download`                      | REST      | Returns task_id          |
| GET    | `/api/models/download/{taskId}/progress`    | **SSE**   | Query param auth         |
| GET    | `/api/models/download/{taskId}/status`      | REST      | Polling fallback         |
| GET    | `/api/models/downloads/active`              | REST      |                          |
| POST   | `/api/models/download/{id}/pause`           | REST      |                          |
| POST   | `/api/models/download/{id}/resume`          | REST      |                          |
| POST   | `/api/models/download/{id}/cancel`          | REST      |                          |
| DELETE | `/api/models/{model_id}`                    | REST      |                          |
| GET    | `/api/models/config/{model_id}`             | REST      |                          |
| GET    | `/api/models/detect-options/{model_id}`     | REST      |                          |
| GET    | `/api/servers`                              | REST      |                          |
| GET    | `/api/servers/embedded`                     | REST      |                          |
| GET    | `/api/servers/models`                       | REST      |                          |
| GET    | `/api/servers/health`                       | REST      |                          |
| GET    | `/api/servers/memory`                       | REST      |                          |
| POST   | `/api/servers/{id}/start`                   | REST      |                          |
| POST   | `/api/servers/{id}/stop`                    | REST      |                          |
| POST   | `/api/servers/{id}/restart`                 | REST      | Legacy                   |
| GET    | `/api/servers/{id}/status`                  | REST      |                          |
| GET    | `/api/servers/{id}/health`                  | REST      |                          |
| POST   | `/api/chat/completions`                     | **SSE**   | POST+fetch (header auth) |
| GET    | `/api/settings/providers`                   | REST      |                          |
| POST   | `/api/settings/providers`                   | REST      |                          |
| DELETE | `/api/settings/providers/{type}`            | REST      |                          |
| GET    | `/api/settings/providers/defaults`          | REST      |                          |
| POST   | `/api/settings/providers/{type}/test`       | REST      |                          |
| GET    | `/api/settings/rules`                       | REST      |                          |
| POST   | `/api/settings/rules`                       | REST      |                          |
| PUT    | `/api/settings/rules/priorities`            | REST      |                          |
| POST   | `/api/settings/rules/test`                  | REST      |                          |
| PUT    | `/api/settings/rules/{id}`                  | REST      |                          |
| DELETE | `/api/settings/rules/{id}`                  | REST      |                          |
| GET    | `/api/settings/pool`                        | REST      |                          |
| PUT    | `/api/settings/pool`                        | REST      |                          |
| GET    | `/api/settings/pool/status`                 | REST      |                          |
| GET    | `/api/settings/timeouts`                    | REST      |                          |
| PUT    | `/api/settings/timeouts`                    | REST      |                          |
| GET    | `/api/system/memory`                        | REST      |                          |
| GET    | `/api/system/info`                          | REST      |                          |
| POST   | `/api/system/launchd/install/{id}`          | REST      |                          |
| POST   | `/api/system/launchd/uninstall/{id}`        | REST      |                          |
| GET    | `/api/system/launchd/status/{id}`           | REST      |                          |
| GET    | `/api/system/audit-logs`                    | REST      | Proxy to mlx_server      |
| GET    | `/api/system/audit-logs/stats`              | REST      | Proxy to mlx_server      |
| GET    | `/api/system/audit-logs/export`             | Streaming | Proxy to mlx_server      |
| WS     | `/api/system/ws/audit-logs`                 | **WS**    | Query param auth         |
| GET    | `/api/mcp/tools`                            | REST      |                          |
| POST   | `/api/mcp/execute`                          | REST      |                          |

### 12.3 Admin Only

| Method | Path                                    | Purpose                    |
|--------|-----------------------------------------|----------------------------|
| GET    | `/api/auth/users`                       | List all users             |
| GET    | `/api/auth/users/pending/count`         | Count pending users        |
| PUT    | `/api/auth/users/{id}`                  | Update user                |
| DELETE | `/api/auth/users/{id}`                  | Delete user                |
| POST   | `/api/auth/users/{id}/reset-password`   | Admin password reset       |

---

## 13. Deployment Modes

| Mode          | Command                   | Behavior                                              |
|---------------|---------------------------|-------------------------------------------------------|
| **Development**| `./scripts/dev.sh`       | Backend on 10241, frontend on 5173 (Vite proxy)      |
| **Production** | `mlx-manager serve`      | Backend on 10242, embedded frontend static files      |
| **Menubar**    | `mlx-manager menubar`    | macOS status bar app, auto-starts server              |
| **Service**    | `mlx-manager install-service` | launchd auto-start on login                      |
