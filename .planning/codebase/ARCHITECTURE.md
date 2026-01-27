# Architecture

**Analysis Date:** 2026-01-27

## Pattern Overview

**Overall:** Client-Server web application with async backend and reactive frontend

**Key Characteristics:**
- **Backend:** FastAPI async REST API with SQLModel/SQLAlchemy for data persistence
- **Frontend:** SvelteKit 2 with Svelte 5 runes for reactive state management
- **Process Management:** Subprocess-based mlx-openai-server orchestration with health monitoring
- **Authentication:** JWT-based with admin/user role separation
- **Data Flow:** Event-driven with polling coordination and Server-Sent Events (SSE) for streaming updates

## Layers

**API Router Layer:**
- Purpose: HTTP endpoint handlers with request/response marshaling and authentication
- Location: `backend/mlx_manager/routers/`
- Contains: Route definitions (`profiles.py`, `servers.py`, `models.py`, `auth.py`, `chat.py`, `mcp.py`, `system.py`)
- Depends on: FastAPI, dependencies (auth), services, database session
- Used by: Frontend via HTTP/API calls

**Service Layer:**
- Purpose: Business logic and external integrations
- Location: `backend/mlx_manager/services/`
- Contains:
  - `server_manager.py`: Process lifecycle management (start, stop, restart, status)
  - `health_checker.py`: Background health polling for running servers
  - `hf_client.py`: HuggingFace Hub integration (search, download, list local models)
  - `auth_service.py`: JWT token generation/validation
  - `launchd.py`: macOS launchd plist generation for MLX server instances
  - `manager_launchd.py`: launchd service for MLX Manager itself
  - `parser_options.py`: Tool/reasoning parser configuration
  - `hf_api.py`: Direct HuggingFace API client
- Depends on: Database session (as dependency), external HTTP clients (httpx), subprocess/psutil, system utilities
- Used by: Routers and main.py lifespan handlers

**Data Access Layer:**
- Purpose: Database operations and session management
- Location: `backend/mlx_manager/database.py`
- Contains: Async SQLite engine setup, session factory, migration utilities, incomplete download recovery
- Depends on: SQLAlchemy, SQLModel, aiosqlite
- Used by: Routers (via Depends(get_db)), services

**Model/Schema Layer:**
- Purpose: Data models (ORM entities), request/response schemas, enums
- Location: `backend/mlx_manager/models.py`, `backend/mlx_manager/types.py`
- Contains:
  - SQLModel tables (User, ServerProfile, Download, RunningInstance, LocalModel)
  - Pydantic response models (ServerProfileResponse, RunningServerResponse, etc.)
  - Enums (UserStatus)
- Depends on: SQLModel, Pydantic
- Used by: Routers, services, database

**Frontend State Layer:**
- Purpose: Svelte 5 runes-based reactive stores with polling coordination
- Location: `frontend/src/lib/stores/`
- Contains: `profiles.svelte.ts`, `servers.svelte.ts`, `models.svelte.ts`, `downloads.svelte.ts`, `auth.svelte.ts`, `system.svelte.ts`
- Pattern: Singleton store instances with reconciliation (in-place array updates to prevent re-renders)
- Depends on: Polling coordinator, API client, Svelte reactivity
- Used by: Page components via store imports

**Frontend Component Layer:**
- Purpose: UI presentation and user interaction
- Location: `frontend/src/lib/components/` and `frontend/src/routes/`
- Contains:
  - Feature components (models/, profiles/, servers/)
  - UI primitives (ui/badge.svelte, button.svelte, etc.)
  - Layout (Navbar.svelte)
- Depends on: Stores, API client (for direct calls), bits-ui components, Tailwind CSS
- Used by: Page routes

**Frontend API Client:**
- Purpose: Type-safe HTTP client with auth token management
- Location: `frontend/src/lib/api/`
- Contains: `client.ts` (grouped endpoints), `types.ts` (TypeScript types), `index.ts` (export barrel)
- Pattern: Grouped namespace objects (auth, profiles, servers, models, etc.) with request/response handling
- Depends on: Fetch API, authStore for token injection
- Used by: Stores and components

## Data Flow

**User Authentication Flow:**

1. User submits login credentials → `POST /api/auth/login` (routers/auth.py)
2. Credential validation → `auth_service.encode_token()` → JWT token returned
3. Frontend stores token in `authStore` (stores/auth.svelte.ts)
4. Subsequent requests inject token via Authorization header (api/client.ts `getAuthHeaders()`)
5. Router dependency `get_current_user` validates token and checks user status

**Server Instance Lifecycle:**

1. User initiates start → `POST /api/servers/{profile_id}/start` (routers/servers.py)
2. Router retrieves profile → `service_manager.start_server(profile)` → subprocess spawned
3. `server_manager.processes` dict tracks running processes
4. `health_checker` background task polls running servers every 30s
5. Frontend `serverStore` polls `/api/servers` every 3 seconds
6. Store updates via `reconcileArray()` to prevent re-renders on unchanged data
7. Components watch `$state` variables reactively

**Model Download Flow:**

1. User triggers download → `POST /api/models/download` (routers/models.py)
2. Download record created in database
3. Background task spawned: `hf_client.download_model()` → yields progress events
4. Task info stored in `download_tasks` dict with task_id
5. Frontend connects to `GET /api/models/download/{task_id}/progress` (SSE)
6. Progress streamed to browser in real-time
7. Database updated on completion/failure
8. Incomplete downloads recovered on app startup

**State Management:**

- **Backend:** SQLite persistent state + in-memory process tracking (ServerManager.processes)
- **Frontend:** Svelte runes-based reactive state in stores
- **Coordination:** Polling coordinator deduplicates concurrent refresh requests to prevent thundering herd
- **Reconciliation:** `reconcileArray()` utility performs in-place array updates, comparing only changed fields

## Key Abstractions

**ServerManager:**
- Purpose: Abstract subprocess lifecycle for mlx-openai-server
- Examples: `backend/mlx_manager/services/server_manager.py`
- Pattern: Singleton service with process dictionary, supports start/stop/restart/health_check
- Key methods: `start_server(profile)`, `stop_server(profile_id)`, `is_running(profile_id)`, `get_process_stats(profile_id)`

**HealthChecker:**
- Purpose: Background task for continuous server health monitoring
- Examples: `backend/mlx_manager/services/health_checker.py`
- Pattern: Async background worker that polls running servers, updates RunningInstance table
- Prevents stale process tracking if process crashes without cleanup

**HFClient:**
- Purpose: Encapsulate HuggingFace Hub interaction
- Examples: `backend/mlx_manager/services/hf_client.py`
- Pattern: Singleton with caching, search_mlx_models(), download_model() (async generator), list_local_models()

**Store (Frontend):**
- Purpose: Centralize reactive state with automatic polling
- Examples: `frontend/src/lib/stores/profiles.svelte.ts`
- Pattern: Class with `$state` for data, methods for mutations, polling registration in constructor
- Uses polling coordinator to prevent duplicate background requests

**PollingCoordinator (Frontend):**
- Purpose: Deduplicate concurrent refresh requests across stores
- Examples: `frontend/src/lib/services/polling-coordinator.svelte.ts`
- Pattern: Central registry that throttles refresh calls (minInterval, maxInterval) and coordinates start/stop

## Entry Points

**Backend:**
- `backend/mlx_manager/main.py`: FastAPI app instance with lifespan handlers (startup/shutdown)
- `backend/mlx_manager/cli.py`: Typer CLI entry point (mlx-manager serve, mlx-manager menubar, etc.)
- `backend/mlx_manager/menubar.py`: macOS statusbar app entry point (rumps-based)

**Frontend:**
- `frontend/src/routes/+layout.svelte`: Root layout, initializes auth and starts polling on mount
- `frontend/src/routes/(protected)/+layout.svelte`: Protected layout, enforces authentication
- `frontend/src/routes/(public)/login/+page.svelte`: Login page

**Initialization (Backend):**

1. `FastAPI(lifespan=lifespan)` defines `async def lifespan(app)` context manager
2. Startup: `init_db()` → alembic migrations, `cleanup_stale_instances()`, `health_checker.start()`, `resume_pending_downloads()`
3. Shutdown: `cancel_download_tasks()`, `health_checker.stop()`, `server_manager.cleanup()`

## Error Handling

**Strategy:** Layered with context-specific detail

**Patterns:**

- **API Layer:** `HTTPException(status_code, detail)` for request validation and business logic errors
  - 400: Bad request (invalid input)
  - 401: Unauthorized (missing/invalid token)
  - 403: Forbidden (insufficient permissions)
  - 404: Not found (resource doesn't exist)
  - 409: Conflict (unique constraint violation, e.g., profile name/port)
  - 500: Internal error (unexpected exception)

- **Service Layer:** Raises `RuntimeError` or domain-specific exceptions which routers catch and map to HTTPException

- **Frontend:** `ApiError` exception with status code, error handler redirects 401 to login, displays user-friendly messages

- **Async Task Errors:** Background tasks (health_checker, download_tasks) log errors and continue; fatal errors reported via UI

## Cross-Cutting Concerns

**Logging:**
- Pattern: `logging.getLogger(__name__)` in each module
- Backend configured at `main.py` startup (INFO level, console output)
- Third-party loggers (httpx, httpcore, uvicorn.access) set to WARNING to reduce noise

**Validation:**
- Backend: FastAPI/Pydantic automatic validation on request body/params
- Router dependency functions validate resource ownership, existence
- Frontend: SvelteKit form validation, API client error formatting

**Authentication:**
- Backend: JWT tokens (HS256, 7-day expiry), dependency injection via OAuth2PasswordBearer
- Frontend: Token stored in `authStore`, injected in all requests, 401 redirects to login
- User status check: Only APPROVED users can access protected endpoints

**Rate Limiting:**
- Backend: Implicit via polling minInterval (throttle concurrent requests to 1/second per resource)
- Frontend: PollingCoordinator enforces minInterval between refresh attempts

**Graceful Degradation:**
- Download recovery: Incomplete downloads resumed on startup (HuggingFace snapshot_download auto-resumes)
- Process cleanup: Stale RunningInstance records removed on startup if process no longer exists
- Health checker: Continues polling even if individual server health check fails

---

*Architecture analysis: 2026-01-27*
