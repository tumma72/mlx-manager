# Architecture

**Analysis Date:** 2026-01-16

## Pattern Overview

**Overall:** Layered Monolith with Embedded SPA

**Key Characteristics:**
- FastAPI backend serving REST API and embedded static SPA frontend
- SQLite database with async access via SQLModel/aiosqlite
- Singleton service pattern for process management and external integrations
- SvelteKit frontend compiled to static assets, embedded in Python package for distribution
- Svelte 5 runes-based reactive state management with polling coordination

## Layers

**Presentation Layer (Frontend):**
- Purpose: User interface for managing MLX models and servers
- Location: `frontend/src/`
- Contains: SvelteKit pages, Svelte 5 components, stores, API client
- Depends on: Backend API via HTTP/SSE
- Used by: End users via browser

**API Layer (Routers):**
- Purpose: HTTP endpoint definitions, request validation, response formatting
- Location: `backend/mlx_manager/routers/`
- Contains: FastAPI router modules (`profiles.py`, `models.py`, `servers.py`, `system.py`)
- Depends on: Services, Database, Models
- Used by: Frontend API client, CLI

**Service Layer:**
- Purpose: Business logic, external integrations, process management
- Location: `backend/mlx_manager/services/`
- Contains: Singleton service classes (`server_manager.py`, `hf_client.py`, `health_checker.py`, `launchd.py`)
- Depends on: Types, Config, external APIs (HuggingFace Hub)
- Used by: Routers

**Data Layer (Models/Database):**
- Purpose: Data persistence, schema definitions, session management
- Location: `backend/mlx_manager/models.py`, `backend/mlx_manager/database.py`
- Contains: SQLModel entities, Pydantic response schemas, async session factory
- Depends on: Config
- Used by: Routers, Services

**Utilities Layer:**
- Purpose: Shared helpers, command building, model detection
- Location: `backend/mlx_manager/utils/`
- Contains: Command builder, fuzzy matcher, model detection, security validation
- Depends on: Config
- Used by: Services, Routers

## Data Flow

**Server Profile Management:**

1. User creates/edits profile via SvelteKit form
2. Frontend API client POSTs to `/api/profiles`
3. `profiles_router` validates with Pydantic, persists via SQLModel
4. Response returned, frontend store updated via polling

**Model Download:**

1. User initiates download from Models page
2. Frontend calls `POST /api/models/download` to get `task_id`
3. Frontend connects to SSE stream at `/api/models/download/{task_id}/progress`
4. `hf_client` service uses `huggingface_hub.snapshot_download()` in executor
5. Progress polled by directory size, streamed to frontend
6. Download state persisted to DB for resume capability on restart

**Server Lifecycle:**

1. User clicks Start on profile card
2. Frontend calls `POST /api/servers/{id}/start`
3. `server_manager.start_server()` spawns `mlx-openai-server` subprocess
4. PID recorded in `RunningInstance` table
5. `health_checker` background task polls `/v1/models` endpoint
6. Frontend polls server list, updates UI based on health status

**State Management:**

- Frontend uses Svelte 5 `$state()` runes for reactive state
- Stores wrap state with methods (`serverStore`, `profileStore`, `systemStore`)
- `pollingCoordinator` service manages timed refreshes with deduplication
- HMR state preservation via `window` globals for development

## Key Abstractions

**ServerProfile:**
- Purpose: Configuration for an MLX server instance
- Examples: `backend/mlx_manager/models.py:ServerProfile`
- Pattern: SQLModel entity with Pydantic validation

**Service Singletons:**
- Purpose: Long-lived services initialized at module level
- Examples: `server_manager`, `hf_client`, `health_checker`, `launchd_manager`
- Pattern: Class instantiated once, imported by routers

**API Client:**
- Purpose: Type-safe HTTP client for frontend-backend communication
- Examples: `frontend/src/lib/api/client.ts`
- Pattern: Namespaced async functions (`profiles.list()`, `servers.start()`)

**Reactive Stores:**
- Purpose: Centralized frontend state with automatic polling
- Examples: `frontend/src/lib/stores/servers.svelte.ts`
- Pattern: Class with `$state()` fields, registered with polling coordinator

## Entry Points

**Web Server (`main.py`):**
- Location: `backend/mlx_manager/main.py`
- Triggers: `uvicorn.run()` from CLI or direct execution
- Responsibilities: FastAPI app setup, lifespan handlers (DB init, health checker start/stop), router mounting, static file serving

**CLI (`cli.py`):**
- Location: `backend/mlx_manager/cli.py`
- Triggers: `mlx-manager` command (entry point in `pyproject.toml`)
- Responsibilities: `serve`, `install-service`, `uninstall-service`, `status`, `menubar`, `version` subcommands

**Menubar App (`menubar.py`):**
- Location: `backend/mlx_manager/menubar.py`
- Triggers: `mlx-manager menubar` CLI command
- Responsibilities: macOS status bar app using `rumps`, auto-starts backend server

**Frontend Root Layout:**
- Location: `frontend/src/routes/+layout.svelte`
- Triggers: Any page navigation
- Responsibilities: Initialize stores, start polling, render navbar and page content

## Error Handling

**Strategy:** Exception-based with HTTP status codes, frontend toast notifications

**Patterns:**
- Routers raise `HTTPException` for client errors (400, 404, 409, 500)
- Services raise `RuntimeError` for operational failures
- Frontend `ApiError` class wraps HTTP errors with status and message
- FastAPI validation errors formatted as `field: error message` strings

## Cross-Cutting Concerns

**Logging:** Python `logging` module, configured in `main.py` to stdout, third-party libraries suppressed

**Validation:** Pydantic models for request/response, SQLModel for DB entities, frontend TypeScript types mirrored from backend

**Authentication:** None (local-only application, binds to 127.0.0.1)

**Background Tasks:** AsyncIO tasks for health checking, download progress tracking; `asynccontextmanager` lifespan for startup/shutdown

---

*Architecture analysis: 2026-01-16*
