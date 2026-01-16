# Codebase Structure

**Analysis Date:** 2026-01-16

## Directory Layout

```
mlx-manager/
├── backend/                    # Python FastAPI backend
│   ├── mlx_manager/           # Main package
│   │   ├── routers/           # API endpoint modules
│   │   ├── services/          # Business logic singletons
│   │   ├── utils/             # Shared utilities
│   │   ├── static/            # Embedded frontend build (production)
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── cli.py             # Typer CLI
│   │   ├── menubar.py         # macOS status bar app
│   │   ├── models.py          # SQLModel/Pydantic schemas
│   │   ├── database.py        # Async SQLite setup
│   │   ├── config.py          # Settings with env vars
│   │   ├── dependencies.py    # FastAPI dependencies
│   │   └── types.py           # TypedDict definitions
│   ├── tests/                 # pytest test suite
│   ├── alembic/               # DB migrations (not actively used)
│   └── pyproject.toml         # Python package config
├── frontend/                  # SvelteKit frontend
│   ├── src/
│   │   ├── routes/            # File-based routing
│   │   ├── lib/
│   │   │   ├── api/           # API client and types
│   │   │   ├── stores/        # Svelte 5 reactive stores
│   │   │   ├── components/    # UI components
│   │   │   ├── services/      # Polling coordinator
│   │   │   └── utils/         # Helper functions
│   │   ├── tests/             # Vitest unit tests
│   │   ├── app.css            # Global styles
│   │   └── app.html           # HTML template
│   ├── static/                # Static assets (favicon, etc.)
│   ├── e2e/                   # Playwright E2E tests
│   └── build/                 # Production build output
├── scripts/                   # Build and dev scripts
├── docs/                      # Documentation
├── Formula/                   # Homebrew formula
├── .planning/                 # GSD planning artifacts
├── Makefile                   # Task runner
└── CLAUDE.md                  # AI assistant instructions
```

## Directory Purposes

**`backend/mlx_manager/`:**
- Purpose: Main Python package, installed via pip
- Contains: FastAPI app, CLI, services, database models
- Key files: `main.py` (app), `cli.py` (entry point), `models.py` (schemas)

**`backend/mlx_manager/routers/`:**
- Purpose: API endpoint definitions, one file per domain
- Contains: `profiles.py`, `models.py`, `servers.py`, `system.py`
- Key files: Each exports a FastAPI `router` with `/api/{domain}` prefix

**`backend/mlx_manager/services/`:**
- Purpose: Business logic, external integrations, singleton services
- Contains: Process management, HuggingFace client, health checker, launchd
- Key files: `server_manager.py` (MLX server lifecycle), `hf_client.py` (model downloads)

**`backend/mlx_manager/utils/`:**
- Purpose: Shared helper functions and utilities
- Contains: Command builder, fuzzy matcher, model detection, security
- Key files: `command_builder.py` (builds mlx-openai-server args), `model_detection.py` (detects model family)

**`frontend/src/routes/`:**
- Purpose: SvelteKit file-based routing
- Contains: Page components, layouts
- Key files: `+layout.svelte` (root layout), `+page.svelte` (home redirect)

**`frontend/src/lib/api/`:**
- Purpose: Type-safe API client for backend communication
- Contains: HTTP client functions, TypeScript types
- Key files: `client.ts` (API methods), `types.ts` (response types)

**`frontend/src/lib/stores/`:**
- Purpose: Svelte 5 runes-based reactive state management
- Contains: Store classes with `$state()` fields
- Key files: `servers.svelte.ts`, `profiles.svelte.ts`, `downloads.svelte.ts`, `system.svelte.ts`

**`frontend/src/lib/components/`:**
- Purpose: Reusable Svelte components organized by feature
- Contains: UI primitives, domain-specific components
- Key files: `ui/` (Button, Card, etc.), `profiles/ProfileCard.svelte`, `models/ModelCard.svelte`

## Key File Locations

**Entry Points:**
- `backend/mlx_manager/main.py`: FastAPI application, lifespan handlers
- `backend/mlx_manager/cli.py`: CLI entry point (`mlx-manager` command)
- `frontend/src/routes/+layout.svelte`: Root layout, store initialization

**Configuration:**
- `backend/mlx_manager/config.py`: `Settings` class with `MLX_MANAGER_` env prefix
- `backend/pyproject.toml`: Python package metadata, dependencies, entry points
- `frontend/svelte.config.js`: SvelteKit adapter, path aliases
- `frontend/vite.config.ts`: Dev server proxy to backend

**Core Logic:**
- `backend/mlx_manager/services/server_manager.py`: MLX server process lifecycle
- `backend/mlx_manager/services/hf_client.py`: HuggingFace model search/download
- `backend/mlx_manager/routers/models.py`: Model download with SSE progress
- `frontend/src/lib/stores/servers.svelte.ts`: Server state with polling

**Testing:**
- `backend/tests/`: pytest test files, `conftest.py` with fixtures
- `frontend/src/tests/`: Vitest unit tests
- `frontend/e2e/`: Playwright E2E tests

## Naming Conventions

**Files:**
- Python: `snake_case.py` (e.g., `server_manager.py`, `hf_client.py`)
- Svelte: `PascalCase.svelte` for components (e.g., `ProfileCard.svelte`)
- Svelte stores: `kebab-case.svelte.ts` (e.g., `servers.svelte.ts`)
- TypeScript: `kebab-case.ts` (e.g., `client.ts`, `types.ts`)

**Directories:**
- Python: `snake_case/` (e.g., `mlx_manager/`, `routers/`)
- Frontend: `kebab-case/` (e.g., `src/lib/`, `components/ui/`)

**Exports:**
- Python services: Singleton instance at module level (e.g., `server_manager = ServerManager()`)
- Svelte stores: Named export (e.g., `export const serverStore = new ServerStore()`)
- API client: Namespaced objects (e.g., `profiles.list()`, `servers.start()`)

## Where to Add New Code

**New API Endpoint:**
- Create or extend router in `backend/mlx_manager/routers/`
- Add route with `@router.get|post|put|delete` decorator
- Register router in `backend/mlx_manager/routers/__init__.py` if new file
- Router auto-included in `main.py` via `__init__.py` exports

**New Service:**
- Create `backend/mlx_manager/services/{service_name}.py`
- Export singleton instance: `my_service = MyService()`
- Add export to `backend/mlx_manager/services/__init__.py`
- Import in routers as needed

**New Frontend Page:**
- Create `frontend/src/routes/{path}/+page.svelte`
- SvelteKit auto-generates route based on directory structure
- Use existing stores or create new one in `frontend/src/lib/stores/`

**New Frontend Component:**
- Add to `frontend/src/lib/components/{feature}/`
- Export from `frontend/src/lib/components/{feature}/index.ts`
- Use path alias: `import { MyComponent } from '$components/feature'`

**New Store:**
- Create `frontend/src/lib/stores/{name}.svelte.ts`
- Use Svelte 5 `$state()` runes for reactive state
- Register with `pollingCoordinator` if needs auto-refresh
- Export from `frontend/src/lib/stores/index.ts`

**New Database Table:**
- Add SQLModel class to `backend/mlx_manager/models.py` with `table=True`
- Tables auto-created on startup via `SQLModel.metadata.create_all()`
- Add migration in `backend/mlx_manager/database.py:migrate_schema()` for schema changes to existing tables

**New Utility Function:**
- Backend: Add to `backend/mlx_manager/utils/{module}.py`
- Frontend: Add to `frontend/src/lib/utils/{module}.ts`

## Special Directories

**`backend/mlx_manager/static/`:**
- Purpose: Embedded frontend build for production distribution
- Generated: Yes, by `scripts/build.sh` copying from `frontend/build/`
- Committed: Yes, included in Python package

**`frontend/build/`:**
- Purpose: SvelteKit production build output
- Generated: Yes, by `npm run build`
- Committed: No (in .gitignore)

**`frontend/.svelte-kit/`:**
- Purpose: SvelteKit generated files
- Generated: Yes, by `npm run dev` or `npm run build`
- Committed: No (in .gitignore)

**`backend/alembic/`:**
- Purpose: Database migration scripts (Alembic)
- Generated: Partially, `env.py` is boilerplate
- Committed: Yes, but not actively used (schema changes via `migrate_schema()`)

**`.planning/`:**
- Purpose: GSD planning documents and codebase analysis
- Generated: No, manually written
- Committed: Varies by project convention

---

*Structure analysis: 2026-01-16*
