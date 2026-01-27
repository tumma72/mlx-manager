# Codebase Structure

**Analysis Date:** 2026-01-27

## Directory Layout

```
mlx-manager/
├── backend/                          # FastAPI backend (Python)
│   ├── mlx_manager/
│   │   ├── __init__.py              # Version export
│   │   ├── main.py                  # FastAPI app, lifespan, static file serving
│   │   ├── cli.py                   # Typer CLI (serve, menubar, install-service, status)
│   │   ├── menubar.py               # macOS rumps statusbar app
│   │   ├── config.py                # Settings (pydantic-settings with MLX_MANAGER_ prefix)
│   │   ├── database.py              # AsyncIO SQLite engine, session factory, migrations
│   │   ├── dependencies.py          # FastAPI dependency functions (auth, profile lookup)
│   │   ├── models.py                # SQLModel entities and Pydantic response schemas
│   │   ├── types.py                 # Type aliases (HealthCheckResult, RunningServerInfo, etc.)
│   │   ├── routers/                 # API endpoint handlers (organized by domain)
│   │   │   ├── __init__.py
│   │   │   ├── auth.py              # POST /api/auth/* (register, login, password reset)
│   │   │   ├── profiles.py          # GET/POST/PUT /api/profiles (CRUD server profiles)
│   │   │   ├── servers.py           # GET/POST /api/servers (list, start, stop, restart, health, logs)
│   │   │   ├── models.py            # GET/POST /api/models (search, download, local list, progress)
│   │   │   ├── chat.py              # POST /api/chat (streaming chat completion)
│   │   │   ├── mcp.py               # GET /api/mcp (Model Context Protocol info)
│   │   │   └── system.py            # GET /api/system (memory, info, launchd service status)
│   │   ├── services/                # Business logic singletons
│   │   │   ├── __init__.py
│   │   │   ├── server_manager.py    # Subprocess lifecycle (start, stop, restart, process tracking)
│   │   │   ├── health_checker.py    # Background async task for server health polling
│   │   │   ├── hf_client.py         # HuggingFace Hub wrapper (search, download, local list)
│   │   │   ├── hf_api.py            # Direct HuggingFace API calls (search, filter)
│   │   │   ├── auth_service.py      # JWT encoding/decoding, password hashing
│   │   │   ├── launchd.py           # macOS launchd plist generation for MLX servers
│   │   │   ├── manager_launchd.py   # launchd service management for MLX Manager itself
│   │   │   └── parser_options.py    # Tool/reasoning parser configuration retrieval
│   │   ├── utils/                   # Utility functions
│   │   │   ├── __init__.py
│   │   │   ├── command_builder.py   # Construct mlx-openai-server CLI command from profile
│   │   │   ├── model_detection.py   # Model family detection, mlx-lm version compatibility
│   │   │   ├── security.py          # Password hashing utilities
│   │   │   └── fuzzy_matcher.py     # Fuzzy string matching for model search results
│   │   ├── static/                  # Embedded frontend build (production only)
│   │   └── templates/               # (if any - not present in current structure)
│   ├── tests/                       # Unit and integration tests
│   │   ├── conftest.py             # pytest fixtures (mock DB session, auth token, etc.)
│   │   ├── test_*.py               # Test modules parallel to source (e.g., test_profiles.py, test_servers.py)
│   │   └── test_routers_*.py       # Direct router tests with mock dependencies
│   ├── alembic/                    # Database migrations (Alembic)
│   │   ├── env.py                  # Alembic environment config
│   │   ├── script.py.mako          # Migration template
│   │   └── versions/               # Migration files
│   ├── pyproject.toml              # Backend dependencies, scripts, metadata
│   ├── alembic.ini                 # Alembic configuration
│   └── .venv/                      # Virtual environment (development)
│
├── frontend/                        # SvelteKit 2 + Svelte 5 frontend (TypeScript)
│   ├── src/
│   │   ├── routes/                 # SvelteKit file-based routing
│   │   │   ├── +layout.svelte      # Root layout (initializes auth)
│   │   │   ├── (public)/           # Unauthenticated pages
│   │   │   │   ├── +layout.svelte  # Public layout (redirect if authenticated)
│   │   │   │   └── login/
│   │   │   │       └── +page.svelte
│   │   │   └── (protected)/        # Authenticated pages (guarded by +layout.ts)
│   │   │       ├── +layout.svelte  # Protected layout (enforces auth, starts polling)
│   │   │       ├── +layout.ts      # Load function (checks auth, redirects if not)
│   │   │       ├── +page.svelte    # Dashboard / home
│   │   │       ├── models/
│   │   │       │   └── +page.svelte
│   │   │       ├── profiles/
│   │   │       │   ├── +page.svelte
│   │   │       │   ├── new/
│   │   │       │   │   └── +page.svelte
│   │   │       │   └── [id]/
│   │   │       │       └── +page.svelte
│   │   │       ├── servers/
│   │   │       │   └── +page.svelte
│   │   │       ├── chat/
│   │   │       │   └── +page.svelte
│   │   │       └── users/
│   │   │           └── +page.svelte
│   │   │
│   │   └── lib/
│   │       ├── api/                # Type-safe API client
│   │       │   ├── client.ts       # Grouped endpoint definitions (auth, profiles, servers, etc.)
│   │       │   ├── types.ts        # TypeScript types for all API responses/requests
│   │       │   ├── client.test.ts  # Vitest tests for API client
│   │       │   └── index.ts        # Export barrel (re-exports types and client)
│   │       │
│   │       ├── stores/             # Svelte 5 runes-based state management
│   │       │   ├── index.ts        # Export barrel (singletons)
│   │       │   ├── auth.svelte.ts  # User authentication state (token, user info)
│   │       │   ├── profiles.svelte.ts  # ServerProfile list with polling
│   │       │   ├── servers.svelte.ts   # RunningServer list with polling, starting/failed states
│   │       │   ├── models.svelte.ts    # Local model list with polling
│   │       │   ├── downloads.svelte.ts # Active downloads and history
│   │       │   ├── system.svelte.ts    # System memory/CPU info
│   │       │   └── *.svelte.test.ts    # Vitest tests for stores
│   │       │
│   │       ├── services/           # Frontend services
│   │       │   ├── polling-coordinator.svelte.ts  # Deduplicate concurrent refresh requests
│   │       │   └── polling-coordinator.test.ts
│   │       │
│   │       ├── components/         # UI components organized by feature
│   │       │   ├── index.ts        # Export barrel
│   │       │   ├── layout/
│   │       │   │   ├── Navbar.svelte
│   │       │   │   └── index.ts
│   │       │   ├── models/         # Model-related components
│   │       │   │   ├── ModelCard.svelte
│   │       │   │   ├── FilterModal.svelte
│   │       │   │   ├── FilterChips.svelte
│   │       │   │   ├── DownloadProgressTile.svelte
│   │       │   │   ├── ModelBadges.svelte
│   │       │   │   ├── ModelSpecs.svelte
│   │       │   │   ├── ModelToggle.svelte
│   │       │   │   ├── badges/     # Model attribute badges
│   │       │   │   ├── filter-types.ts  # Filter type definitions
│   │       │   │   └── index.ts
│   │       │   ├── profiles/       # Profile CRUD components
│   │       │   │   ├── ProfileCard.svelte
│   │       │   │   ├── ProfileForm.svelte
│   │       │   │   └── index.ts
│   │       │   ├── servers/        # Server state display components
│   │       │   │   ├── ServerCard.svelte
│   │       │   │   ├── ServerTile.svelte
│   │       │   │   ├── StartingTile.svelte
│   │       │   │   ├── MetricGauge.svelte
│   │       │   │   ├── ProfileSelector.svelte
│   │       │   │   ├── *.test.ts   # Component tests
│   │       │   │   └── index.ts
│   │       │   └── ui/             # Reusable UI primitives
│   │       │       ├── button.svelte
│   │       │       ├── badge.svelte
│   │       │       ├── card.svelte
│   │       │       ├── input.svelte
│   │       │       ├── select.svelte
│   │       │       ├── confirm-dialog.svelte
│   │       │       ├── error-message.svelte
│   │       │       ├── markdown.svelte
│   │       │       ├── thinking-bubble.svelte
│   │       │       ├── tool-call-bubble.svelte
│   │       │       └── index.ts
│   │       │
│   │       ├── utils/              # Utility functions
│   │       │   ├── format.ts       # String formatting (bytes, time, etc.)
│   │       │   ├── reconcile.ts    # In-place array reconciliation (prevent re-renders)
│   │       │   ├── index.ts        # Export barrel
│   │       │   └── *.test.ts       # Tests
│   │       │
│   │       └── index.ts            # Main export barrel
│   │
│   ├── tests/
│   │   └── setup.ts                # Vitest configuration and global setup
│   │
│   ├── static/                     # Static assets (favicon, etc.)
│   ├── svelte.config.js            # SvelteKit configuration
│   ├── vite.config.ts              # Vite build configuration
│   ├── tsconfig.json               # TypeScript configuration
│   ├── tailwind.config.ts          # Tailwind CSS configuration
│   ├── package.json                # Frontend dependencies and scripts
│   └── .svelte-kit/                # SvelteKit generated directory (not committed)
│
├── docs/
│   └── ARCHITECTURE.md             # (Legacy - see .planning/codebase/ARCHITECTURE.md)
│
├── scripts/                        # Build and development scripts
│   ├── build.sh                    # Full build (backend + frontend)
│   ├── dev.sh                      # Start dev servers (backend + frontend)
│   └── ...
│
├── .planning/
│   └── codebase/                   # GSD codebase analysis documents
│       ├── ARCHITECTURE.md
│       ├── STRUCTURE.md
│       └── ...
│
└── Makefile                        # Root-level automation (install-dev, test, build, dev, ci)
```

## Directory Purposes

**backend/mlx_manager/:**
- Purpose: Core application code
- Contains: FastAPI app, SQLModel entities, routers, services, utilities
- Key files: `main.py` (entry point), `models.py` (data models), `database.py` (persistence)

**backend/tests/:**
- Purpose: Automated test suite
- Contains: Unit tests, integration tests, fixtures
- Key pattern: Test modules named `test_*.py` parallel to source (test_profiles.py → routers/profiles.py)

**backend/alembic/:**
- Purpose: Database schema migrations
- Contains: Migration scripts (auto-generated or manually written)
- Key files: `env.py` (runtime config), `versions/` (migration history)

**frontend/src/routes/:**
- Purpose: Page structure and layout
- Contains: SvelteKit file-based routing with layout grouping
- Pattern: `(public)` for login, `(protected)` for authenticated pages

**frontend/src/lib/:**
- Purpose: Reusable code (components, stores, utilities)
- Contains: API client, state stores, UI components, utilities
- No direct routing - imported by pages

**frontend/src/lib/stores/:**
- Purpose: Reactive state with polling
- Pattern: Singleton instances registered with polling coordinator
- Key pattern: In-place array reconciliation via `reconcileArray()` to prevent re-renders

**frontend/src/lib/components/:**
- Purpose: UI presentation
- Organization: Grouped by feature (models/, profiles/, servers/) + UI primitives (ui/)
- Pattern: Components are purely presentational, consuming stores

## Key File Locations

**Entry Points:**
- `backend/mlx_manager/main.py`: FastAPI app (HTTP), lifespan handlers
- `backend/mlx_manager/cli.py`: CLI entry point (mlx-manager command)
- `backend/mlx_manager/menubar.py`: macOS menubar app entry point
- `frontend/src/routes/+layout.svelte`: Frontend root layout

**Configuration:**
- `backend/mlx_manager/config.py`: Application settings (env vars with MLX_MANAGER_ prefix)
- `backend/alembic.ini`: Database migration configuration
- `frontend/svelte.config.js`: SvelteKit framework configuration
- `frontend/vite.config.ts`: Build configuration
- `frontend/tailwind.config.ts`: CSS framework configuration

**Core Logic:**
- `backend/mlx_manager/services/server_manager.py`: Subprocess lifecycle management
- `backend/mlx_manager/services/health_checker.py`: Background health polling
- `backend/mlx_manager/services/hf_client.py`: HuggingFace integration
- `frontend/src/lib/stores/servers.svelte.ts`: Server state with polling
- `frontend/src/lib/services/polling-coordinator.svelte.ts`: Request deduplication

**Testing:**
- `backend/tests/conftest.py`: pytest fixtures and shared test utilities
- `backend/tests/test_*.py`: Unit tests for modules
- `frontend/src/tests/setup.ts`: Vitest configuration
- `frontend/src/lib/**/*.test.ts`: Component and store tests

## Naming Conventions

**Backend Files:**
- `*.py`: Python source files
- `test_*.py`: Unit/integration test files
- `*_test.py`: Alternative test naming (not preferred, but seen in some routers)
- Services/utilities: `module_name.py` (e.g., `server_manager.py`)
- Routers: `domain_name.py` (e.g., `profiles.py`)

**Backend Classes/Functions:**
- `PascalCase`: Classes (ServerManager, HealthChecker, ServerProfile)
- `snake_case`: Functions and methods (start_server, get_db)
- `CONSTANT_CASE`: Module-level constants (STATIC_DIR)
- `_private`: Functions/attributes prefixed with `_` for module-private scope

**Frontend Files:**
- `*.svelte`: Svelte component files
- `*.ts`: TypeScript/JavaScript files
- `*.test.ts`: Vitest test files
- `*.svelte.ts`: Svelte context files (stores, services)
- `+page.svelte` / `+layout.svelte`: SvelteKit routing files

**Frontend Classes/Functions:**
- `PascalCase`: Component names and class names (ServerCard, ApiError)
- `camelCase`: Function names, variables, store methods (startServer, fetchProfiles)
- `CONSTANT_CASE`: Constants (API_BASE)
- Prefix with `$`: Derived values and reactive statements (`$derived`, `$state`)

**Directories:**
- `snake_case`: Most directories (mlx_manager, server_manager)
- Feature grouping: (routes), services, utils, components organized by domain
- Public/Protected grouping: `(public)` and `(protected)` for route organization

## Where to Add New Code

**New API Endpoint:**
1. Create/edit router in `backend/mlx_manager/routers/{domain}.py`
2. Define request/response models in `backend/mlx_manager/models.py` (append to appropriate section)
3. Implement business logic in `backend/mlx_manager/services/` if needed (don't put logic in router)
4. Register router in `backend/mlx_manager/routers/__init__.py` if new router file created
5. Include router in `backend/mlx_manager/main.py` via `app.include_router()`
6. Add tests in `backend/tests/test_{domain}.py`

**New UI Page:**
1. Create `frontend/src/routes/(protected)/{feature}/+page.svelte` (or (public) if unauthenticated)
2. Import stores from `frontend/src/lib/stores/`
3. Create/reuse components from `frontend/src/lib/components/`
4. Add layout file `frontend/src/routes/(protected)/{feature}/+layout.svelte` if needed

**New Feature Component:**
1. Create `frontend/src/lib/components/{feature}/{ComponentName}.svelte`
2. Export in `frontend/src/lib/components/{feature}/index.ts` (barrel file)
3. Add tests in `frontend/src/lib/components/{feature}/{ComponentName}.test.ts`
4. Use stores for state, pass props for configuration

**New Service:**
1. Create `backend/mlx_manager/services/service_name.py`
2. Implement as singleton class or module-level functions
3. Export from `backend/mlx_manager/services/__init__.py` if intended as public service
4. Instantiate at module level (e.g., `server_manager = ServerManager()`)
5. Routers depend on service via import (not dependency injection for singletons)

**New Store:**
1. Create `frontend/src/lib/stores/feature.svelte.ts`
2. Define class with `$state` properties and methods
3. Export singleton instance (e.g., `export const featureStore = new FeatureStore()`)
4. Register with polling coordinator in constructor if background refresh needed
5. Export from `frontend/src/lib/stores/index.ts`

**Utilities:**
- Python: `backend/mlx_manager/utils/{utility_name}.py`
- TypeScript: `frontend/src/lib/utils/{utility_name}.ts`
- Both: Organize by functionality, export from barrel files

## Special Directories

**backend/mlx_manager/static/:**
- Purpose: Embedded frontend build (production only)
- Generated: Yes (built by `npm run build`, copied by backend build script)
- Committed: No (in .gitignore, built during deployment)
- Contents: SvelteKit build output (_app/ assets, index.html, favicon.png)

**backend/.venv/:**
- Purpose: Python virtual environment
- Generated: Yes (pip install -e ".[dev]")
- Committed: No (in .gitignore)
- Initialization: Run `pip install -e ".[dev]"` from backend/ directory

**frontend/.svelte-kit/:**
- Purpose: SvelteKit build cache and generated types
- Generated: Yes (during npm run dev / npm run build)
- Committed: No (in .gitignore)
- Initialization: Auto-generated on first npm run dev

**backend/alembic/versions/:**
- Purpose: Database migration history
- Generated: Yes (alembic revision --autogenerate -m "description")
- Committed: Yes (source of truth for schema evolution)
- Usage: Run `alembic upgrade head` to apply pending migrations

**frontend/coverage/**
- Purpose: Code coverage reports
- Generated: Yes (npm run test -- --coverage)
- Committed: No (in .gitignore)
- View: Open frontend/coverage/index.html in browser

**backend/htmlcov/**
- Purpose: Code coverage reports (HTML)
- Generated: Yes (pytest --cov --cov-report=html)
- Committed: No (in .gitignore)
- View: Open backend/htmlcov/index.html in browser

---

*Structure analysis: 2026-01-27*
