# Technology Stack

**Analysis Date:** 2026-01-27

## Languages

**Primary:**
- Python 3.11-3.12 - Backend API, server management, CLI, menubar app
- TypeScript 5 - Frontend UI components and stores
- JavaScript - SvelteKit configuration and build setup

**Supporting:**
- HTML5/CSS3 (via Svelte 5 components) - UI rendering
- Bash - Development scripts and build automation

## Runtime

**Environment:**
- Python 3.11 - Minimum supported version
- Python 3.12 - Latest tested version
- Node.js 20+ (implied by Vite 6.0 requirement)

**Package Manager:**
- uv - Python package manager (primary, used in CI and scripts)
- npm - Node.js package manager
- Lockfile: `uv.lock` (backend), `package-lock.json` (frontend)

## Frameworks

**Core:**
- FastAPI 0.115+ - Web API framework with async/await support
- uvicorn 0.32+ - ASGI application server with hot reload
- SvelteKit 2.49+ - Full-stack framework for frontend with file-based routing
- Svelte 5 - Component framework with runes-based reactivity

**Data/ORM:**
- SQLModel 0.0.22+ - SQL ORM combining SQLAlchemy and Pydantic
- SQLAlchemy (async) - Async database layer via `sqlalchemy.ext.asyncio`
- aiosqlite 0.20+ - Async SQLite driver

**Build/Dev:**
- Vite 6.0+ - Frontend build tool with HMR
- SvelteKit adapter-static - Static site generation for production frontend
- Alembic 1.18+ - Database schema migrations

**CLI/Menubar:**
- typer 0.12+ - CLI framework for command-line interface (`mlx-manager` CLI)
- rumps 0.4+ - macOS status bar app (menubar application)

**Authentication:**
- PyJWT 2.8+ - JWT token creation and validation
- pwdlib[argon2] 0.3+ - Password hashing with Argon2

**Testing:**
- pytest 8.0+ - Python test framework
- pytest-asyncio 0.24+ - Async test support
- pytest-cov 4.0+ - Test coverage reporting
- vitest 4.0+ - JavaScript/TypeScript unit testing
- Playwright 1.57+ - E2E browser testing
- @testing-library - Component testing utilities (DOM, Svelte, jest-dom)

**Quality Assurance:**
- ruff 0.8+ - Python linting and formatting
- mypy 1.13+ - Python static type checking
- eslint 9.0+ - JavaScript/TypeScript linting
- prettier 3.4+ - Code formatting (JavaScript/TypeScript/Svelte)
- svelte-check 4.0+ - Svelte component type checking

## Key Dependencies

**Critical:**
- huggingface-hub 0.27+ - Model search, download, and caching from HuggingFace Hub
- mlx-openai-server 1.4+ (macOS only) - Local OpenAI-compatible inference server
- mlx-vlm 0.3.9+ - Vision language model support for multimodal models
- httpx 0.28+ - Async HTTP client for HuggingFace API calls and chat requests

**System/Infrastructure:**
- psutil 6.0+ - System metrics (CPU, memory monitoring)
- python-multipart 0.0.6+ - HTTP form parsing for file uploads
- pydantic-settings 2.0+ - Environment variable configuration management
- transformers 5.0rc1+ - Model detection and configuration parsing

**Model Support:**
- rapidfuzz 3.0+ - Fuzzy string matching for model name resolution and search filtering

## Configuration

**Environment:**
- Settings via `pydantic_settings.BaseSettings` with `MLX_MANAGER_` prefix
- Default paths use `~/.mlx-manager/` for database, config, and cache
- Offline mode support via `MLX_MANAGER_OFFLINE_MODE` environment variable

**Key Configuration Points:**
- `MLX_MANAGER_DATABASE_PATH` - SQLite database location (default: `~/.mlx-manager/mlx-manager.db`)
- `MLX_MANAGER_HF_CACHE_PATH` - HuggingFace model cache (default: `~/.cache/huggingface/hub`)
- `MLX_MANAGER_HF_ORGANIZATION` - Filter models by author/org (optional, defaults to all)
- `MLX_MANAGER_JWT_SECRET` - JWT signing key (required for production)
- `MLX_MANAGER_DEFAULT_PORT_START` - Base port for new server profiles (default: 10240)

**Build:**
- `pyproject.toml` - Python project metadata, dependencies, tool config
- `package.json` - Node.js project metadata and scripts
- `svelte.config.js` - SvelteKit adapter and alias configuration
- `vite.config.ts` - Frontend build config with API proxy to backend
- `tailwind.config.ts` - TailwindCSS utility customization
- `.pre-commit-config.yaml` - Git hooks for linting/formatting on commit

**Databases:**
- SQLite 3 (async via aiosqlite) - Single-file database at `~/.mlx-manager/mlx-manager.db`
- Schema managed by SQLModel with Alembic migrations

## Platform Requirements

**Development:**
- macOS (for menubar app via rumps library)
- Python 3.11+ with venv or uv for isolation
- Node.js 20+ with npm
- Pre-commit hooks (`pip install pre-commit && pre-commit install`)

**Production:**
- Deployment target: Apple Silicon Macs (primary), Intel Macs (supported)
- Packaged as PyPI wheel: `mlx-manager` on PyPI
- Alternative: Homebrew formula in `Formula/` directory for native distribution
- Embedded frontend: Static SvelteKit build bundled in Python package

**External Requirements:**
- HuggingFace Hub connectivity (for model search/download)
- mlx-openai-server executable (auto-installed as dependency on macOS)
- MLX framework (indirect via mlx-openai-server)

---

*Stack analysis: 2026-01-27*
