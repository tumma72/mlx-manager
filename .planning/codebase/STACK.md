# Technology Stack

**Analysis Date:** 2026-01-16

## Languages

**Primary:**
- Python 3.11-3.12 - Backend API, CLI, menubar app
- TypeScript 5.7+ - Frontend SvelteKit application

**Secondary:**
- Shell/Bash - Build scripts (`scripts/dev.sh`, `scripts/build.sh`)

## Runtime

**Backend:**
- Python 3.11+ (requires `>=3.11,<3.13`)
- No `.python-version` file; version specified in `backend/pyproject.toml`

**Frontend:**
- Node.js 20.x (specified in CI workflow)
- No `.nvmrc` file at project level

**Package Managers:**
- uv - Python package management (modern alternative to pip)
- npm - Node.js package management
- Lockfiles: `backend/uv.lock`, `frontend/package-lock.json`

## Frameworks

**Backend:**
- FastAPI 0.115+ - Async REST API framework
- SQLModel 0.0.22+ - SQLAlchemy-based ORM with Pydantic validation
- Typer 0.12+ - CLI framework with Rich integration
- rumps 0.4+ - macOS status bar app (darwin only)

**Frontend:**
- SvelteKit 2.49+ - Full-stack Svelte framework
- Svelte 5.0+ - Reactive UI framework (uses runes: `$state`, `$derived`)
- Vite 6.0+ - Build tool and dev server

**Styling:**
- TailwindCSS 3.4+ - Utility-first CSS framework
- bits-ui 1.0.0-next - Headless UI components
- tailwind-variants - Variant-based component styling
- @tailwindcss/typography - Prose styling plugin

**Testing:**
- pytest 8.0+ / pytest-asyncio - Backend unit tests
- vitest 4.0+ - Frontend unit tests
- Playwright 1.57+ - E2E browser tests
- @testing-library/svelte - Component testing

**Build/Dev:**
- Ruff 0.8+ - Python linting and formatting (replaces black/flake8/isort)
- mypy 1.13+ - Python type checking
- ESLint 9.0+ - TypeScript/Svelte linting
- Prettier 3.4+ - Code formatting

## Key Dependencies

**Backend Critical:**
- `huggingface-hub>=0.27.0` - HuggingFace model downloads
- `aiosqlite>=0.20.0` - Async SQLite driver
- `httpx>=0.28.0` - Async HTTP client
- `uvicorn[standard]>=0.32.0` - ASGI server
- `mlx-openai-server>=1.4.0` - MLX inference server (darwin only)
- `transformers>=5.0.0rc1` - Tokenizer/model config parsing
- `psutil>=6.0.0` - Process management
- `rapidfuzz>=3.0.0` - Fuzzy string matching
- `alembic>=1.18.1` - Database migrations
- `pydantic-settings>=2.0.0` - Settings management with env vars

**Frontend Critical:**
- `lucide-svelte` - Icon library
- `marked` - Markdown rendering
- `clsx` / `tailwind-merge` - Class utilities

## Configuration

**Backend Environment:**
- Prefix: `MLX_MANAGER_` for all env vars
- Key vars:
  - `MLX_MANAGER_DATABASE_PATH`: SQLite path (default: `~/.mlx-manager/mlx-manager.db`)
  - `MLX_MANAGER_HF_CACHE_PATH`: HuggingFace cache (default: `~/.cache/huggingface/hub`)
  - `MLX_MANAGER_HF_ORGANIZATION`: Model org filter (default: `mlx-community`)
  - `MLX_MANAGER_DEFAULT_PORT_START`: Starting port (default: `10240`)
  - `MLX_MANAGER_OFFLINE_MODE`: Disable HF API calls
- Config file: `backend/mlx_manager/config.py`

**Frontend Configuration:**
- `frontend/vite.config.ts` - Dev server proxy (`/api` -> `localhost:8080`)
- `frontend/tailwind.config.ts` - Theme with shadcn/ui CSS variables
- `frontend/tsconfig.json` - Strict TypeScript with bundler resolution

**Build Configuration:**
- `backend/pyproject.toml` - Python project metadata, ruff, mypy, pytest, coverage
- `frontend/package.json` - Scripts, dependencies
- `Makefile` - Root-level task automation

## Platform Requirements

**Development:**
- macOS recommended (menubar app, launchd services)
- Python 3.11 or 3.12
- Node.js 20.x
- uv (`pip install uv`)

**Production:**
- macOS required (Apple Silicon for MLX)
- Self-hosted: `mlx-manager serve` or `mlx-manager install-service`
- Can run on macOS Intel but MLX features require Apple Silicon

**CI/CD:**
- GitHub Actions on ubuntu-latest (for tests)
- macOS not available in CI (MLX features skipped)

---

*Stack analysis: 2026-01-16*
