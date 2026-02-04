# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX Model Manager is a web application for managing MLX-optimized language models on Apple Silicon Macs. It provides a UI for browsing/downloading models from HuggingFace's mlx-community, managing server profiles, controlling server instances, and configuring launchd services.

## Installation

### From PyPI (Recommended)
```bash
pip install mlx-manager
# or with uv
uvx mlx-manager serve
```

### From Source
```bash
./scripts/build.sh
pip install backend/dist/mlx_manager-*.whl
```

## Usage

```bash
# Start the web server
mlx-manager serve

# Launch menubar app (with auto-start server)
mlx-manager menubar

# Install as launchd service (auto-start on login)
mlx-manager install-service

# Show running servers
mlx-manager status
```

## Development Commands

### Quick Start (Makefile)

The project includes a root-level Makefile for all common operations:

```bash
# View all available commands
make help

# Install development dependencies
make install-dev

# Run all tests
make test

# Run all quality checks (lint + check + test)
make ci

# Start development servers
make dev

# Build for production
make build
```

### Alternative: Direct Commands

```bash
# Start both backend and frontend in development mode
./scripts/dev.sh
```

### Backend (FastAPI + SQLModel)
```bash
cd backend
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run development server
uvicorn mlx_manager.main:app --reload --port 8080

# Linting (with auto-fix)
ruff check . --fix

# Formatting
ruff format .

# Type checking
mypy mlx_manager

# Run all tests
pytest -v

# Run tests with coverage
pytest --cov=mlx_manager --cov-report=term-missing

# Run specific test file
pytest tests/test_profiles.py -v

# Run specific test
pytest tests/test_profiles.py::test_create_profile -v
```

### Frontend (SvelteKit 2 + Svelte 5 + TailwindCSS)
```bash
cd frontend

# Install dependencies
npm install

# Development server
npm run dev

# Type checking
npm run check

# Linting
npm run lint

# Formatting
npm run format

# Unit tests
npm run test

# Unit tests with watch mode
npm run test:watch

# E2E tests (requires dev server)
npm run test:e2e

# Production build
npm run build
```

## Quality Gates

Pre-commit hooks enforce quality on every commit. Install with:
```bash
pip install pre-commit
pre-commit install
```

### Running All Quality Checks

**Backend:**
```bash
cd backend && source .venv/bin/activate
ruff check . && ruff format --check . && mypy mlx_manager && pytest -v
```

**Frontend:**
```bash
cd frontend
npm run check && npm run lint && npm run test
```

### Test Coverage (Current: 67%)

| Module | Coverage |
|--------|----------|
| routers/profiles.py | 100% |
| routers/system.py | 92% |
| services/health_checker.py | 100% |
| services/launchd.py | 96% |
| services/hf_client.py | 89% |
| services/server_manager.py | 80% |

## Architecture

### Backend Structure
- **mlx_manager/main.py**: FastAPI app with lifespan handlers and static file serving
- **mlx_manager/cli.py**: CLI entry point (typer-based)
- **mlx_manager/menubar.py**: macOS status bar app (rumps-based)
- **mlx_manager/routers/**: API endpoints organized by domain (profiles, models, servers, system, settings, chat)
- **mlx_manager/mlx_server/**: Embedded MLX inference server (mounted at /v1):
  - `services/inference.py`: Direct inference via mlx-lm/mlx-vlm/mlx-embeddings
  - `models/pool.py`: Model pool manager with LRU eviction
  - `routers/`: OpenAI-compatible API endpoints (chat, completions, embeddings)
- **mlx_manager/services/**: Business logic singletons:
  - `hf_client.py`: HuggingFace Hub integration for model search/download
  - `health_checker.py`: Background health monitoring
  - `launchd.py`: macOS launchd plist generation for MLX Manager
  - `manager_launchd.py`: launchd service for MLX Manager itself
- **mlx_manager/models.py**: SQLModel entities and Pydantic response schemas
- **mlx_manager/database.py**: Async SQLite via aiosqlite with session management
- **mlx_manager/config.py**: Settings with `MLX_MANAGER_` env prefix, defaults in `~/.mlx-manager/`
- **mlx_manager/static/**: Embedded frontend build (production only)

### Frontend Structure
- **src/lib/api/**: Type-safe API client (`client.ts`) with typed responses (`types.ts`)
- **src/lib/stores/**: Svelte 5 runes-based stores (profiles, servers, system)
- **src/lib/components/**: UI components organized by feature (models/, profiles/, servers/, ui/)
- **src/routes/**: SvelteKit file-based routing (models/, profiles/, servers/)
- Vite proxies `/api` requests to backend at `localhost:8080`

### Key Patterns
- Backend uses singleton services instantiated at module level
- Database sessions via FastAPI's `Depends(get_db)` injection
- Frontend uses Svelte 5 runes (`$state`, `$derived`) for reactivity
- UI built with bits-ui components and Tailwind variants

## API Endpoints
- `GET/POST /api/profiles` - Server profile CRUD
- `GET /api/models/search`, `POST /api/models/download` - HuggingFace integration
- `POST /api/servers/{id}/start|stop|restart` - Instance control
- `POST /api/system/launchd/install|uninstall/{id}` - macOS service management

## Configuration
Backend settings can be configured via environment variables with `MLX_MANAGER_` prefix:
- `MLX_MANAGER_DATABASE_PATH`: SQLite database location (default: `~/.mlx-manager/mlx-manager.db`)
- `MLX_MANAGER_HF_CACHE_PATH`: HuggingFace cache directory
- `MLX_MANAGER_DEFAULT_PORT_START`: Starting port for new profiles (default: 10240)
- `MLX_MANAGER_LOG_LEVEL`: Log level (default: INFO, use DEBUG for verbose output)

## Testing Models

When testing inference features, use these recommended models from mlx-community:

| Model | Type | Features | Notes |
|-------|------|----------|-------|
| `mlx-community/Qwen3-0.6B-4bit-DWQ` | Text (LM) | Thinking, Tools | Small and fast, good for quick tests |
| `mlx-community/GLM-4.7-Flash-4bit` | Text (LM) | Thinking, Tools | Very powerful, reliable tool calling |
| `mlx-community/gemma-3-27b-it-4bit-DWQ` | Vision | Images, Videos | Multimodal input support |
| `mlx-community/all-MiniLM-L6-v2-4bit` | Embeddings | — | Requires mlx-embeddings (not yet in Profile UI) |
| `mlx-community/Kokoro-82M-4bit` | Audio | TTS, STT | Text-to-speech/Speech-to-text (not yet supported) |

**Current limitations:**
- Profile model_type only supports "lm" (text) and "multimodal" (vision), not "embeddings" or "audio"
- Audio models (TTS/STT) are not yet integrated
