# External Integrations

**Analysis Date:** 2026-01-27

## APIs & External Services

**HuggingFace Hub:**
- Largest integration: Model search, download, and caching
- SDK: `huggingface-hub` package
- REST API: Direct HTTP calls via httpx to `https://huggingface.co/api`
- Files: `mlx_manager/services/hf_client.py`, `mlx_manager/services/hf_api.py`
- Capabilities:
  - Search MLX models by query, author/organization, sort order
  - Download models with progress tracking and resumable downloads
  - Fetch accurate model sizes via `usedStorage` API field
  - List locally cached models and extract config.json for model characteristics
  - Delete downloaded models from cache

**MLX OpenAI Server:**
- Local inference service (subprocess management)
- Binary: `mlx-openai-server` executable (auto-installed on macOS)
- Integration: Server subprocess spawned and managed by `ServerManager`
- Protocol: OpenAI-compatible chat completions API on local port
- Files: `mlx_manager/services/server_manager.py`
- Health checks: HTTP GET requests via httpx to `/health` endpoint

**Chat Completions (Local Inference):**
- Destination: mlx-openai-server running on configurable port (default: 10240+)
- Protocol: OpenAI chat completions API (compatible)
- Streaming: Server-Sent Events (SSE) with typed chunks
- Files: `mlx_manager/routers/chat.py`
- Features:
  - Tool/function calling support with OpenAI function format
  - Thinking model support (GLM-4-turbo-style `<think>` tags or `reasoning_content` field)
  - Chat template customization per profile

## Data Storage

**Databases:**
- SQLite 3 (single-file, async via aiosqlite)
  - Location: `~/.mlx-manager/mlx-manager.db`
  - Config key: `MLX_MANAGER_DATABASE_PATH`
  - Client: SQLModel + SQLAlchemy async engine
  - Tables: Users, ServerProfiles, Settings, RunningInstances, Downloads, DownloadRecords

**File Storage:**
- Local filesystem only (no cloud storage)
- Model cache: HuggingFace cache directory (`~/.cache/huggingface/hub` default)
  - Config key: `MLX_MANAGER_HF_CACHE_PATH`
  - Format: HuggingFace Hub cache structure (models--author--name directories with snapshots)
- Server logs: `~/.mlx-manager/logs/` (per-profile log files)
- Database: `~/.mlx-manager/mlx-manager.db`

**Caching:**
- In-memory: Download task tracking in memory (lost on restart, resumed from DB)
- File-based: HuggingFace Hub snapshot downloads with resumable progress
- No Redis or external caching service

## Authentication & Identity

**Auth Provider:**
- Custom JWT-based authentication (no OAuth/SSO provider)
- Implementation: `mlx_manager/services/auth_service.py`
  - Password hashing: Argon2 via pwdlib
  - Token generation: PyJWT with configurable expiry
  - Algorithm: HS256 (symmetric key)

**User Management:**
- Database: SQLite users table
- Fields: email, hashed_password, is_admin, status (pending/approved/disabled), created_at
- Features: Multi-user with admin approval workflow
- Defaults:
  - JWT_SECRET: `CHANGE_ME_IN_PRODUCTION` (must override in production)
  - JWT_ALGORITHM: HS256
  - JWT_EXPIRE_DAYS: 7 (configurable via `MLX_MANAGER_JWT_EXPIRE_DAYS`)

**Token Flow:**
1. POST `/api/auth/login` - Email/password validation → JWT token
2. Authorization: `Bearer <token>` header on protected endpoints
3. Frontend stores token in memory/localStorage via `authStore`
4. Session expiry: 401 response triggers login redirect

## Monitoring & Observability

**Error Tracking:**
- Built-in logging via Python `logging` module
- Levels: INFO (default), DEBUG (configurable), ERROR, WARNING
- No external error tracking (Sentry, DataDog, etc.)

**Logs:**
- Console output via `logging.StreamHandler(sys.stdout)`
- Log level: Configurable per server profile (`log_level` field in ServerProfile)
- Server logs: File-based per profile at `~/.mlx-manager/logs/{profile_id}.log`
- Suppressed packages: httpx, httpcore, uvicorn.access (set to WARNING)

**Health Monitoring:**
- Background service: `mlx_manager/services/health_checker.py`
- Mechanism: Periodic HTTP health checks to running servers
- Interval: Configurable via `MLX_MANAGER_HEALTH_CHECK_INTERVAL` (default: 30 seconds)
- Metrics collected: CPU usage, memory usage, response time

## CI/CD & Deployment

**Hosting:**
- PyPI: Package distributed as Python wheel (`mlx-manager`)
- Homebrew: macOS binary formula in `Formula/` directory
- Source: GitHub repository with CI/CD workflows

**CI Pipeline:**
- GitHub Actions (`.github/workflows/ci.yml`)
- Triggers: Push to main, pull requests
- Backend: Python 3.12, ruff lint/format, mypy type check, pytest
- Frontend: Node 20, eslint, prettier format check, vitest unit tests, Playwright E2E
- Coverage reporting: pytest-cov badge via gist
- Status: Visible in README with CI badge

**Deployment:**
- Release workflow: `.github/workflows/release.yml`
- PyPI workflow: `.github/workflows/deploy_to_pypi.yml`
- Build: `pip install mlx-manager` or `pip install -e backend/` for development
- Frontend embedding: SvelteKit static build bundled in Python package at `mlx_manager/static/`

## Environment Configuration

**Required env vars:**
- `MLX_MANAGER_JWT_SECRET` - Must be set to a strong random string in production (default: "CHANGE_ME_IN_PRODUCTION")

**Optional env vars:**
- `MLX_MANAGER_DATABASE_PATH` - Database location (default: `~/.mlx-manager/mlx-manager.db`)
- `MLX_MANAGER_HF_CACHE_PATH` - Model cache directory (default: `~/.cache/huggingface/hub`)
- `MLX_MANAGER_HF_ORGANIZATION` - Filter models by org (default: None = all)
- `MLX_MANAGER_OFFLINE_MODE` - Disable HuggingFace API calls (default: false)
- `MLX_MANAGER_DEFAULT_PORT_START` - Base port for profiles (default: 10240)
- `MLX_MANAGER_MAX_MEMORY_PERCENT` - Memory usage limit (default: 80)
- `MLX_MANAGER_HEALTH_CHECK_INTERVAL` - Health check frequency in seconds (default: 30)

**Secrets location:**
- Local environment variables for development
- System env vars or `.env` file (not committed) for production
- JWT secret must be set via `MLX_MANAGER_JWT_SECRET` environment variable

**Secrets NOT in use:**
- HuggingFace token: Optional `HF_TOKEN` env var (uses huggingface-hub default locations)
- API keys: No third-party API keys required

## Webhooks & Callbacks

**Incoming:**
- None - Application receives no inbound webhooks

**Outgoing:**
- None - Application sends no external webhooks or callbacks
- Model downloads: Async generator yields progress events (in-process only)
- Health checks: HTTP GET requests to local mlx-openai-server (internal)

**Event Mechanisms:**
- Server-Sent Events (SSE): Frontend `/api/models/downloads` endpoint streams progress
- Polling: Frontend polls `/api/servers` for server status updates
- In-process: Background tasks for downloads and health monitoring

## Model Operations

**Model Search:**
Files: `mlx_manager/services/hf_api.py`, `mlx_manager/services/hf_client.py`

Search Parameters:
- Query string (model name, architecture, etc.)
- Author/organization filter (e.g., "mlx-community")
- Sort: "downloads", "likes", "lastModified"
- Size filter: Max size in GB
- Limit: Results per search

API Calls:
1. GET `https://huggingface.co/api/models` - Search with filter="mlx"
2. GET `https://huggingface.co/api/models/{model_id}` - Fetch usedStorage (parallel)
3. GET `https://huggingface.co/{model_id}/resolve/main/config.json` - Model config

**Model Download:**
- Uses `huggingface_hub.snapshot_download()` for resumable downloads
- Dry-run first to calculate total size
- Directory polling to report progress (1-second interval)
- Supports interrupted resumption via database recovery on startup

**Model Detection:**
- Extract model characteristics from `config.json`
- Detect model family (Llama, Qwen, Mistral, etc.)
- Check mlx-lm version support before launching server

---

*Integration audit: 2026-01-27*
