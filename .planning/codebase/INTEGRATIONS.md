# External Integrations

**Analysis Date:** 2026-01-16

## APIs & External Services

**HuggingFace Hub:**
- Purpose: Model search, metadata, and downloads
- SDK: `huggingface-hub` for `snapshot_download()`
- REST API: Direct calls to `https://huggingface.co/api` for search
  - Uses `expand=safetensors` to get model sizes in single request
  - Avoids N+1 API calls from SDK
- Client: `backend/mlx_manager/services/hf_client.py`
- API wrapper: `backend/mlx_manager/services/hf_api.py`
- Auth: None required (public models only)
- Configurable: `MLX_MANAGER_HF_ORGANIZATION` (default: `mlx-community`)
- Offline mode: `MLX_MANAGER_OFFLINE_MODE=true` disables all HF calls

**MLX OpenAI Server:**
- Purpose: Local LLM inference with OpenAI-compatible API
- Package: `mlx-openai-server>=1.4.0`
- Subprocess management: `backend/mlx_manager/services/server_manager.py`
- Health check: Polls `/v1/models` endpoint (no `/health` available)
- Darwin only: Conditional dependency in `pyproject.toml`

## Data Storage

**SQLite Database:**
- Location: `~/.mlx-manager/mlx-manager.db` (configurable)
- Driver: `aiosqlite` (async)
- ORM: SQLModel (SQLAlchemy + Pydantic)
- Connection: `backend/mlx_manager/database.py`
- Schema migration: Manual `ALTER TABLE` in `migrate_schema()`
- Tables:
  - `server_profiles` - Server configuration profiles
  - `running_instances` - Active server tracking
  - `downloaded_models` - Model cache metadata
  - `downloads` - Download progress tracking
  - `settings` - Key-value app settings

**File Storage:**
- HuggingFace cache: `~/.cache/huggingface/hub` (configurable)
- Server logs: `/tmp/mlx-manager-server-{profile_id}.log`
- Launchd plists: `~/Library/LaunchAgents/com.mlx-manager.*.plist`

**Caching:**
- HuggingFace Hub built-in caching for model files
- No Redis/Memcached - single-user desktop app

## Authentication & Identity

**Auth Provider:** None
- Single-user desktop application
- No login/authentication required
- CORS configured for localhost origins only

## Monitoring & Observability

**Error Tracking:** None (no Sentry/Datadog)

**Logs:**
- Python logging to stdout
- Log levels: configurable via `MLX_MANAGER_DEBUG`
- Server subprocess logs: captured to temp files
- Third-party noise suppression: httpx, httpcore, uvicorn.access, huggingface_hub

**Health Monitoring:**
- Background service: `backend/mlx_manager/services/health_checker.py`
- Polls running servers every 30s (configurable)
- Updates `running_instances.health_status` in database

## CI/CD & Deployment

**Hosting:**
- Self-hosted desktop application
- No cloud deployment

**CI Pipeline:**
- GitHub Actions: `.github/workflows/ci.yml`
- Jobs:
  - `backend` - Python tests (ubuntu-latest, Python 3.12)
  - `frontend` - TypeScript tests (ubuntu-latest, Node 20)
  - `e2e` - Playwright tests (requires both)
  - `update-coverage-badge` - Dynamic Gist badge (main only)
- Pre-commit hooks: `.pre-commit-config.yaml`
  - ruff (lint/format)
  - mypy (type check)
  - prettier (format)
  - eslint + svelte-check
  - Backend/frontend tests (pre-push only)

**Release:**
- `.github/workflows/release.yml` - GitHub releases
- `.github/workflows/deploy_to_pypi.yml` - PyPI publishing
- Homebrew formula: `Formula/` directory

## Environment Configuration

**Required env vars:** None (all have defaults)

**Optional env vars:**
- `MLX_MANAGER_DATABASE_PATH` - Database location
- `MLX_MANAGER_HF_CACHE_PATH` - HuggingFace cache
- `MLX_MANAGER_HF_ORGANIZATION` - Model filter
- `MLX_MANAGER_DEFAULT_PORT_START` - Port allocation start
- `MLX_MANAGER_OFFLINE_MODE` - Disable network features
- `MLX_MANAGER_DEBUG` - Enable debug logging

**Secrets location:**
- GitHub secrets for CI: `GIST_TOKEN` (coverage badge)
- No runtime secrets (public API access only)

## Webhooks & Callbacks

**Incoming:**
- None (desktop app)

**Outgoing:**
- None (no external webhooks)

## macOS System Integration

**launchd Services:**
- Manager: `backend/mlx_manager/services/manager_launchd.py`
  - Label: `com.mlx-manager`
  - Auto-start MLX Manager on login
- MLX Servers: `backend/mlx_manager/services/launchd.py`
  - Label: `com.mlx-manager.{profile-name}`
  - Auto-start individual MLX servers
- Plist location: `~/Library/LaunchAgents/`
- Commands: `launchctl load/unload/start/stop/list`

**Status Bar App:**
- Framework: rumps
- Features: server start/stop, health status, dashboard link
- Entry point: `mlx-manager menubar`

---

*Integration audit: 2026-01-16*
