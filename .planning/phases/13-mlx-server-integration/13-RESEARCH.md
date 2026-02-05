# Phase 13: MLX Server Integration - Research

**Researched:** 2026-01-31
**Domain:** FastAPI application composition, subprocess-to-embedded migration, shared state management
**Confidence:** HIGH

## Summary

This research investigates how to embed the MLX Server (built in phases 7-12) directly into the MLX Manager FastAPI application, replacing the legacy subprocess-based mlx-openai-server architecture. The investigation covers FastAPI application composition patterns, shared resource management (database sessions, model pools, configuration), legacy code removal strategies, and UI integration patterns.

**Key Findings:**
1. **FastAPI `app.mount()` is the standard pattern** - Sub-applications mount at a path prefix (e.g., `/v1`) and operate independently with their own routes, middleware, and lifespan
2. **Shared state via `app.state`** - Resources initialized in the parent app's lifespan can be accessed by mounted apps through `request.app.state`
3. **Database sessions can be shared** - Both apps can use the same SQLAlchemy engine/session factory from parent app's lifespan
4. **Legacy removal is straightforward** - server_manager.py, command_builder.py, parser_options.py can be deleted; Chat UI changes from proxy to direct endpoint
5. **Settings already store configuration** - Phase 11 settings endpoints (model pool, providers, routing rules) are ready to drive the embedded server

**Primary recommendation:** Mount MLX Server at `/v1` prefix in MLX Manager, initialize shared resources (database sessions, model pool) in parent lifespan, update Chat UI to call embedded `/v1/chat/completions`, and delete all subprocess management code.

## Standard Stack

The established libraries/tools for this domain:

### Core (Already Installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | 0.115+ | Parent and sub-application framework | Native sub-app mounting via `.mount()` |
| SQLAlchemy async | Latest | Shared database sessions | Connection pooling, engine sharing across apps |
| Pydantic Settings | 2.10+ | Configuration management | Settings from env vars or database |

### Supporting (Already Installed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| aiosqlite | Latest | Async SQLite driver | Both apps use SQLite for persistence |
| Uvicorn | 0.34+ | Single ASGI server | Runs combined application |

### No New Dependencies Required
All libraries needed for integration are already installed from previous phases.

## Architecture Patterns

### Recommended Project Structure
```
backend/mlx_manager/
├── main.py                      # MODIFIED: Mount MLX Server at /v1
├── routers/
│   ├── chat.py                  # MODIFIED: Remove proxy, use embedded server
│   ├── settings.py              # EXISTS: Drive embedded server config
│   └── ...
├── services/
│   ├── server_manager.py        # DELETE: Legacy subprocess management
│   ├── encryption_service.py    # KEEP: Used by settings
│   └── ...
├── utils/
│   ├── command_builder.py       # DELETE: Built commands for mlx-openai-server
│   ├── parser_options.py        # DELETE: Parsed mlx-openai-server help output
│   └── model_detection.py       # KEEP: Still used for model family detection
├── mlx_server/                  # EXISTS: Built in phases 7-12
│   ├── main.py                  # Sub-application FastAPI instance
│   ├── api/v1/                  # /v1/chat/completions, /v1/models, etc.
│   ├── models/pool.py           # Model pool manager (singleton)
│   ├── services/batching/       # Continuous batching scheduler
│   └── database.py              # Audit log database (separate from manager DB)

frontend/src/
├── routes/(protected)/
│   ├── chat/                    # MODIFIED: Connect to /v1/chat/completions
│   └── settings/                # EXISTS: UI already built in Phase 11
```

### Pattern 1: Mounting Sub-Application with Path Prefix

**What:** Mount MLX Server as independent FastAPI app at `/v1` prefix
**When to use:** When you need isolated apps with different concerns but want single server process
**Example:**
```python
# Source: https://fastapi.tiangolo.com/advanced/sub-applications/
from fastapi import FastAPI
from mlx_manager.mlx_server.main import app as mlx_server_app

# Parent app (MLX Manager)
app = FastAPI(title="MLX Model Manager")

# Mount sub-application
# All MLX Server routes automatically prefixed with /v1
app.mount("/v1", mlx_server_app)

# Parent routes remain at root level
app.include_router(profiles_router)  # /api/profiles
app.include_router(models_router)    # /api/models
app.include_router(settings_router)  # /api/settings

# Result:
# MLX Manager routes: /api/profiles, /api/models, /api/settings, etc.
# MLX Server routes: /v1/chat/completions, /v1/models, /v1/embeddings, etc.
```

**Why this works:**
- Sub-app has own lifespan, middleware, and error handlers
- Routes automatically prefixed (no code changes needed in mlx_server/)
- OpenAPI docs can be separate or merged
- Each app can have own dependency injection scope

**Source:** [FastAPI Sub Applications Documentation](https://fastapi.tiangolo.com/advanced/sub-applications/)

### Pattern 2: Shared State via Parent Lifespan

**What:** Initialize shared resources in parent app's lifespan, access in both apps
**When to use:** When both apps need same resources (database engine, config)
**Example:**
```python
# Source: https://github.com/fastapi/fastapi/discussions/11742
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize shared resources
    await init_db()  # Manager database (profiles, models, settings)

    # Store references in app.state for access from mounted apps
    app.state.manager_db = engine
    app.state.settings = get_settings()

    yield

    # Shutdown: Cleanup
    await engine.dispose()

app = FastAPI(lifespan=lifespan)

# Mounted app can access parent state
@mlx_server_app.get("/v1/config")
def get_config(request: Request):
    # Access parent app state through request.app.state
    settings = request.app.state.settings
    return {"max_memory_gb": settings.max_memory_gb}
```

**Key insight:** Mounted apps receive the parent app instance, so `request.app.state` gives access to parent-initialized resources.

**Source:** [FastAPI State Management](https://sqlpey.com/python/fastapi-state-management-app-vs-request-state/)

### Pattern 3: Database Session Sharing

**What:** Both apps use same database engine but maintain separate session factories
**When to use:** When apps need to query/update shared database (Settings → Model Pool)
**Example:**
```python
# Source: https://github.com/fastapi/fastapi/discussions/9097
# In parent lifespan
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create shared engine
    engine = create_async_engine(DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Store engine in state
    app.state.db_engine = engine

    yield

    await engine.dispose()

# In both apps: Create session factory from shared engine
def get_db(request: Request):
    """Dependency for database sessions."""
    engine = request.app.state.db_engine
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    return async_session()

# Usage in endpoints (both apps)
@router.get("/settings/pool")
async def get_pool_config(session: AsyncSession = Depends(get_db)):
    result = await session.execute(select(ServerConfig))
    return result.scalar_one_or_none()
```

**Critical consideration:** MLX Server has its own audit database (mlx-server.db) separate from Manager database (mlx-manager.db). Only settings need to be shared.

**Source:** [FastAPI Database Connection Pool](https://medium.com/@balakrishna0106/building-a-high-performance-api-with-fastapi-and-postgresql-connection-pool-fbdee86326e4)

### Pattern 4: Configuration Flow (Settings → Embedded Server)

**What:** Settings UI updates database, embedded server reads from database on-demand
**When to use:** User changes model pool settings in UI, server should reflect changes
**Example:**
```python
# Settings endpoint (already exists from Phase 11)
@router.put("/api/settings/pool")
async def update_pool_config(data: ServerConfigUpdate, session: AsyncSession):
    # Update database
    config.memory_limit_value = data.memory_limit_value
    config.eviction_policy = data.eviction_policy
    await session.commit()

    # Trigger server reconfiguration
    # Option 1: Direct access to model pool singleton
    from mlx_manager.mlx_server.models.pool import model_pool
    model_pool.update_config(
        max_memory_gb=data.memory_limit_value,
        eviction_policy=data.eviction_policy
    )

    # Option 2: Emit event that server listens to
    await app.state.config_changed.emit("pool_updated")

    return {"success": True}

# In MLX Server's model pool manager
class ModelPoolManager:
    def update_config(self, max_memory_gb: float, eviction_policy: str):
        """Apply new configuration without restarting server."""
        self.max_memory_gb = max_memory_gb
        self.eviction_policy = eviction_policy
        # May need to evict models if new limit is lower
        if self._current_memory_usage() > max_memory_gb:
            self._evict_to_fit(max_memory_gb)
```

**Implementation options:**
- Direct function call (simple, works if both apps in same process)
- Event bus (more decoupled, useful for future extensions)
- Periodic polling (simplest, server checks DB every N seconds)

### Pattern 5: Chat UI Direct Integration

**What:** Change Chat UI from proxy pattern to direct embedded endpoint calls
**When to use:** Migrating from subprocess proxy to embedded server
**Example:**
```python
# BEFORE (Phase 5): Proxy to subprocess
@router.post("/api/chat/completions")
async def chat_completions(request: ChatRequest):
    # Get profile to determine server URL
    profile = await db.get(ServerProfile, request.profile_id)
    server_url = f"http://{profile.host}:{profile.port}/v1/chat/completions"

    # Proxy to subprocess
    async with httpx.AsyncClient() as client:
        response = await client.post(server_url, json=body)
    return response

# AFTER (Phase 13): Direct call to embedded server
@router.post("/api/chat/completions")
async def chat_completions(request: ChatRequest):
    # No profile lookup needed - server is embedded
    # Direct call to mounted /v1 endpoint
    from mlx_manager.mlx_server.api.v1.chat import chat_completions as mlx_chat

    # Transform request to MLX Server format
    mlx_request = ChatCompletionRequest(
        model=request.model,
        messages=request.messages,
        stream=True
    )

    # Call embedded endpoint directly
    return await mlx_chat(mlx_request)

# ALTERNATIVE: Remove proxy endpoint entirely, let frontend call /v1 directly
# Frontend changes from:
#   POST /api/chat/completions (proxy)
# To:
#   POST /v1/chat/completions (direct)
```

**Recommendation:** Remove proxy endpoint, update frontend to call `/v1/chat/completions` directly. This simplifies architecture and eliminates unnecessary HTTP hop.

### Anti-Patterns to Avoid

- **Importing app instances directly:** Don't `from mlx_server.main import app`; use `.mount()` instead
- **Shared database connections:** Don't share connection objects; share engine and create sessions per-request
- **Tight coupling:** Don't call sub-app endpoints from parent routes; keep concerns separate
- **Port conflicts:** Don't configure MLX Server port when embedded; parent app controls binding

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sub-app composition | Custom ASGI middleware | `app.mount()` | FastAPI's mount handles path prefixing, OpenAPI merging, lifespan coordination |
| Shared resources | Global variables | `app.state` | Avoids import-time side effects, testable, request-scoped access |
| Database pooling | Manual connection management | SQLAlchemy engine + sessionmaker | Built-in pooling, async support, connection lifecycle management |
| Configuration reload | Server restart | Dynamic config update methods | Model pool can update limits without restart, faster for users |

**Key insight:** FastAPI's mounting and state management patterns are battle-tested. Custom solutions introduce bugs around lifespan coordination, middleware ordering, and OpenAPI schema conflicts.

## Common Pitfalls

### Pitfall 1: Separate Lifespan Events Don't Coordinate

**What goes wrong:** Parent app starts before sub-app is ready, or shutdown order causes resource leaks.

**Why it happens:** Both apps have separate lifespan context managers that run independently. If parent app depends on sub-app resources (or vice versa), initialization order matters.

**How to avoid:**
```python
# Initialize shared resources in parent lifespan FIRST
@asynccontextmanager
async def parent_lifespan(app: FastAPI):
    # 1. Initialize shared resources (database, config)
    await init_db()
    app.state.db_engine = engine

    # 2. Mount sub-app (its lifespan runs after parent starts)
    app.mount("/v1", mlx_server_app)

    yield

    # Shutdown happens in reverse order (sub-app first, then parent)
    await engine.dispose()

# Sub-app lifespan can access parent state
@asynccontextmanager
async def mlx_server_lifespan(app: FastAPI):
    # Parent resources already initialized
    # Initialize sub-app-specific resources (model pool)
    pool.model_pool = ModelPoolManager()

    yield

    # Cleanup sub-app resources
    await pool.model_pool.cleanup()
```

**Warning signs:**
- `NoneType has no attribute` errors during startup
- Resources not cleaned up on shutdown
- Sub-app tries to use parent resources before they're initialized

**Source:** [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)

### Pitfall 2: Database Path Conflicts

**What goes wrong:** Both apps try to create/manage the same database file, causing lock errors or schema conflicts.

**Why it happens:** MLX Server has separate audit database (`mlx-server.db`) from Manager database (`mlx-manager.db`). If paths aren't configured correctly, they may conflict.

**How to avoid:**
```python
# MLX Manager database (profiles, models, settings)
MANAGER_DB_PATH = "~/.mlx-manager/mlx-manager.db"

# MLX Server database (audit logs only)
MLX_SERVER_DB_PATH = "~/.mlx-manager/mlx-server.db"

# Each app initializes its own database
@asynccontextmanager
async def parent_lifespan(app: FastAPI):
    # Manager DB
    manager_engine = create_async_engine(f"sqlite+aiosqlite:///{MANAGER_DB_PATH}")
    await init_manager_db(manager_engine)

    # MLX Server initializes its own DB in its lifespan
    app.mount("/v1", mlx_server_app)

    yield
```

**Settings database is shared:** Model pool settings, provider credentials, and routing rules are stored in manager DB and read by embedded server.

**Warning signs:**
- SQLite "database is locked" errors
- Schema mismatch errors (tables from different apps in same DB)
- Audit logs not appearing

### Pitfall 3: Port Configuration Ignored

**What goes wrong:** MLX Server settings include `host` and `port` configuration, but these are ignored when embedded.

**Why it happens:** When mounted as sub-app, MLX Server doesn't bind to a port—the parent app controls the socket. Settings meant for standalone mode are now irrelevant.

**How to avoid:**
```python
# Document in MLX Server config
class MLXServerSettings(BaseSettings):
    # NOTE: host/port are ignored when embedded in MLX Manager
    # Only used when running mlx_server.main:app standalone
    host: str = Field(default="127.0.0.1", description="Host (standalone mode only)")
    port: int = Field(default=10242, description="Port (standalone mode only)")

    # These settings ARE used when embedded
    max_memory_gb: float = Field(default=48.0)
    enable_batching: bool = Field(default=False)

# When running embedded
# mlx-manager serve  → Parent app binds to 127.0.0.1:10242
#                       MLX Server mounted at /v1 (no separate port)

# When running standalone (for testing)
# uvicorn mlx_server.main:app --port 10242  → Uses config values
```

**Warning signs:**
- Logs show "Binding to 127.0.0.1:10242" when embedded (should not happen)
- Port conflicts between parent and sub-app
- Sub-app unreachable at configured port

### Pitfall 4: OpenAPI Schema Conflicts

**What goes wrong:** Both apps define routes with same paths, causing OpenAPI schema conflicts or route shadowing.

**Why it happens:** Parent app routes and sub-app routes may overlap (e.g., both define `/health`).

**How to avoid:**
```python
# Use clear path prefixes
# Parent app: /api/profiles, /api/models, /api/settings
# Sub-app: /v1/chat/completions, /v1/models, /v1/embeddings

app.include_router(profiles_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.mount("/v1", mlx_server_app)

# Health checks at different paths
@app.get("/health")  # Manager health
async def manager_health():
    return {"status": "healthy", "app": "manager"}

@mlx_server_app.get("/health")  # MLX Server health (becomes /v1/health)
async def mlx_health():
    return {"status": "healthy", "app": "mlx_server"}

# OpenAPI docs
# Manager: http://localhost:10242/docs
# MLX Server (if exposed): http://localhost:10242/v1/docs
```

**Warning signs:**
- Routes return unexpected responses (wrong handler executed)
- OpenAPI docs show duplicate operation IDs
- 404 errors on routes that should exist

### Pitfall 5: Legacy Code References Remain

**What goes wrong:** After removing subprocess code, orphaned imports or references cause runtime errors.

**Why it happens:** Subprocess management code was deeply integrated (server_manager called from routers, health checks, launchd service).

**How to avoid:**
```bash
# Comprehensive search for legacy references
grep -r "server_manager" backend/mlx_manager --exclude-dir=__pycache__
grep -r "mlx-openai-server" backend/mlx_manager --exclude-dir=__pycache__
grep -r "command_builder" backend/mlx_manager --exclude-dir=__pycache__
grep -r "parser_options" backend/mlx_manager --exclude-dir=__pycache__

# Files to delete
backend/mlx_manager/services/server_manager.py
backend/mlx_manager/utils/command_builder.py
backend/mlx_manager/utils/parser_options.py

# Files to update (remove imports/calls)
backend/mlx_manager/routers/chat.py        # Remove proxy logic
backend/mlx_manager/routers/servers.py     # Remove start/stop endpoints
backend/mlx_manager/services/health_checker.py  # Remove subprocess checks
backend/mlx_manager/main.py                # Remove server_manager.cleanup()
```

**Testing checklist:**
- [ ] All tests pass after removal
- [ ] No import errors on startup
- [ ] No references to deleted modules in error logs
- [ ] Frontend chat works with embedded `/v1` endpoint

**Source:** Common refactoring practice, validated through static analysis tools

## Code Examples

Verified patterns from official sources and existing codebase:

### Mounting MLX Server in Main App
```python
# Source: mlx_manager/main.py + FastAPI docs
from contextlib import asynccontextmanager
from fastapi import FastAPI
from mlx_manager.mlx_server.main import app as mlx_server_app

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()  # Manager database
    await cleanup_stale_instances()
    await health_checker.start()

    yield

    # Shutdown
    await health_checker.stop()
    # MLX Server cleanup happens in its own lifespan

app = FastAPI(
    title="MLX Model Manager",
    version=__version__,
    lifespan=lifespan,
)

# Include Manager routers
app.include_router(profiles_router)
app.include_router(models_router)
app.include_router(settings_router)

# Mount MLX Server at /v1
app.mount("/v1", mlx_server_app)
```

### Settings Driving Embedded Server Configuration
```python
# Source: Existing settings.py + model pool manager
# In settings router (already exists from Phase 11)
@router.put("/api/settings/pool")
async def update_pool_config(
    data: ServerConfigUpdate,
    session: AsyncSession = Depends(get_db),
):
    """Update model pool configuration and apply to embedded server."""
    # Update database
    config = await session.get(ServerConfig, 1)
    if data.memory_limit_value:
        config.memory_limit_value = data.memory_limit_value
    if data.eviction_policy:
        config.eviction_policy = data.eviction_policy
    await session.commit()

    # Apply to embedded server's model pool
    from mlx_manager.mlx_server.models.pool import model_pool
    if model_pool:
        model_pool.update_limits(
            max_memory_gb=config.memory_limit_value,
            eviction_policy=config.eviction_policy
        )

    return {"success": True}

# In model pool manager (add update method)
class ModelPoolManager:
    def update_limits(self, max_memory_gb: float, eviction_policy: str):
        """Update pool limits without restart."""
        old_limit = self.max_memory_gb
        self.max_memory_gb = max_memory_gb
        self.eviction_policy = eviction_policy

        # Evict models if new limit is lower
        if max_memory_gb < old_limit:
            self._evict_to_fit(max_memory_gb)
```

### Chat UI Direct Endpoint Call
```python
# Source: Simplified from existing chat.py
# BEFORE: Proxy pattern (DELETE THIS)
@router.post("/api/chat/completions")
async def chat_completions(request: ChatRequest):
    profile = await db.get(ServerProfile, request.profile_id)
    server_url = f"http://{profile.host}:{profile.port}/v1/chat/completions"
    async with httpx.AsyncClient() as client:
        response = await client.stream("POST", server_url, json=body)
        # ... proxy response

# AFTER: Direct call (OPTION 1 - Keep wrapper endpoint)
@router.post("/api/chat/completions")
async def chat_completions(request: ChatRequest):
    # Transform to MLX Server request format
    from mlx_manager.mlx_server.schemas.openai import ChatCompletionRequest
    mlx_request = ChatCompletionRequest(
        model=request.model,
        messages=request.messages,
        stream=True
    )

    # Call embedded endpoint
    from mlx_manager.mlx_server.api.v1.chat import chat_completions as mlx_chat
    return await mlx_chat(mlx_request)

# AFTER: No wrapper (OPTION 2 - Frontend calls /v1 directly)
# Delete /api/chat/completions endpoint entirely
# Frontend changes:
#   POST /api/chat/completions  →  POST /v1/chat/completions
```

### Audit Log Integration
```python
# Source: MLX Server already has audit logging
# No changes needed - audit logs automatically populate when using embedded server

# Audit logs stored in separate database
# ~/.mlx-manager/mlx-server.db (audit logs)
# ~/.mlx-manager/mlx-manager.db (profiles, settings)

# Query audit logs (new endpoint to add)
@router.get("/api/audit/logs")
async def get_audit_logs(
    limit: int = 100,
    session: AsyncSession = Depends(get_mlx_server_db),
):
    """Get recent audit logs from embedded MLX Server."""
    from mlx_manager.mlx_server.models.audit import AuditLog
    result = await session.execute(
        select(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
```

### Legacy Code Removal Script
```bash
# Source: Standard refactoring practice
#!/bin/bash
# Remove legacy subprocess management code

# Delete files
rm backend/mlx_manager/services/server_manager.py
rm backend/mlx_manager/utils/command_builder.py
rm backend/mlx_manager/utils/parser_options.py

# Remove from __init__.py imports
sed -i '' '/server_manager/d' backend/mlx_manager/services/__init__.py

# Verify no references remain
echo "Checking for remaining references..."
grep -r "server_manager" backend/mlx_manager --exclude-dir=__pycache__ || echo "✓ No server_manager references"
grep -r "command_builder" backend/mlx_manager --exclude-dir=__pycache__ || echo "✓ No command_builder references"
grep -r "mlx-openai-server" backend/mlx_manager --exclude-dir=__pycache__ || echo "✓ No mlx-openai-server references"

echo "Legacy code removal complete"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Subprocess management | Embedded FastAPI sub-app | 2024+ | Single process, shared resources, faster startup |
| Separate port per service | Path-based routing | FastAPI 0.65+ | No port conflicts, simpler deployment |
| Environment-only config | Database-backed settings | Modern apps | Dynamic config without restart |
| HTTP proxy pattern | Direct function calls | When embedding | No network overhead, simpler error handling |
| Manual ASGI composition | `app.mount()` | FastAPI 0.60+ | Automatic path prefixing, lifespan coordination |

**Deprecated/outdated:**
- **Subprocess server management:** Modern approach is embedding or containerization
- **Fixed configuration:** Database-backed settings allow dynamic updates
- **Proxy endpoints:** Direct calls or path-based routing eliminate HTTP hop
- **Per-service ports:** Path prefixing (`/v1`, `/api`) on single port is cleaner

## Open Questions

Things that couldn't be fully resolved:

1. **Model pool reconfiguration without restart**
   - What we know: Model pool can update `max_memory_gb` and evict models
   - What's unclear: Can eviction policy change without disrupting in-flight requests?
   - Recommendation: Implement graceful policy transition (finish current requests with old policy, apply new policy to queued requests)

2. **Settings update propagation delay**
   - What we know: Settings stored in database, embedded server can read on-demand
   - What's unclear: Should server poll DB every N seconds or use direct function calls?
   - Recommendation: Start with direct function calls (simpler), add polling if decoupling needed later

3. **Profile management UI relevance**
   - What we know: Profiles (host/port/model) were for managing multiple subprocess servers
   - What's unclear: Are profiles still relevant with embedded server, or should they be deprecated?
   - Recommendation: Keep profiles for backward compatibility initially, evaluate removal in future phase

4. **Audit log database consolidation**
   - What we know: MLX Server uses separate `mlx-server.db` for audit logs
   - What's unclear: Should audit logs move to main `mlx-manager.db` for unified querying?
   - Recommendation: Keep separate initially (cleaner separation of concerns), consolidate if querying across both becomes common

## Sources

### Primary (HIGH confidence)
- [FastAPI Sub Applications Documentation](https://fastapi.tiangolo.com/advanced/sub-applications/) - Official mounting guide
- [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/) - Resource initialization patterns
- [FastAPI State Management](https://sqlpey.com/python/fastapi-state-management-app-vs-request-state/) - `app.state` vs `request.state`
- [SQLAlchemy Async Engine](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) - Connection pooling
- Existing codebase:
  - `backend/mlx_manager/main.py` - Parent app structure
  - `backend/mlx_manager/mlx_server/main.py` - Sub-app structure
  - `backend/mlx_manager/routers/settings.py` - Settings endpoints (Phase 11)
  - `backend/mlx_manager/mlx_server/database.py` - Audit database setup

### Secondary (MEDIUM confidence)
- [FastAPI Database Connection Pooling](https://github.com/fastapi/fastapi/discussions/9097) - Community discussion on shared pools
- [Building High-Performance API with Connection Pool](https://medium.com/@balakrishna0106/building-a-high-performance-api-with-fastapi-and-postgresql-connection-pool-fbdee86326e4) - Connection pooling patterns
- [FastAPI Dependency Injection in Lifespan](https://github.com/fastapi/fastapi/discussions/11742) - Shared state patterns
- [FastAPI Bigger Applications Guide](https://fastapi.tiangolo.com/tutorial/bigger-applications/) - Multi-file organization

### Tertiary (LOW confidence - architectural decisions, not code patterns)
- General subprocess-to-embedded migration patterns - No authoritative 2026 documentation found; approach inferred from FastAPI patterns
- Model pool dynamic reconfiguration - Pattern exists but implementation details vary by use case

## Metadata

**Confidence breakdown:**
- FastAPI mounting patterns: HIGH - Official documentation, widely used pattern
- Database session sharing: HIGH - SQLAlchemy best practices, tested in production
- Settings integration: HIGH - Already implemented in Phase 11, just needs wiring
- Legacy code removal: MEDIUM - Straightforward but requires careful testing for orphaned references
- Dynamic configuration update: MEDIUM - Pattern exists but needs testing under load

**Research date:** 2026-01-31
**Valid until:** ~60 days (FastAPI patterns are stable, codebase changes monthly)

**Recommended next steps for planner:**
1. Mount MLX Server at `/v1` prefix in main.py
2. Update Chat UI to call `/v1/chat/completions` instead of proxy endpoint
3. Wire settings endpoints to update embedded server's model pool
4. Delete server_manager.py, command_builder.py, parser_options.py
5. Remove all references to mlx-openai-server subprocess
6. Add endpoint to query audit logs from embedded server's database
7. Update tests to work with embedded server (no subprocess spawning)
8. Verify model pool settings changes take effect without restart
