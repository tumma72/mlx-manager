---
phase: "adhoc"
plan: "P3-4"
subsystem: "mlx-server-database"
tags: ["mlx-server", "database", "connection-pooling", "embedded-mode", "sqlalchemy", "sqlite"]
depends_on: ["P3-3"]
provides: ["shared-engine-injection", "embedded-mode-connection-pooling"]
affects: ["mlx-server-database", "mlx-manager-main", "audit-logging"]
tech-stack:
  added: []
  patterns: ["engine injection for embedded mode", "scoped create_all with tables= kwarg", "shared connection pool via global injection"]
key-files:
  created: []
  modified:
    - backend/mlx_manager/mlx_server/database.py
    - backend/mlx_manager/main.py
    - backend/tests/mlx_server/test_mlx_server_database.py
decisions:
  - "set_shared_engine() sets both _shared_engine/_shared_session_factory tracking globals AND _engine/_async_session operational globals — single source of truth"
  - "init_db() passes tables=[AuditLog.__table__] to create_all to avoid touching host-application tables in embedded mode"
  - "session_factory arg is optional: set_shared_engine auto-creates one from the engine when omitted"
  - "reset_for_testing() clears all four globals (_engine, _async_session, _shared_engine, _shared_session_factory) for complete test isolation"
  - "type: ignore[attr-defined] on AuditLog.__table__ suppresses SQLModel mypy false-positive (attr exists at runtime via SQLAlchemy metaclass)"
metrics:
  duration: "~15 min"
  completed: "2026-03-04"
---

# Phase Adhoc Plan P3-4: Connection Pooling for Embedded Mode Summary

**One-liner:** `set_shared_engine(engine, async_session)` in `main.py` injects the manager's SQLAlchemy engine into `mlx_server/database.py` before mounting, eliminating duplicate connections to the same SQLite file in embedded mode.

## What Was Built

### `set_shared_engine()` function (`mlx_server/database.py`)

```python
_shared_engine = None
_shared_session_factory = None

def set_shared_engine(engine, session_factory=None) -> None:
    """Inject a shared engine for embedded mode."""
    global _shared_engine, _shared_session_factory, _engine, _async_session
    _shared_engine = engine
    _engine = engine
    if session_factory is not None:
        _shared_session_factory = session_factory
        _async_session = session_factory
    else:
        _shared_session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        _async_session = _shared_session_factory
```

Sets `_engine` and `_async_session` directly so all downstream calls (`_get_engine()`, `_get_session_factory()`, `init_db()`, `get_session()`) immediately use the shared pool without any additional logic changes.

### `reset_for_testing()` update

```python
def reset_for_testing() -> None:
    global _engine, _async_session, _shared_engine, _shared_session_factory
    _engine = None
    _async_session = None
    _shared_engine = None
    _shared_session_factory = None
```

All four globals cleared so tests cannot inadvertently inherit shared-engine state from a previous test.

### `init_db()` scoped to audit table only

```python
async def init_db() -> None:
    from mlx_manager.mlx_server.models.audit import AuditLog  # noqa: F401

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(
            SQLModel.metadata.create_all,
            tables=[AuditLog.__table__],  # type: ignore[attr-defined]
        )
```

Previously `create_all()` was called without a `tables=` filter, meaning it would attempt to create ALL SQLModel-registered tables (including host-application tables) when given a shared engine. Scoping to `[AuditLog.__table__]` makes the call idempotent and non-invasive in both standalone and embedded modes.

### Engine injection in `main.py`

```python
from mlx_manager.database import (
    async_session,
    engine,
    ...
)
from mlx_manager.mlx_server.database import set_shared_engine

set_shared_engine(engine, async_session)

mlx_server_app = create_mlx_server_app(embedded=True)
app.mount("/v1", mlx_server_app, name="mlx_server")
```

The injection happens at module load time (before `create_mlx_server_app`), so `init_db()` inside the MLX Server lifespan already sees the shared engine.

### Tests (`test_mlx_server_database.py`)

5 new tests across 2 new classes:

**`TestResetForTesting` (1 new test):**
- `test_reset_clears_shared_engine_state` — verifies all four globals are cleared, including `_shared_engine` and `_shared_session_factory`

**`TestSetSharedEngine` (4 new tests):**
- `test_set_shared_engine_sets_engine_globals` — both tracking and operational globals point to injected objects
- `test_set_shared_engine_without_factory_creates_one` — auto-creates sessionmaker when `session_factory` is omitted
- `test_set_shared_engine_makes_get_engine_return_it` — `_get_engine()` returns the injected engine
- `test_backward_compat_no_shared_engine_creates_own` — standalone mode (no injection) still creates its own real engine

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Assign both `_shared_*` AND `_engine`/`_async_session` in `set_shared_engine()` | `_get_engine()` and `_get_session_factory()` check `_engine`/`_async_session` directly; tracking globals add observability without changing existing read paths |
| `session_factory` arg optional | Callers that only have an engine (e.g. tests) can omit the factory; the function builds a compatible one |
| Scoped `create_all` with `tables=[AuditLog.__table__]` | Prevents MLX Server from re-creating host-application tables when sharing an engine; idempotent on audit_logs in both modes |
| Injection before `create_mlx_server_app()` in main.py | MLX Server lifespan calls `init_db()` on startup; engine must be injected before that runs |
| `type: ignore[attr-defined]` for `__table__` | SQLAlchemy sets `__table__` via metaclass at model creation; mypy's SQLModel stubs don't declare it but it is always present at runtime |

## Verification

```
# New shared-engine tests
cd backend && python -m pytest tests/mlx_server/test_mlx_server_database.py -v
# 29 passed (was 24 before)

# Full MLX server suite
cd backend && python -m pytest tests/mlx_server/ -x -q
# 1988 passed

# Ruff lint
ruff check mlx_manager/mlx_server/database.py mlx_manager/main.py tests/mlx_server/test_mlx_server_database.py
# clean

# mypy
mypy mlx_manager/mlx_server/database.py
# Success: no issues found in 1 source file
```

## Deviations from Plan

None — plan executed exactly as written.

## Commit

- `513a5d0`: feat(P3-4): connection pooling for embedded mode
