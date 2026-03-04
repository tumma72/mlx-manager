---
phase: "adhoc"
plan: "P3-2"
subsystem: "mlx-server-audit-rotation"
tags: ["mlx-server", "audit-log", "database", "cleanup", "scheduling", "config"]
depends_on: []
provides: ["audit-log-time-based-cleanup-scheduled", "audit-log-size-based-rotation", "audit-cleanup-background-task"]
affects: ["mlx-server-lifespan", "mlx-server-database", "audit-log-retention"]
tech-stack:
  added: []
  patterns: ["asyncio.create_task for background scheduling", "CancelledError graceful shutdown pattern", "os.path.getsize for DB size gate", "batch delete with VACUUM for disk reclaim"]
key-files:
  created: []
  modified:
    - backend/mlx_manager/mlx_server/config.py
    - backend/mlx_manager/mlx_server/database.py
    - backend/mlx_manager/mlx_server/main.py
    - backend/tests/mlx_server/test_mlx_server_database.py
decisions:
  - "cleanup_by_size() is a separate public function so it can be independently tested and patched"
  - "VACUUM only runs after actual deletes to avoid slow no-op on large DBs"
  - "Batch size of 1000 records balances memory use vs round-trip overhead"
  - "cleanup_old_logs() chains to cleanup_by_size() so callers get combined result"
  - "Initial cleanup at startup before background task creation avoids first-interval wait"
  - "audit_cleanup_task.cancel() + try/except CancelledError prevents blocking shutdown"
  - "Background task sleep comes AFTER cleanup (not before) so first run is immediate on loop restart"
metrics:
  duration: "~10 min"
  completed: "2026-03-04"
---

# Phase Adhoc Plan P3-2: Audit Log Rotation Scheduling + Size-Based Rotation Summary

**One-liner:** Scheduled audit log cleanup via `asyncio.create_task` loop with size-based rotation using `os.path.getsize` + batch DELETE + VACUUM, controlled by two new `MLXServerSettings` fields.

## What Was Built

Wired up the existing but uncalled `cleanup_old_logs()` function into a periodic background task, and added a companion `cleanup_by_size()` function for size-based rotation.

### Config Fields Added (`config.py`)

```python
audit_max_mb: int = Field(
    default=100,
    ge=1,
    le=10000,
    description="Maximum audit log database size in MB before oldest records are purged",
)
audit_cleanup_interval_minutes: int = Field(
    default=60,
    ge=1,
    le=1440,
    description="How often to run audit log cleanup in minutes",
)
```

Both use the `MLX_SERVER_` env prefix. Defaults mean cleanup runs hourly with a 100 MB size cap — non-aggressive but active.

### `cleanup_by_size()` (`database.py`)

New public async function:

1. Gets DB file path from `settings.get_database_path()`
2. Returns 0 immediately if file does not exist
3. Compares `os.path.getsize()` against `audit_max_mb * 1024 * 1024`
4. Returns 0 immediately if under limit (no DB operations)
5. If over limit: deletes oldest 1000 records by `timestamp ASC` in a loop until under limit
6. After all deletes: runs `VACUUM` to reclaim disk space
7. Returns total deleted record count

### `cleanup_old_logs()` update (`database.py`)

After time-based deletion, now calls `cleanup_by_size()` and returns the combined total of time-deleted + size-deleted records.

### Lifespan Integration (`main.py`)

Three changes to the `lifespan()` handler:

1. **Initial cleanup at startup** — runs `cleanup_old_logs()` once before the background task starts; errors are caught and logged as warnings (non-fatal)
2. **Background task creation** — `asyncio.create_task(_audit_cleanup_loop())` after DB init
3. **Graceful shutdown** — `audit_cleanup_task.cancel()` + `await audit_cleanup_task` with `CancelledError` catch ensures the task doesn't block shutdown

The `_audit_cleanup_loop()` coroutine:

```python
async def _audit_cleanup_loop() -> None:
    interval = mlx_server_settings.audit_cleanup_interval_minutes * 60
    while True:
        try:
            await cleanup_old_logs()
        except Exception as exc:
            logger.warning("Audit cleanup failed: %s", exc)
        await asyncio.sleep(interval)
```

Errors are caught and logged; the loop continues rather than dying on transient DB failures.

### Tests Added (`test_mlx_server_database.py`)

13 new tests across 4 new test classes:

**`TestCleanupBySize` (4 tests):**
- `test_cleanup_by_size_no_op_when_under_limit` — returns 0 when file under limit
- `test_cleanup_by_size_no_op_when_db_missing` — returns 0 when file does not exist
- `test_cleanup_by_size_deletes_when_over_limit` — deletes records when mocked size exceeds limit
- `test_cleanup_by_size_runs_vacuum_after_delete` — VACUUM executes without error after deletes

**`TestCleanupOldLogsCallsSize` (1 test):**
- `test_cleanup_old_logs_calls_cleanup_by_size` — verifies `cleanup_by_size` is called via mock

**`TestConfigSettings` (6 tests):**
- Default values for both new fields
- Validation rejects `audit_max_mb=0` and `audit_max_mb=10001`
- Validation rejects `audit_cleanup_interval_minutes=0` and `audit_cleanup_interval_minutes=1441`

**`TestAuditCleanupLoop` (2 tests):**
- `test_audit_cleanup_loop_calls_cleanup` — loop calls `cleanup_old_logs` on each iteration
- `test_audit_cleanup_loop_handles_exceptions` — loop continues after cleanup raises `RuntimeError`

Also updated 2 existing `TestCleanupOldLogs` tests to add `audit_max_mb=100` and a non-existent `get_database_path` return so `cleanup_by_size()` becomes a no-op in those tests.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| `cleanup_by_size()` as separate public function | Independently testable; `cleanup_old_logs()` chains it |
| VACUUM only after actual deletes | VACUUM on a large DB is slow; skip when nothing was removed |
| Batch size 1000 | Keeps each DB transaction bounded; avoids loading all IDs into memory |
| Sleep AFTER cleanup in loop | First interval runs cleanup; then waits — matches standard polling pattern |
| Initial startup cleanup | Handles stale data from long-offline servers without waiting 60 min |
| Non-fatal task cancellation | `CancelledError` is expected during shutdown; suppress it cleanly |

## Verification

```
# Database tests
cd backend && python -m pytest tests/mlx_server/test_mlx_server_database.py -v
# 24 passed

# Full MLX server suite
cd backend && python -m pytest tests/mlx_server/ -x -q
# 1966 passed

# Ruff lint + format
ruff check && ruff format --check
# All checks passed

# mypy (pre-existing errors in middleware unchanged)
mypy mlx_manager/mlx_server/database.py mlx_manager/mlx_server/config.py mlx_manager/mlx_server/main.py
# 5 pre-existing errors in middleware files (unchanged from before this task)
```

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

**Minor test maintenance:**
- Updated 2 existing `TestCleanupOldLogs` tests to mock `audit_max_mb` and `get_database_path` after `cleanup_old_logs()` now delegates to `cleanup_by_size()`. This is necessary test maintenance when chaining functions, not a plan deviation.

## Commit

- `11121d2`: feat(P3-2): audit log rotation scheduling and size-based rotation
