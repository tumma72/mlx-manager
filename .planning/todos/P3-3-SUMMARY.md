---
phase: "adhoc"
plan: "P3-3"
subsystem: "mlx-server-config-hot-reload"
tags: ["mlx-server", "config", "SIGHUP", "hot-reload", "admin-api", "signal-handling"]
depends_on: ["P3-2"]
provides: ["config-hot-reload-via-sighup", "admin-reload-config-endpoint", "immutable-settings-detection"]
affects: ["mlx-server-lifespan", "mlx-server-config", "mlx-server-admin-api"]
tech-stack:
  added: []
  patterns: ["loop.add_signal_handler for async-safe signal handling", "lru_cache.cache_clear for settings invalidation", "model_dump diff for change detection", "frozenset for immutable field declaration"]
key-files:
  created:
    - backend/tests/mlx_server/test_config_hot_reload.py
  modified:
    - backend/mlx_manager/mlx_server/config.py
    - backend/mlx_manager/mlx_server/main.py
    - backend/mlx_manager/mlx_server/api/v1/admin.py
decisions:
  - "IMMUTABLE_SETTINGS frozenset declares fields that need a restart: host, port, database_path"
  - "reload_settings() is a shared helper so SIGHUP handler and admin endpoint share identical logic"
  - "loop.add_signal_handler() used over signal.signal() for async event-loop safety"
  - "try/except (NotImplementedError, AttributeError) skips SIGHUP gracefully on Windows"
  - "loop.remove_signal_handler(SIGHUP) called on lifespan shutdown to avoid dangling handler"
  - "Admin endpoint returns structured JSON with changes dict and warnings list"
  - "lru_cache.cache_clear() called inside reload_settings(); callers get a fresh settings object immediately"
metrics:
  duration: "~62 min"
  completed: "2026-03-04"
---

# Phase Adhoc Plan P3-3: Config Hot-Reload via SIGHUP Summary

**One-liner:** SIGHUP signal handler + `POST /v1/admin/reload-config` both call a shared `reload_settings()` helper that clears the `lru_cache`, diffs old vs new values, and warns on immutable fields (host, port, database_path) that need a restart.

## What Was Built

### `IMMUTABLE_SETTINGS` constant (`config.py`)

```python
IMMUTABLE_SETTINGS: frozenset[str] = frozenset({"database_path", "host", "port"})
```

Centralised declaration of which settings require a server restart.

### `reload_settings()` helper (`config.py`)

```python
def reload_settings() -> dict[str, Any]:
    old_settings = get_settings()
    old_values = old_settings.model_dump()

    get_settings.cache_clear()
    new_settings = get_settings()
    new_values = new_settings.model_dump()

    changes = {}
    warnings = []
    for key, old_val in old_values.items():
        new_val = new_values.get(key)
        if old_val != new_val:
            changes[key] = {"old": old_val, "new": new_val}
            if key in IMMUTABLE_SETTINGS:
                warnings.append(f"{key} changed ... requires a server restart to take effect")

    return {"changes": changes, "warnings": warnings}
```

Called by both the SIGHUP callback and the admin endpoint — identical logic, no duplication.

### SIGHUP handler (`main.py`)

Registered during lifespan startup via `loop.add_signal_handler(signal.SIGHUP, handle_sighup)`:

```python
def handle_sighup() -> None:
    logger.info("SIGHUP received, reloading configuration...")
    result = reload_settings()
    changes, warnings = result["changes"], result["warnings"]
    if changes:
        for name, diff in changes.items():
            logger.info("Config reloaded: %s changed from %r to %r", name, diff["old"], diff["new"])
    else:
        logger.info("Config reloaded: no changes detected")
    for warning in warnings:
        logger.warning("Config hot-reload: %s", warning)
```

Platform guard:

```python
try:
    loop.add_signal_handler(signal.SIGHUP, handle_sighup)
except (NotImplementedError, AttributeError):
    logger.debug("SIGHUP not available on this platform; skipping signal handler")
```

On shutdown, `loop.remove_signal_handler(signal.SIGHUP)` removes the handler cleanly.

### Admin reload endpoint (`api/v1/admin.py`)

```
POST /v1/admin/reload-config
```

Protected by the existing `verify_admin_token` dependency on the admin router.

```python
class ReloadConfigResponse(BaseModel):
    reloaded: bool
    changes: dict[str, Any]
    warnings: list[str]
```

Example response when `audit_retention_days` changed and `port` was also edited:

```json
{
  "reloaded": true,
  "changes": {
    "audit_retention_days": {"old": 30, "new": 90},
    "port": {"old": 10242, "new": 9999}
  },
  "warnings": [
    "port changed from 10242 to 9999 but requires a server restart to take effect"
  ]
}
```

### Tests (`test_config_hot_reload.py`)

17 new unit tests across 3 classes:

**`TestReloadSettings` (5 tests):**
- `test_reload_clears_cache` — verifies `cache_clear()` is invoked
- `test_reload_returns_no_changes_when_env_unchanged` — empty changes dict when nothing differs
- `test_reload_detects_mutable_setting_change` — mutable field change reported without warning
- `test_reload_warns_on_immutable_setting_change` — warning generated when `port` changes
- `test_reload_warns_on_all_immutable_settings` — all three immutable fields produce three warnings
- `test_immutable_settings_constant` — IMMUTABLE_SETTINGS contains the expected three fields

**`TestSIGHUPHandler` (4 tests):**
- `test_sighup_handler_registered_on_unix` — verifies SIGHUP registered via `loop.add_signal_handler`
- `test_sighup_registration_skipped_when_not_implemented` — no crash when platform raises `NotImplementedError`
- `test_sighup_handler_calls_reload_settings` — captured callback delegates to `reload_settings()`
- `test_sighup_handler_logs_changes` — callback logs changes at INFO, warnings at WARNING

**`TestAdminReloadConfig` (7 tests):**
- `test_reload_config_returns_reloaded_true`
- `test_reload_config_returns_empty_when_no_changes`
- `test_reload_config_reports_changed_settings`
- `test_reload_config_reports_immutable_warnings`
- `test_reload_config_calls_reload_settings`
- `test_reload_config_multiple_warnings`
- `test_reload_config_response_model`

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Shared `reload_settings()` helper | SIGHUP handler and admin endpoint share identical diff/warn logic — no duplication |
| `loop.add_signal_handler()` not `signal.signal()` | Async-safe; callbacks run on the event loop thread, not a raw OS signal handler thread |
| `(NotImplementedError, AttributeError)` guard | `NotImplementedError` on Windows; `AttributeError` covers edge cases where `signal.SIGHUP` is missing |
| `loop.remove_signal_handler()` on shutdown | Prevents stale handler from firing after pool cleanup |
| `frozenset` for IMMUTABLE_SETTINGS | O(1) membership test; immutable data structure matches the semantics |
| Structured response model | Enables programmatic inspection of what changed; easier to parse than plain text |

## Verification

```
# New hot-reload tests
cd backend && python -m pytest tests/mlx_server/test_config_hot_reload.py -v
# 17 passed

# Existing admin + main tests unaffected
cd backend && python -m pytest tests/mlx_server/test_admin.py tests/mlx_server/test_mlx_server_main.py -q
# 56 passed

# Full MLX server suite
cd backend && python -m pytest tests/mlx_server/ -x -q
# 1983 passed

# Ruff lint (2 pre-existing E501 in export_audit_logs, unchanged)
ruff check mlx_manager/mlx_server/config.py mlx_manager/mlx_server/main.py mlx_manager/mlx_server/api/v1/admin.py
# Only pre-existing errors

# mypy (5 pre-existing errors in middleware files, unchanged)
mypy mlx_manager/mlx_server/config.py mlx_manager/mlx_server/main.py mlx_manager/mlx_server/api/v1/admin.py
# Pre-existing errors only
```

## Deviations from Plan

None — plan executed exactly as written.

## Commit

- `bfc9785`: feat(P3-3): config hot-reload via SIGHUP and admin endpoint
