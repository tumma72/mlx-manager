---
phase: 13-mlx-server-integration
plan: 02
subsystem: backend/services
tags: [cleanup, refactor, embedded-server]
dependency-graph:
  requires: [13-01]
  provides: [legacy-code-removed, embedded-server-status-api]
  affects: [13-03, 13-04, 13-05]
tech-stack:
  added: []
  patterns: [embedded-server-model, model-pool-integration]
key-files:
  created: []
  modified:
    - backend/mlx_manager/routers/servers.py
    - backend/mlx_manager/models.py
    - backend/mlx_manager/main.py
    - backend/mlx_manager/services/__init__.py
    - backend/mlx_manager/services/health_checker.py
    - backend/mlx_manager/services/launchd.py
    - backend/mlx_manager/utils/__init__.py
    - backend/mlx_manager/utils/fuzzy_matcher.py
    - backend/mlx_manager/routers/models.py
    - backend/mlx_manager/routers/system.py
  deleted:
    - backend/mlx_manager/services/server_manager.py
    - backend/mlx_manager/utils/command_builder.py
    - backend/mlx_manager/services/parser_options.py
decisions: []
metrics:
  duration: 10m 23s
  completed: 2026-01-31
---

# Phase 13 Plan 02: Remove Legacy Subprocess Management Summary

Removed all mlx-openai-server subprocess management code and simplified the codebase for the embedded MLX Server model.

## Commits

| Commit | Description |
|--------|-------------|
| 78dc389 | chore(13-02): complete cleanup of legacy subprocess management |
| b5ad419 | refactor(13-02): simplify servers router for embedded MLX Server |

## What Changed

### Deleted Files
- `server_manager.py` - Subprocess lifecycle management for mlx-openai-server
- `command_builder.py` - CLI command building for mlx-openai-server
- `parser_options.py` - Parser discovery from mlx-openai-server

### Models
- **RunningInstance** model removed - No longer needed for subprocess PID tracking
- Added comment indicating manual cleanup of running_instances table

### Services
- **health_checker.py** - Rewritten to monitor model pool instead of subprocesses
- **launchd.py** - Updated to use `mlx-manager serve` instead of mlx-openai-server

### Routers
- **servers.py** - Completely rewritten for embedded server:
  - New `EmbeddedServerStatus` response model
  - New `LoadedModelInfo` for model details
  - New `/models`, `/health`, `/memory` endpoints
  - Legacy `/start`, `/stop`, `/restart` return informative messages
- **models.py** - Parser options endpoint returns empty (deprecated)
- **system.py** - Parser options endpoint returns empty (deprecated)

### Utils
- **fuzzy_matcher.py** - Added stub `get_parser_options()` returning empty lists
- **__init__.py** files updated to remove deleted imports

## Deviations from Plan

None - plan executed as written.

## Test Impact

Tests in `test_main.py` and `conftest.py` fail because they reference deleted modules:
- `mock_find_mlx_openai_server` fixture patches deleted `command_builder`
- `mock_server_manager` fixture patches deleted `server_manager`
- Tests for `cleanup_stale_instances()` reference deleted function

These failures are expected and will be addressed in Plan 05 (Test Updates).

## Next Phase Readiness

**Ready for Plan 03** (Chat UI Integration):
- Servers router now provides embedded server status
- Model pool status available via new endpoints
- Legacy subprocess code removed

**Blockers:** None

## Key Decisions Made

1. **Stub parser options** - Parser options endpoints return empty lists rather than failing, maintaining API compatibility
2. **Legacy endpoints preserved** - start/stop/restart endpoints kept with informative messages for backward compatibility
3. **Model pool integration** - Servers router now queries model pool directly for status
