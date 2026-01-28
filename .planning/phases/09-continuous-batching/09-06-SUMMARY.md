---
phase: 09-continuous-batching
plan: 06
subsystem: batching
tags: ["scheduler", "api", "chat", "integration", "singleton"]
dependency-graph:
  requires: ["09-04"]
  provides: ["SchedulerManager singleton", "chat endpoint batching integration", "enable_batching config"]
  affects: ["09-07"]
tech-stack:
  added: []
  patterns: ["singleton with lazy initialization", "feature flag", "graceful fallback"]
key-files:
  created:
    - "backend/mlx_manager/mlx_server/services/batching/scheduler_manager.py"
    - "backend/tests/mlx_server/batching/test_scheduler_manager.py"
  modified:
    - "backend/mlx_manager/mlx_server/services/batching/__init__.py"
    - "backend/mlx_manager/mlx_server/api/v1/chat.py"
    - "backend/mlx_manager/mlx_server/config.py"
    - "backend/mlx_manager/mlx_server/main.py"
decisions:
  - id: "scheduler-singleton"
    choice: "Module-level singleton with init/get/reset functions"
    rationale: "Matches existing patterns (model pool), enables testing via reset"
  - id: "endpoint-priority"
    choice: "/v1/batch/* routes get LOW priority"
    rationale: "System-determined priority per CONTEXT.md, client cannot request priority"
  - id: "feature-disabled-default"
    choice: "enable_batching=False by default"
    rationale: "Safety - batching is experimental until full MLX integration complete"
  - id: "fallback-to-direct"
    choice: "Graceful fallback to direct inference if scheduler unavailable"
    rationale: "Reliability - don't break existing functionality during rollout"
metrics:
  duration: "~4 min"
  completed: "2026-01-28"
---

# Phase 09 Plan 06: API Integration and Scheduler Management Summary

SchedulerManager singleton with per-model scheduler instances, chat endpoint batching integration with OpenAI-format streaming, and priority routing by endpoint.

## Objective Achievement

Connected the continuous batching scheduler to the chat completions endpoint with:
- SchedulerManager for per-model scheduler lifecycle
- Priority determination by endpoint (batch/* = LOW, others = NORMAL)
- Feature flag control via enable_batching setting
- Graceful fallback to direct inference on scheduler failure

## Changes Made

### Task 1: SchedulerManager Singleton (77575b6)

Created `/backend/mlx_manager/mlx_server/services/batching/scheduler_manager.py`:
- `SchedulerManager` class with per-model scheduler lazy initialization
- `get_scheduler(model_id)` - returns existing or creates new scheduler
- `configure_scheduler(model_id, model, tokenizer, adapter)` - placeholder for engine setup
- `get_priority_for_request(api_key, endpoint)` - priority determination logic
- `shutdown()` - graceful shutdown of all schedulers
- Singleton pattern: `init_scheduler_manager()`, `get_scheduler_manager()`, `reset_scheduler_manager()`

Updated `__init__.py` to export new symbols.

### Task 2: Chat Endpoint Integration (6fad8bb)

Updated `/backend/mlx_manager/mlx_server/api/v1/chat.py`:
- Added batching path check in `_handle_text_request()`
- Implemented `_handle_batched_request()` - tokenizes and submits to scheduler
- Implemented `_stream_batched_response()` - SSE format with OpenAI ChatCompletionChunk
- Implemented `_complete_batched_response()` - full response with usage stats
- Graceful fallback to `_handle_direct_request()` on scheduler errors

### Task 3: Lifespan and Config (527cf27)

Updated `/backend/mlx_manager/mlx_server/config.py`:
- `enable_batching: bool = False` - feature flag (disabled by default)
- `batch_block_pool_size: int = 1000` - KV cache blocks per model
- `batch_max_batch_size: int = 8` - max concurrent requests

Updated `/backend/mlx_manager/mlx_server/main.py`:
- Initialize SchedulerManager in lifespan if batching enabled
- Shutdown SchedulerManager gracefully on app shutdown

Created `/backend/tests/mlx_server/batching/test_scheduler_manager.py`:
- 21 tests covering singleton, get_scheduler, priority, shutdown, configure, concurrency

## Verification Results

- **All batching tests pass**: 128 tests for plan-relevant files
- **Ruff lint**: Clean
- **Mypy**: Pre-existing errors in batch_inference.py (from 09-05), files modified in this plan are clean
- **Feature flag works**: `enable_batching` controls activation

## Deviations from Plan

None - plan executed exactly as written.

## Key Patterns Established

1. **Singleton with reset**: Module-level `_scheduler_manager` with `init/get/reset` functions enables testing
2. **Feature flag pattern**: `settings.enable_batching` controls whether batching path is taken
3. **Graceful degradation**: Catch scheduler exceptions and fall back to direct inference
4. **Priority by endpoint**: System determines priority, not client - `/v1/batch/*` gets LOW

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scheduler lifecycle | Lazy init on first request | Avoids startup cost for unused models |
| Priority source | Endpoint path | System-controlled, not client-requestable |
| Default state | Disabled | Safe rollout - existing behavior unchanged |
| Error handling | Fall back to direct | Reliability over optimization |

## Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `scheduler_manager.py` | Created | SchedulerManager singleton |
| `__init__.py` | Modified | Export new symbols |
| `chat.py` | Modified | Batching integration |
| `config.py` | Modified | Batching config flags |
| `main.py` | Modified | Lifespan initialization |
| `test_scheduler_manager.py` | Created | 21 test cases |

## Next Steps

Plan 09-07 will complete the integration by:
1. Wiring BatchInferenceEngine to real MLX generation
2. Full end-to-end test with actual model

## Commit Summary

| Commit | Type | Description |
|--------|------|-------------|
| 77575b6 | feat | SchedulerManager singleton |
| 6fad8bb | feat | Chat endpoint batching integration |
| 527cf27 | feat | Lifespan init and config flags |
