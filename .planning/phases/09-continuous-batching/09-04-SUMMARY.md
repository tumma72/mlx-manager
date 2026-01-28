---
phase: 09-continuous-batching
plan: 04
subsystem: batching
tags: [scheduler, continuous-batching, async, threading]

dependency-graph:
  requires: ["09-01", "09-02", "09-03"]
  provides: ["ContinuousBatchingScheduler", "iteration-level-scheduling"]
  affects: ["09-05"]

tech-stack:
  added: []
  patterns: ["async-generator-streaming", "queue-based-threading", "adaptive-timing"]

key-files:
  created:
    - backend/mlx_manager/mlx_server/services/batching/scheduler.py
    - backend/tests/mlx_server/batching/test_scheduler.py
  modified:
    - backend/mlx_manager/mlx_server/services/batching/__init__.py
    - backend/mlx_manager/mlx_server/services/batching/request.py

decisions:
  - id: adaptive-timing
    choice: "idle_wait_ms=50.0, load_wait_ms=5.0 defaults"
    rationale: "Longer wait when idle accumulates requests for batching; shorter wait under load for responsiveness"
  - id: memory-error-retry-delay
    choice: "Sleep idle_wait_ms after MemoryError before retrying"
    rationale: "Prevents busy loop when request is too large for available blocks"
  - id: block-table-type-union
    choice: "BlockTable | list[int] | None for backward compatibility"
    rationale: "Supports both new BlockTable system and legacy list[int] format"
  - id: output-queue-none-signal
    choice: "None in output_queue signals completion"
    rationale: "Standard sentinel value pattern for async queue completion"

metrics:
  duration: "~12 min"
  completed: "2026-01-28"
---

# Phase 9 Plan 4: Continuous Batching Scheduler Summary

**One-liner:** Iteration-level scheduler that fills batch from priority queue, runs generation steps, and immediately frees slots on completion.

## What Was Built

### ContinuousBatchingScheduler

Core scheduler implementing continuous batching with iteration-level scheduling:

```python
scheduler = ContinuousBatchingScheduler(
    model_id="test-model",
    block_manager=block_manager,
    max_batch_size=8,
    idle_wait_ms=50.0,
    load_wait_ms=5.0,
)
await scheduler.start()
# ... requests processed ...
await scheduler.stop()
```

Key features:
- **Iteration-level scheduling**: Requests join at step boundaries, not mid-generation
- **True continuous batching**: Completed requests immediately free slots
- **Adaptive timing**: Longer wait when idle (accumulate requests), shorter under load
- **Memory-aware**: Handles block allocation failures gracefully with retry delay
- **Graceful shutdown**: Waits for running requests, cancels waiting ones

### Request Lifecycle

1. `submit()` - Add request to priority queue, yield tokens from output_queue
2. Request picked up at next step boundary
3. `_allocate_prompt_blocks()` - Allocate KV cache blocks for prompt
4. `_batch_step()` - Generate one token for all running requests
5. `_release_blocks()` - Free blocks when request completes
6. `None` sent to output_queue signals completion

### State Management

- `running: list[BatchRequest]` - Currently generating (up to max_batch_size)
- `waiting: PriorityQueueWithAging` - Pending requests
- `_step_lock: asyncio.Lock` - Prevents mid-step modifications

## Key Implementation Details

### Scheduling Loop

```python
async def _scheduling_loop(self):
    while not self._shutdown:
        async with self._step_lock:
            # Fill batch from waiting queue
            while len(self.running) < self.max_batch_size and not self.waiting.empty():
                request = await self.waiting.get()
                request.block_table = self._allocate_prompt_blocks(request)
                self.running.append(request)

            if self.running:
                await self._batch_step()

                # Remove completed (continuous batching)
                for r in [r for r in self.running if r.is_complete]:
                    await r.output_queue.put(None)
                    self._release_blocks(r)
                    self.running.remove(r)
```

### _batch_step Placeholder

Current implementation is a stub that simulates token generation. Full implementation in Plan 05 will:
1. Run prefill for new requests
2. Run decode for all requests using MLX
3. Distribute tokens via output_queue
4. Use queue-based threading for Metal thread affinity

## Verification Results

- **Tests**: 24 scheduler tests + 107 total batching tests passing
- **Lint**: ruff clean
- **Types**: mypy clean (batching module)

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 34c4207 | feat | add ContinuousBatchingScheduler core structure |
| 2fe1d00 | feat | implement scheduling loop with batch generation |
| 08e0391 | test | add comprehensive scheduler unit tests |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Infinite loop on MemoryError**
- **Found during:** Task 3 (tests hanging)
- **Issue:** When block allocation failed, request was re-queued and immediately retried in tight loop
- **Fix:** Added `await asyncio.sleep(idle_wait_ms / 1000.0)` after MemoryError before breaking
- **Files modified:** scheduler.py
- **Commit:** 08e0391

**2. [Rule 1 - Bug] Type mismatch for BlockTable**
- **Found during:** mypy check
- **Issue:** BatchRequest.block_table typed as `list[int] | None` but scheduler assigns `BlockTable`
- **Fix:** Updated type to `BlockTable | list[int] | None` for backward compatibility
- **Files modified:** request.py
- **Commit:** 08e0391

**3. [Rule 1 - Bug] output_queue type didn't allow None**
- **Found during:** mypy check
- **Issue:** `asyncio.Queue[dict[str, Any]]` doesn't accept None completion signal
- **Fix:** Changed to `asyncio.Queue[dict[str, Any] | None]`
- **Files modified:** request.py
- **Commit:** 08e0391

## Next Phase Readiness

Plan 09-05 can now implement:
- Real MLX generation in `_batch_step()`
- Integration with model pool
- Queue-based threading for Metal affinity
- Stop token detection

All prerequisites satisfied:
- Priority queue with aging (09-01)
- Paged block manager (09-02)
- Prefix cache (09-03)
- Scheduler with iteration-level loop (this plan)
