---
phase: 09-continuous-batching
plan: 05
subsystem: batching
tags: [mlx, inference, threading, batch-generation, metal-affinity]
dependency_graph:
  requires: [09-03, 09-04]
  provides: [batch-inference-engine, scheduler-mlx-integration]
  affects: [09-06, 09-07]
tech_stack:
  added: []
  patterns: [queue-based-threading, metal-thread-affinity]
key_files:
  created:
    - backend/mlx_manager/mlx_server/services/batching/batch_inference.py
    - backend/tests/mlx_server/batching/test_batch_inference.py
  modified:
    - backend/mlx_manager/mlx_server/services/batching/scheduler.py
    - backend/mlx_manager/mlx_server/services/batching/__init__.py
decisions:
  - id: sequential-generation-mlx-lm
    title: "Sequential generation within thread due to mlx-lm limitation"
    rationale: "mlx-lm doesn't support true batched generation yet (Issue #548)"
  - id: token-id-zero-default
    title: "Use token_id=0 for error/no-token cases"
    rationale: "Provides consistent tuple structure (str, int, bool) for all results"
metrics:
  duration: 5m38s
  completed: 2026-01-28
---

# Phase 09 Plan 05: Batch Inference Engine Summary

BatchInferenceEngine with Queue-based MLX threading pattern for multi-request token generation

## What Changed

### Files Created

1. **backend/mlx_manager/mlx_server/services/batching/batch_inference.py**
   - `BatchInferenceEngine` class for batch token generation
   - Queue-based threading pattern for MLX Metal affinity
   - Stop token detection from model adapter
   - Prefix cache integration (optional)
   - `generate_batch_step()` async wrapper for scheduler
   - `generate_tokens_for_batch()` synchronous core logic

2. **backend/tests/mlx_server/batching/test_batch_inference.py**
   - 22 tests covering engine initialization
   - Tests for request context preparation
   - Tests for batch token generation with mocked MLX
   - Tests for stop token detection
   - Tests for thread safety and Queue-based communication
   - Tests for scheduler integration

### Files Modified

1. **backend/mlx_manager/mlx_server/services/batching/scheduler.py**
   - Added `set_model()` method for runtime model configuration
   - Updated `_batch_step()` to use `BatchInferenceEngine`
   - Added `_inference_engine` and `_prefix_cache` attributes
   - Accepts model/tokenizer/adapter in `__init__`

2. **backend/mlx_manager/mlx_server/services/batching/__init__.py**
   - Added `BatchInferenceEngine` to exports

## Technical Details

### Queue-based Threading Pattern

```python
# Critical pattern for MLX Metal affinity
result_queue: Queue[dict | Exception] = Queue()

def run_generation():
    results = self.generate_tokens_for_batch(requests, sampler)
    result_queue.put(results)

gen_thread = threading.Thread(target=run_generation, daemon=True)
gen_thread.start()

result = await loop.run_in_executor(None, lambda: result_queue.get(timeout=60))
```

This pattern ensures all MLX operations happen in a single dedicated thread that owns the Metal GPU context, while still providing an async interface for the scheduler.

### Sequential Generation Note

Due to mlx-lm Issue #548, true batched generation is not yet supported. The engine generates sequentially within the dedicated thread:
- Still maintains Metal context consistency
- Future: native batch KV cache when mlx-lm supports it

## Decisions Made

| Decision | Context | Outcome |
|----------|---------|---------|
| Sequential in-thread generation | mlx-lm doesn't batch | Sequential but thread-safe |
| token_id=0 for errors | Consistent tuple structure | (text, int, bool) always |
| daemon=True for thread | Don't block process exit | Clean shutdown |
| 60s timeout | Prevent infinite hangs | TimeoutError raised |

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

### All Tests Pass
- 150 batching tests passing (22 new for batch_inference)
- All quality gates pass:
  - `ruff check`: All checks passed
  - `mypy`: No errors in batching module (external dep warning only)

### Key Verified Behaviors
- BatchInferenceEngine stores model/tokenizer/adapter
- Stop tokens extracted from adapter
- Generation runs in dedicated thread
- Results passed via Queue
- Scheduler calls inference engine when model is set
- Stop tokens mark requests as complete

## Commit History

| Hash | Message |
|------|---------|
| 780f64f | feat(09-05): create batch inference engine with MLX threading |
| 8a93f81 | feat(09-05): integrate batch inference into scheduler |
| ca11473 | test(09-05): add comprehensive batch inference engine tests |
| 2eb9441 | chore(09-05): export BatchInferenceEngine and format batching module |

## Next Phase Readiness

**Plan 06 (API Integration):**
- BatchInferenceEngine ready for integration
- Scheduler accepts model via `set_model()`
- Next: Connect chat/completions endpoints to scheduler
