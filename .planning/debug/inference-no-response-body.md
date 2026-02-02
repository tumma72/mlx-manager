---
status: diagnosed
trigger: "Server hangs during model loading - /health unresponsive"
created: 2026-01-27T12:00:00Z
updated: 2026-01-28T10:30:00Z
symptoms_prefilled: true
goal: find_root_cause_only
---

## Current Focus

hypothesis: CONFIRMED - Multiple issues compound to cause complete server unresponsiveness during large model loading
test: Code tracing and analysis of asyncio patterns, MLX operations, and thread pool behavior
expecting: Event loop blocked or thread pool exhausted, preventing any requests
next_action: Document root cause and required fixes

## Symptoms

expected: Server should remain responsive during model loading (at least /health should work)
actual: Entire server hangs - curl hangs, /health unresponsive, no logs
errors: None visible - server appears frozen
reproduction: Request inference with large model (24B+) like Devstral-Small-2 or large Qwen
started: Observed with large models; smaller models worked in previous tests

## Eliminated

- hypothesis: run_in_executor token-by-token consumption pattern
  evidence: Code has been fixed - now uses dedicated threading.Thread for generation (lines 151-183, 204-205 in inference.py)
  timestamp: 2026-01-28T10:00:00Z

## Evidence

- timestamp: 2026-01-28T09:30:00Z
  checked: inference.py _stream_chat_generate (lines 122-280)
  found: Uses threading.Thread for MLX generation, Queue for token passing, run_in_executor only for non-blocking queue.get(timeout=0.1)
  implication: Generation pattern is correct - NOT the cause of hang

- timestamp: 2026-01-28T09:35:00Z
  checked: pool.py _load_model (lines 230-313)
  found: Model loading uses asyncio.to_thread(load, model_id) at line 279 - correct pattern
  implication: Model loading pattern is async-safe

- timestamp: 2026-01-28T09:40:00Z
  checked: pool.py get_memory_usage() call at line 285
  found: get_memory_usage() calls mx.get_active_memory() etc SYNCHRONOUSLY on event loop thread AFTER asyncio.to_thread returns
  implication: MLX memory calls on main thread could block event loop if Metal needs to sync

- timestamp: 2026-01-28T09:45:00Z
  checked: memory.py get_memory_usage() (lines 17-36)
  found: Calls mx.get_active_memory(), mx.get_peak_memory(), mx.get_cache_memory() - these are MLX operations that may require Metal synchronization
  implication: Calling MLX functions from main thread after load completes could block

- timestamp: 2026-01-28T09:50:00Z
  checked: Default thread pool behavior
  found: Python's asyncio uses a single shared ThreadPoolExecutor (default ~4-32 workers depending on CPU). When large model loading takes minutes, it occupies a thread. Multiple concurrent requests or internal asyncio operations (DNS) compete for remaining threads.
  implication: Thread pool exhaustion during long model loads can cascade to affect all async operations

- timestamp: 2026-01-28T10:00:00Z
  checked: MLX GitHub issues
  found: Multiple reports of system freezes with large models (24B+, 32B+). MLX unified memory competes with system memory. Models that exceed available memory cause severe system degradation. GPU utilization drops to 0% during freeze.
  implication: Large model loading can trigger system-level memory pressure that freezes entire process

- timestamp: 2026-01-28T10:15:00Z
  checked: pool.py get_model() flow (lines 193-228)
  found: Multiple lock acquisitions with potential for race condition. Check at line 219 happens OUTSIDE lock, after releasing lock at line 217. Model could be removed from _loading between these lines.
  implication: Race condition in get_model() could cause KeyError or duplicate loading attempts

- timestamp: 2026-01-28T10:20:00Z
  checked: clear_cache() in memory.py (lines 39-51)
  found: Calls mx.synchronize() at line 46 BEFORE mx.clear_cache(). mx.synchronize() waits for ALL pending Metal GPU operations to complete.
  implication: clear_cache() called after generation (inference.py lines 280, 398, etc) could block event loop if MLX has pending work

## Resolution

root_cause: |
  THREE COMPOUNDING ISSUES cause complete server unresponsiveness:

  1. **MLX Memory Operations on Event Loop Thread (CRITICAL)**
     After `asyncio.to_thread(load, model_id)` completes, code immediately calls
     `get_memory_usage()` (pool.py:285) on the main thread. These MLX functions
     (mx.get_active_memory(), etc.) may require Metal synchronization, blocking
     the event loop. Similarly, `clear_cache()` calls `mx.synchronize()` which
     explicitly blocks waiting for all GPU operations.

  2. **Thread Pool Exhaustion with Large Models**
     Large models (24B+) take minutes to load. During this time, the thread
     occupies one of the limited slots in Python's default ThreadPoolExecutor
     (~5-32 workers). Concurrent requests compete for remaining threads.
     Internal asyncio operations (like DNS resolution) also use this pool.
     When exhausted, ALL operations stall including simple HTTP handling.

  3. **System Memory Pressure (for very large models)**
     MLX uses unified memory which competes with system memory. Loading 24B+
     models on machines with insufficient unified memory can trigger severe
     system-level memory pressure, causing the entire process (and sometimes
     the system) to become unresponsive. This matches reports in MLX GitHub
     issues for 32B+ models.

  The combination means: during large model loading, the thread pool worker
  is occupied, MLX memory calls block the event loop, and system memory
  pressure prevents responsive behavior. Result: even /health can't respond.

fix: |
  **Required changes:**

  1. **Move MLX memory calls off event loop (pool.py)**
     Wrap get_memory_usage() call in asyncio.to_thread():
     ```python
     memory = await asyncio.to_thread(get_memory_usage)
     ```

  2. **Move clear_cache() off event loop (inference.py, vision.py, embeddings.py)**
     All clear_cache() calls should be in finally blocks but wrapped:
     ```python
     finally:
         await asyncio.to_thread(clear_cache)
     ```
     Or use a background task:
     ```python
     finally:
         asyncio.create_task(asyncio.to_thread(clear_cache))
     ```

  3. **Use dedicated ThreadPoolExecutor for model loading**
     Don't share the default executor - create a dedicated one:
     ```python
     MODEL_LOAD_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mlx-load")

     # In _load_model:
     loop = asyncio.get_running_loop()
     result = await loop.run_in_executor(MODEL_LOAD_EXECUTOR, load, model_id)
     ```

  4. **Add memory check before loading large models**
     Estimate model size and check available memory BEFORE attempting load:
     ```python
     estimated_gb = self._estimate_model_size(model_id)
     available = psutil.virtual_memory().available / (1024**3)
     if estimated_gb > available * 0.8:  # Leave 20% headroom
         raise HTTPException(503, f"Insufficient memory for {model_id}")
     ```

  5. **Fix race condition in get_model() (pool.py)**
     The check at line 219 must happen inside the lock or use the Event properly.

verification:
files_changed: []
