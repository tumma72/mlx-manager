---
phase: 09-continuous-batching
plan: 01
subsystem: inference
tags: [continuous-batching, priority-queue, dataclass, asyncio, heapq]

# Dependency graph
requires:
  - phase: 08-multi-model-support
    provides: MLX server foundation with model pool and inference services
provides:
  - RequestStatus enum for request lifecycle tracking
  - Priority enum for request prioritization
  - BatchRequest dataclass for request state management
  - PriorityQueueWithAging for fair scheduling with starvation prevention
affects: [09-02, 09-03, 09-04, scheduler, continuous-batching]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - heapq-based priority queue with dataclass ordering
    - asyncio.Lock for async-safe operations
    - time-based priority aging for fairness

key-files:
  created:
    - backend/mlx_manager/mlx_server/services/batching/types.py
    - backend/mlx_manager/mlx_server/services/batching/request.py
    - backend/mlx_manager/mlx_server/services/batching/priority_queue.py
    - backend/tests/mlx_server/batching/test_types.py
    - backend/tests/mlx_server/batching/test_priority_queue.py
  modified:
    - backend/mlx_manager/mlx_server/services/batching/__init__.py

key-decisions:
  - "Priority as IntEnum with HIGH=0, NORMAL=1, LOW=2 (lower numeric = higher priority)"
  - "Aging rate 0.1: LOW becomes NORMAL after 10s, HIGH after 20s"
  - "QueueEntry dataclass with order=True for heapq comparison"
  - "entry_count as FIFO tie-breaker for same-priority requests"

patterns-established:
  - "BatchRequest state transitions via mark_*() methods"
  - "Priority aging formula: effective = base - (wait_time * rate)"
  - "Async lock pattern for queue thread safety"

# Metrics
duration: 3min
completed: 2026-01-28
---

# Phase 09 Plan 01: Foundation Types and Priority Queue Summary

**RequestStatus/Priority enums, BatchRequest dataclass, and PriorityQueueWithAging with heapq-based scheduling and starvation prevention via time-based priority aging**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-28T19:23:02Z
- **Completed:** 2026-01-28T19:26:20Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- RequestStatus enum with 5 lifecycle states (WAITING -> PREFILLING -> RUNNING -> COMPLETED/CANCELLED)
- Priority enum with explicit numeric ordering (HIGH=0 < NORMAL=1 < LOW=2)
- BatchRequest dataclass with state management, timing, and streaming support
- PriorityQueueWithAging preventing starvation via time-based priority promotion

## Task Commits

Each task was committed atomically:

1. **Task 1: Create batching types and request dataclass** - `f8d8c5e` (feat)
2. **Task 2: Implement priority queue with aging** - `7f0cad4` (feat)
3. **Task 3: Add unit tests for types and priority queue** - `8600c75` (test)

## Files Created/Modified
- `backend/mlx_manager/mlx_server/services/batching/types.py` - RequestStatus and Priority enums
- `backend/mlx_manager/mlx_server/services/batching/request.py` - BatchRequest dataclass with lifecycle management
- `backend/mlx_manager/mlx_server/services/batching/priority_queue.py` - PriorityQueueWithAging with heapq and aging
- `backend/mlx_manager/mlx_server/services/batching/__init__.py` - Module exports
- `backend/tests/mlx_server/batching/test_types.py` - 15 tests for types and BatchRequest
- `backend/tests/mlx_server/batching/test_priority_queue.py` - 12 tests for priority queue

## Decisions Made
- **Priority numeric values:** HIGH=0, NORMAL=1, LOW=2 - lower value = higher priority for natural heapq ordering
- **Aging rate 0.1:** Provides reasonable starvation prevention (LOW becomes HIGH after 20s) without being too aggressive
- **QueueEntry as dataclass(order=True):** Enables direct heapq comparison without custom __lt__ methods
- **entry_count tie-breaker:** Guarantees FIFO ordering for requests with identical effective priority

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Foundation types ready for scheduler implementation (09-02)
- BatchRequest integrates with PriorityQueueWithAging via base_priority property
- 27 tests provide regression coverage for core batching infrastructure
- Note: Pre-existing test failures in test_block_manager.py (from 09-02) are unrelated to this plan

---
*Phase: 09-continuous-batching*
*Completed: 2026-01-28*
