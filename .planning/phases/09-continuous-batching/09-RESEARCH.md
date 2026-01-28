# Phase 9: Continuous Batching & Paged KV Cache - Research

**Researched:** 2026-01-28
**Domain:** High-throughput LLM inference with memory-efficient batching
**Confidence:** MEDIUM

## Summary

This phase implements continuous batching with paged KV cache to achieve 2-4x throughput improvement over the current single-request inference. The research covers three core components: (1) iteration-level scheduling for continuous batching, (2) paged KV cache with fixed-size blocks, and (3) priority queue with aging-based starvation prevention.

The existing codebase (Phase 7-8) provides a solid foundation with model pool management, adapter registry, and queue-based threading for MLX Metal thread affinity. The main challenge is that **mlx-lm does not natively support paged attention** - it uses RotatingKVCache for single-request inference. Therefore, we must implement custom KV cache management at a higher level, managing block allocation and prefix sharing while still using mlx-lm's stream_generate for token generation.

The vLLM-MLX project demonstrates that 3.4x throughput improvement is achievable on Apple Silicon with continuous batching. Their approach uses the native MLX caching but adds a scheduler layer on top. Given that mlx-lm v0.30.5 has batch KV cache support in development (Issue #548), our implementation should be forward-compatible.

**Primary recommendation:** Implement iteration-level scheduling with a block-based memory manager that tracks logical-to-physical block mappings, using hash-based prefix matching for cache sharing. Process multiple sequences per token generation step by managing separate KV cache states per request.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mlx-lm | 0.30.5 | Token generation | Apple-maintained, prompt cache support |
| asyncio.PriorityQueue | stdlib | Request scheduling | Coroutine-safe, heap-based |
| heapq | stdlib | Priority management | O(log n) operations |
| dataclasses | stdlib | Request/Block state | Clean state management |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| logfire | latest | Observability | Token/batch metrics |
| psutil | latest | Memory monitoring | System memory pressure |

### What We Build (Custom)
| Component | Purpose | Why Custom |
|-----------|---------|------------|
| BatchScheduler | Iteration-level scheduling | MLX-specific thread affinity |
| PagedBlockManager | Block allocation/eviction | mlx-lm lacks paged attention |
| PrefixCache | Hash-based prefix sharing | Model-scoped prefix reuse |
| PriorityQueueWithAging | Starvation prevention | Standard asyncio.PriorityQueue lacks aging |

## Architecture Patterns

### Recommended Project Structure
```
mlx_server/
  services/
    batching/
      __init__.py
      scheduler.py        # ContinuousBatchingScheduler
      block_manager.py    # PagedBlockManager
      prefix_cache.py     # PrefixCache with hash matching
      priority_queue.py   # PriorityQueueWithAging
      request.py          # BatchRequest dataclass
      types.py            # RequestStatus, Priority enums
```

### Pattern 1: Iteration-Level Scheduling Loop

**What:** The scheduler processes one token generation step for all active sequences, then immediately fills freed slots with waiting requests.

**When to use:** Always - this is the core continuous batching pattern.

**Example:**
```python
# Source: vLLM scheduler architecture (blog.vllm.ai)
class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size: int, block_manager: PagedBlockManager):
        self.running: list[BatchRequest] = []
        self.waiting: PriorityQueueWithAging = PriorityQueueWithAging()
        self.max_batch_size = max_batch_size
        self.block_manager = block_manager

    async def scheduling_loop(self):
        """Main iteration-level scheduling loop."""
        while True:
            # 1. Generate next token for all running sequences
            if self.running:
                await self._batch_generate_step()

            # 2. Remove completed sequences (frees slots immediately)
            self._remove_completed()

            # 3. Fill batch with waiting requests
            await self._fill_batch()

            # 4. Yield to event loop if no work
            if not self.running and self.waiting.empty():
                await asyncio.sleep(0.01)  # Idle wait
```

### Pattern 2: Block Table with Logical-to-Physical Mapping

**What:** Each request maintains a block table that maps logical block indices to physical block IDs. Enables non-contiguous KV cache allocation.

**When to use:** For paged KV cache management.

**Example:**
```python
# Source: PagedAttention paper (arxiv.org/abs/2309.06180)
@dataclass
class BlockTable:
    """Maps logical block indices to physical block IDs."""
    request_id: str
    logical_to_physical: dict[int, int] = field(default_factory=dict)
    num_tokens: int = 0

    def allocate_block(self, block_manager: 'PagedBlockManager') -> int:
        """Get or allocate physical block for current logical index."""
        logical_idx = self.num_tokens // BLOCK_SIZE
        if logical_idx not in self.logical_to_physical:
            physical_id = block_manager.allocate()
            self.logical_to_physical[logical_idx] = physical_id
        return self.logical_to_physical[logical_idx]
```

### Pattern 3: Hash-Based Prefix Matching

**What:** Use content-based hashing to identify shareable prefix blocks. Hash includes block content and position.

**When to use:** For prefix caching lookup.

**Example:**
```python
# Source: vLLM prefix caching design (docs.vllm.ai)
def compute_block_hash(
    token_ids: list[int],
    block_start: int,
    block_size: int,
    prev_hash: int = 0
) -> int:
    """Compute hash for a block of tokens with position context."""
    block_tokens = tuple(token_ids[block_start:block_start + block_size])
    # Include prev_hash to chain blocks (position matters)
    return hash((prev_hash, block_tokens))
```

### Pattern 4: Priority Queue with Aging

**What:** Gradually increase effective priority of waiting requests to prevent starvation. Lower numeric priority = higher urgency.

**When to use:** For priority-based request scheduling.

**Example:**
```python
# Source: xappsoftware.com Python priority queue with aging
@dataclass(order=True)
class PrioritizedRequest:
    effective_priority: float = field(compare=True)
    entry_time: float = field(compare=False)
    request: BatchRequest = field(compare=False)

    def age(self, current_time: float, aging_rate: float = 0.1):
        """Decrease priority (higher urgency) based on wait time."""
        wait_time = current_time - self.entry_time
        aging_bonus = wait_time * aging_rate
        self.effective_priority = self.request.base_priority - aging_bonus
```

### Anti-Patterns to Avoid

- **Global KV cache for all requests:** Each request needs isolated KV state to prevent cross-contamination (mlx-omni-server bug).
- **Blocking on single request completion:** Static batching - defeats the purpose.
- **Pre-allocating max_tokens memory per request:** Wastes 60-80% of memory. Allocate blocks on-demand.
- **Using RotatingKVCache for multi-request:** It's designed for single continuous generation, not batch.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Token generation | Custom attention | mlx-lm stream_generate | MLX-optimized, model-specific |
| Chat templates | String formatting | tokenizer.apply_chat_template | Model-specific special tokens |
| Stop token detection | Simple string match | Model adapter get_stop_tokens | Model family variations |
| Memory usage tracking | Manual calculation | mx.get_active_memory() | MLX-accurate reporting |
| Priority comparison | Custom comparators | heapq with tuples | O(log n) guaranteed |

**Key insight:** mlx-lm handles the complex attention computation. Our job is managing WHICH requests get generated and managing the cache state BETWEEN requests - not implementing attention ourselves.

## Common Pitfalls

### Pitfall 1: MLX Thread Affinity Violation
**What goes wrong:** Calling MLX operations from multiple threads causes undefined behavior or crashes.
**Why it happens:** MLX Metal has thread affinity - operations must stay on the thread that created the context.
**How to avoid:** Use a single dedicated generation thread with Queue for communication (current inference.py pattern).
**Warning signs:** Sporadic crashes, memory corruption, inconsistent outputs.

### Pitfall 2: KV Cache State Contamination
**What goes wrong:** Tokens from one request appear in another request's output.
**Why it happens:** Sharing cache state across requests without proper isolation.
**How to avoid:** Each request gets its own KV cache state. Never share mutable cache between requests.
**Warning signs:** Model outputs contain fragments from other requests.

### Pitfall 3: Memory Fragmentation Without Paging
**What goes wrong:** Out of memory despite having enough total free memory.
**Why it happens:** Requesting contiguous blocks for max_tokens leaves gaps.
**How to avoid:** Fixed-size blocks (32 tokens as per CONTEXT.md) with block tables.
**Warning signs:** 503 errors when memory usage shows available capacity.

### Pitfall 4: Priority Starvation
**What goes wrong:** Low-priority requests never complete.
**Why it happens:** High-priority requests continuously preempt them.
**How to avoid:** Aging mechanism - waiting time increases effective priority.
**Warning signs:** Timeout errors only on low-priority requests.

### Pitfall 5: Prefix Cache Over-Eviction
**What goes wrong:** Cache miss rate increases despite warm cache.
**Why it happens:** Evicting prefix blocks currently in use or about to be reused.
**How to avoid:** Reference counting on blocks; only evict ref_count=0 blocks.
**Warning signs:** Prefix cache hit rate drops under load.

### Pitfall 6: Request Join Mid-Generation
**What goes wrong:** Attempting to add request to batch during token generation corrupts state.
**Why it happens:** Not respecting iteration boundaries.
**How to avoid:** New requests wait for current step to complete (CONTEXT.md decision).
**Warning signs:** Assertion errors, inconsistent token counts.

## Code Examples

### Request Lifecycle Management

```python
# Source: Pattern synthesis from vLLM + project requirements
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import AsyncGenerator
import asyncio
import time

class RequestStatus(Enum):
    WAITING = auto()    # In priority queue
    PREFILLING = auto() # Computing initial KV cache
    RUNNING = auto()    # Generating tokens
    COMPLETED = auto()  # Done, pending cleanup
    CANCELLED = auto()  # Aborted

class Priority(Enum):
    HIGH = 0    # Special API keys
    NORMAL = 1  # User-facing
    LOW = 2     # Batch/script

@dataclass
class BatchRequest:
    request_id: str
    model_id: str
    prompt_tokens: list[int]
    max_tokens: int
    priority: Priority

    # State
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    block_table: 'BlockTable' = field(default=None)

    # Streaming
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Timing for aging
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None

    @property
    def base_priority(self) -> int:
        return self.priority.value

    def is_complete(self) -> bool:
        return (
            len(self.generated_tokens) >= self.max_tokens or
            self.status in (RequestStatus.COMPLETED, RequestStatus.CANCELLED)
        )
```

### Block Manager with LRU Eviction

```python
# Source: Pattern from PagedAttention + CONTEXT.md decisions
BLOCK_SIZE = 32  # tokens per block (from CONTEXT.md)
MEMORY_PRESSURE_THRESHOLD = 0.85  # from CONTEXT.md

@dataclass
class KVBlock:
    block_id: int
    ref_count: int = 0
    last_used: float = field(default_factory=time.time)
    is_prefix_cached: bool = False
    prefix_hash: int | None = None

class PagedBlockManager:
    def __init__(self, num_blocks: int):
        self.blocks = {i: KVBlock(block_id=i) for i in range(num_blocks)}
        self.free_blocks: list[int] = list(range(num_blocks))
        self.prefix_cache: dict[int, int] = {}  # hash -> block_id

    def allocate(self) -> int:
        """Allocate a free block, evicting if necessary."""
        if not self.free_blocks:
            self._evict_lru_prefix_blocks()
        if not self.free_blocks:
            raise MemoryError("No free KV cache blocks")

        block_id = self.free_blocks.pop()
        self.blocks[block_id].ref_count = 1
        self.blocks[block_id].last_used = time.time()
        return block_id

    def release(self, block_id: int):
        """Decrement ref count, free if zero."""
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count <= 0 and not block.is_prefix_cached:
            self.free_blocks.append(block_id)

    def _evict_lru_prefix_blocks(self):
        """Evict prefix cache blocks first (CONTEXT.md decision)."""
        # Find prefix blocks with ref_count=0, sorted by last_used
        evictable = [
            b for b in self.blocks.values()
            if b.is_prefix_cached and b.ref_count == 0
        ]
        evictable.sort(key=lambda b: b.last_used)

        for block in evictable[:max(1, len(evictable) // 4)]:
            if block.prefix_hash in self.prefix_cache:
                del self.prefix_cache[block.prefix_hash]
            block.is_prefix_cached = False
            self.free_blocks.append(block.block_id)
```

### Priority Queue with Aging

```python
# Source: xappsoftware.com pattern + asyncio adaptation
import heapq
from dataclasses import dataclass, field
from typing import Optional

@dataclass(order=True)
class QueueEntry:
    effective_priority: float
    entry_count: int = field(compare=True)  # Tie-breaker for FIFO within priority
    request: BatchRequest = field(compare=False)
    entry_time: float = field(compare=False, default_factory=time.time)

class PriorityQueueWithAging:
    def __init__(self, aging_rate: float = 0.1):
        """
        Args:
            aging_rate: Priority decrease per second of waiting.
                        With rate=0.1, a LOW (2) request becomes NORMAL (1)
                        after 10 seconds, HIGH (0) after 20 seconds.
        """
        self._heap: list[QueueEntry] = []
        self._entry_counter = 0
        self._aging_rate = aging_rate
        self._lock = asyncio.Lock()

    async def put(self, request: BatchRequest):
        async with self._lock:
            entry = QueueEntry(
                effective_priority=float(request.base_priority),
                entry_count=self._entry_counter,
                request=request,
            )
            self._entry_counter += 1
            heapq.heappush(self._heap, entry)

    async def get(self) -> BatchRequest:
        async with self._lock:
            self._update_priorities()
            if not self._heap:
                raise IndexError("Queue is empty")
            entry = heapq.heappop(self._heap)
            return entry.request

    def _update_priorities(self):
        """Recompute effective priorities based on wait time."""
        current_time = time.time()
        for entry in self._heap:
            wait_time = current_time - entry.entry_time
            entry.effective_priority = entry.request.base_priority - (wait_time * self._aging_rate)
        heapq.heapify(self._heap)  # Restore heap property

    def empty(self) -> bool:
        return len(self._heap) == 0

    def qsize(self) -> int:
        return len(self._heap)
```

### Continuous Batching Scheduler Core

```python
# Source: Synthesis of vLLM patterns + MLX thread affinity requirements
class ContinuousBatchingScheduler:
    def __init__(
        self,
        block_manager: PagedBlockManager,
        max_batch_size: int = 8,
        idle_wait_ms: float = 50.0,
        load_wait_ms: float = 5.0,
    ):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.idle_wait_ms = idle_wait_ms
        self.load_wait_ms = load_wait_ms

        self.running: list[BatchRequest] = []
        self.waiting = PriorityQueueWithAging()

        self._generation_thread: threading.Thread | None = None
        self._token_queues: dict[str, asyncio.Queue] = {}
        self._shutdown = False

    async def submit(self, request: BatchRequest) -> AsyncGenerator[str, None]:
        """Submit request and return token stream."""
        self._token_queues[request.request_id] = request.output_queue
        await self.waiting.put(request)

        # Yield tokens as they arrive
        try:
            while True:
                token = await request.output_queue.get()
                if token is None:  # End signal
                    break
                yield token
        finally:
            del self._token_queues[request.request_id]

    async def _scheduling_loop(self):
        """Main loop - runs in async context."""
        while not self._shutdown:
            # Adaptive wait: longer when idle to accumulate requests
            wait_ms = self.idle_wait_ms if not self.running else self.load_wait_ms

            # Fill batch from waiting queue
            while len(self.running) < self.max_batch_size and not self.waiting.empty():
                request = await self.waiting.get()
                request.status = RequestStatus.PREFILLING
                request.started_at = time.time()
                # Allocate initial blocks for prompt
                request.block_table = self._allocate_prompt_blocks(request)
                self.running.append(request)

            if self.running:
                # Generate one token for all running requests
                await self._batch_step()

                # Remove completed requests
                completed = [r for r in self.running if r.is_complete()]
                for request in completed:
                    request.status = RequestStatus.COMPLETED
                    await request.output_queue.put(None)  # Signal completion
                    self._release_blocks(request)
                    self.running.remove(request)
            else:
                await asyncio.sleep(wait_ms / 1000.0)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Static batching | Continuous/iteration-level batching | ORCA (2022), vLLM (2023) | 2-23x throughput |
| Contiguous KV allocation | Paged KV cache | vLLM PagedAttention (2023) | <4% memory waste vs 60-80% |
| No prefix sharing | Automatic prefix caching | vLLM APC (2024) | 5.8x speedup on shared prefixes |
| Simple FIFO queue | Priority with aging | Common pattern | Prevents starvation |

**Deprecated/outdated:**
- **mx.metal.get_active_memory()**: Replaced by mx.get_active_memory() in MLX 0.30+
- **mlx_lm stream_generate temp/top_p params**: Replaced by make_sampler() API

## Open Questions

### 1. MLX Batch KV Cache API
**What we know:** mlx-lm Issue #548 discusses batch KV cache support. BatchKVCache exists with filter/extend but lacks ability to accept pre-built caches.
**What's unclear:** Timeline for native batch KV cache support; whether we can leverage it.
**Recommendation:** Build our own block management now; monitor mlx-lm for future native support. Design for forward compatibility.

### 2. Optimal Block Pool Sizing
**What we know:** CONTEXT.md allows Claude's discretion on pre-allocate vs dynamic. Apple unified memory reduces transfer overhead.
**What's unclear:** Best initial allocation size for Apple Silicon unified memory.
**Recommendation:** Start with dynamic allocation (allocate on-demand from available memory). Monitor fragmentation. Add pre-allocation if needed.

### 3. Vision Model Batching
**What we know:** Vision models use mlx-vlm with Processor (not Tokenizer). Phase 8 added vision support.
**What's unclear:** Whether vision embeddings can be batched efficiently; prefix caching for image inputs.
**Recommendation:** Start with text-only batching. Add vision batching as future enhancement.

### 4. Multi-Model Batch Scheduling
**What we know:** Model pool supports multiple models. Batching is typically per-model.
**What's unclear:** Whether to allow mixed-model batches or require model homogeneity.
**Recommendation:** Require model homogeneity within a batch (simplest). Different models use separate batch slots.

## Sources

### Primary (HIGH confidence)
- [vLLM Documentation](https://docs.vllm.ai/en/stable/design/paged_attention/) - PagedAttention design
- [vLLM Blog: Anatomy of vLLM](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) - Scheduler architecture
- [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm) - v0.30.5 API, KV cache options
- [Python asyncio.PriorityQueue docs](https://docs.python.org/3/library/asyncio-queue.html) - Standard library

### Secondary (MEDIUM confidence)
- [vLLM-MLX GitHub](https://github.com/waybarrios/vllm-mlx) - MLX batching proof of concept (3.4x speedup verified)
- [PagedAttention Paper (arxiv 2309.06180)](https://arxiv.org/abs/2309.06180) - Original algorithm
- [SGLang RadixAttention Blog](https://lmsys.org/blog/2024-01-17-sglang/) - Prefix caching with radix tree
- [xappsoftware: Priority Queue with Aging](https://www.xappsoftware.com/wordpress/2018/02/05/python-3-priority-queues-with-aging-avoiding-starvation/) - Aging implementation

### Tertiary (LOW confidence)
- [mlx-lm Issue #548](https://github.com/ml-explore/mlx-lm/issues/548) - Batch KV cache plans (unresolved)
- Various Medium articles on continuous batching - General patterns

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - mlx-lm is verified, but batch KV cache is custom
- Architecture patterns: MEDIUM - vLLM patterns proven, MLX adaptation unverified
- Pitfalls: HIGH - Thread affinity, cache isolation well-documented from Phase 7-8

**Research date:** 2026-01-28
**Valid until:** 2026-02-28 (30 days - stable domain but mlx-lm evolving)
