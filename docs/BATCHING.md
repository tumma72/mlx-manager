# Continuous Batching for MLX Inference

This document describes the continuous batching system implemented in MLX Manager for improved inference throughput.

## Overview

### What is Continuous Batching?

Continuous batching (also called iteration-level scheduling) is a technique for processing multiple LLM inference requests simultaneously. Unlike static batching which waits for all requests in a batch to complete before accepting new ones, continuous batching:

- Adds new requests at token generation boundaries (iteration-level)
- Removes completed requests immediately, freeing slots
- Maximizes GPU utilization by keeping the batch full

### Why Does It Improve Throughput?

Traditional sequential processing:
```
Request 1: [generate tokens...........] done
Request 2:                              [generate tokens...........] done
Request 3:                                                          [generate tokens...........] done
```

Continuous batching:
```
Request 1: [generate tokens...........] done
Request 2: [generate tokens..................] done
Request 3:      [generate tokens...............] done
                 ^-- joins at token boundary
```

Key benefits:
- **Higher GPU utilization**: GPU processes multiple sequences concurrently
- **Reduced memory waste**: Paged KV cache allocates memory on-demand
- **Lower latency under load**: Requests start immediately rather than waiting
- **Better throughput**: 2-4x improvement typical (vLLM-MLX achieved 3.4x on M4 Max)

### Performance Target

Based on vLLM-MLX benchmarks:
- **Baseline**: 328 tokens/second (sequential)
- **Batched**: 1,112 tokens/second (3.4x improvement)
- **Target**: 2-4x improvement over sequential baseline

## Architecture

```
                    +-------------------+
                    |   API Endpoint    |
                    | (chat, completions)|
                    +--------+----------+
                             |
                             v
+------------+     +-------------------+     +-------------------+
|  Request   | --> |  Priority Queue   | --> |    Scheduler      |
|  Creation  |     |   (with aging)    |     | (iteration-level) |
+------------+     +-------------------+     +--------+----------+
                                                      |
                   +----------------------------------+
                   |                                  |
                   v                                  v
          +----------------+               +-------------------+
          | Block Manager  |               | Batch Inference   |
          | (paged KV)     |               |    Engine         |
          +-------+--------+               +--------+----------+
                  |                                 |
                  v                                 v
          +----------------+               +-------------------+
          | Prefix Cache   |               |   MLX Model       |
          | (hash-based)   |               |  (generation)     |
          +----------------+               +-------------------+
```

## Components

### 1. PriorityQueueWithAging

**File**: `batching/priority_queue.py`

Request ordering with starvation prevention.

**Features**:
- Three priority levels: HIGH (0), NORMAL (1), LOW (2)
- Lower numeric value = higher priority
- Aging mechanism: +0.1 effective priority per second
- FIFO ordering for same effective priority (entry_count tie-breaker)

**Example**:
```python
queue = PriorityQueueWithAging()
await queue.put(BatchRequest(..., priority=Priority.LOW))
# After 10s, LOW becomes NORMAL; after 20s, becomes HIGH
request = await queue.get()  # Gets highest effective priority
```

### 2. PagedBlockManager

**File**: `batching/block_manager.py`

Fixed-size block allocation for KV cache memory.

**Key constants**:
- `BLOCK_SIZE = 32`: Tokens per block
- Default pool: 1024 blocks (~32K tokens capacity)

**Features**:
- O(1) allocation via stack-based free list
- Reference counting for block sharing
- LRU eviction when pool exhausted
- ~4% internal fragmentation (vs 60-80% with pre-allocation)

**Example**:
```python
manager = PagedBlockManager(num_blocks=1024)
block_id = manager.allocate()  # Get a free block
manager.increment_ref(block_id)  # Share the block
manager.release(block_id)  # Decrement ref, free if 0
```

### 3. PrefixCache

**File**: `batching/prefix_cache.py`

Hash-based prefix sharing for KV block reuse.

**Features**:
- Computes block hashes from token sequences
- Hash chaining for position context (same tokens at different positions = different hash)
- Shares KV cache blocks across requests with common prefixes
- Reduces computation for repeated system prompts

**Example**:
```python
cache = PrefixCache(model_id="model", block_manager=manager)
# Lookup cached blocks for prompt
cached_blocks = cache.lookup(prompt_tokens)
# After generation, cache the computed blocks
cache.insert(prompt_tokens, block_table)
```

### 4. ContinuousBatchingScheduler

**File**: `batching/scheduler.py`

Core iteration-level scheduling loop.

**Responsibilities**:
- Maintains `running` (generating) and `waiting` (queued) request sets
- Fills batch from waiting queue up to `max_batch_size`
- Executes generation step for all running requests
- Removes completed requests immediately
- Adaptive timing: waits longer when idle to accumulate requests

**Parameters**:
- `max_batch_size`: Maximum concurrent requests (default: 8)
- `idle_wait_ms`: Wait when no requests (default: 50ms)
- `load_wait_ms`: Wait between steps under load (default: 5ms)

**Example**:
```python
scheduler = ContinuousBatchingScheduler(
    model_id="model",
    block_manager=manager,
    max_batch_size=8,
)
scheduler.set_model(model, tokenizer, adapter)
await scheduler.start()

# Submit request and stream tokens
async for token in scheduler.submit(request):
    yield token

await scheduler.stop()
```

### 5. BatchInferenceEngine

**File**: `batching/batch_inference.py`

MLX generation with thread affinity.

**Critical**: Uses Queue-based threading pattern for MLX Metal context.
MLX Metal operations have thread affinity requirements - all generation
must happen in a dedicated thread that owns the Metal context.

**Note**: mlx-lm doesn't support true batched generation yet (Issue #548).
Currently generates sequentially within a single thread.

**Example**:
```python
engine = BatchInferenceEngine(
    model=model,
    tokenizer=tokenizer,
    adapter=adapter,
    prefix_cache=cache,
)
results = await engine.generate_batch_step(requests, sampler)
# results: {request_id: (token_text, token_id, is_stop)}
```

### 6. SchedulerManager

**File**: `batching/scheduler_manager.py`

Per-model scheduler lifecycle management.

**Pattern**: Module-level singleton with init/get/reset functions.

**Features**:
- Lazily creates schedulers per model_id
- Graceful shutdown of all schedulers
- Fallback detection for when batching is disabled

**Example**:
```python
# At startup
await init_scheduler_manager(model_manager)

# During request handling
manager = get_scheduler_manager()
scheduler = await manager.get_or_create_scheduler(model_id)

# At shutdown
await manager.shutdown()
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_SERVER_ENABLE_BATCHING` | `false` | Enable continuous batching |
| `MLX_SERVER_BATCH_MAX_BATCH_SIZE` | `8` | Max concurrent requests per step |
| `MLX_SERVER_BATCH_BLOCK_POOL_SIZE` | `1024` | Number of KV cache blocks |

### Config Object

```python
from mlx_manager.mlx_server.config import get_config

config = get_config()
config.enable_batching  # bool
config.batch_max_batch_size  # int
config.batch_block_pool_size  # int
```

### Feature Flag

Batching is disabled by default (`enable_batching=False`). When disabled:
- Direct inference is used (existing behavior)
- Scheduler manager returns None
- API endpoints fall back to synchronous generation

To enable:
```bash
export MLX_SERVER_ENABLE_BATCHING=true
```

## Priority System

### Priority Levels

| Level | Value | Use Case |
|-------|-------|----------|
| HIGH | 0 | Premium tier, system requests |
| NORMAL | 1 | Default for standard requests |
| LOW | 2 | Batch endpoints, background tasks |

### Priority Determination

1. **Endpoint-based**: `/v1/batch/*` endpoints get LOW priority
2. **Default**: All other endpoints get NORMAL priority
3. **Future**: API key tier could influence priority

### Aging Mechanism

Prevents request starvation:
- Effective priority = base_priority - (age_seconds * aging_rate)
- Default aging_rate: 0.1/second
- LOW priority becomes NORMAL after ~10 seconds
- LOW becomes HIGH after ~20 seconds

## Request Lifecycle

```
WAITING --> PREFILLING --> RUNNING --> COMPLETED
    |           |             |           |
    |           v             v           |
    +----> [CANCELLED] <------+           |
                                          |
                                          v
                                    [Response sent]
```

1. **WAITING**: Request in priority queue
2. **PREFILLING**: Allocating blocks, processing prompt
3. **RUNNING**: Active generation, producing tokens
4. **COMPLETED**: All tokens generated or stop token hit
5. **CANCELLED**: Request cancelled by client or timeout

## Limitations

### Current Limitations

1. **Text-only batching**: Vision models fall back to direct inference
2. **Sequential generation**: mlx-lm doesn't support true batched generation (Issue #548)
3. **Per-model batching**: No cross-model batches (each model has its own scheduler)
4. **No dynamic batch sizing**: Fixed max_batch_size regardless of request sizes

### Known Issues

1. **Memory pressure**: Large batch sizes can exhaust Metal memory
2. **Thread affinity**: MLX Metal requires dedicated thread (handled internally)
3. **Prefix cache invalidation**: Not yet implemented for model switches

## Troubleshooting

### High Memory Usage

**Symptom**: MemoryError or system slowdown

**Solutions**:
1. Reduce `batch_block_pool_size` (e.g., 512 instead of 1024)
2. Reduce `batch_max_batch_size` (e.g., 4 instead of 8)
3. Use smaller model quantization (4-bit instead of 8-bit)

### Low Throughput

**Symptom**: Throughput < 1.5x baseline

**Checks**:
1. Verify batching is enabled: `MLX_SERVER_ENABLE_BATCHING=true`
2. Check batch size is reasonable: 4-16 typically optimal
3. Ensure multiple concurrent requests during test
4. Profile to check if IO-bound vs compute-bound

### Request Starvation

**Symptom**: Low-priority requests never complete

**Checks**:
1. Verify aging is working: requests should age by 0.1/sec
2. Check if high-priority requests dominating
3. Consider increasing aging_rate

### Generation Errors

**Symptom**: Requests fail with generation errors

**Checks**:
1. Check MLX thread affinity: errors indicate threading issues
2. Verify model is loaded correctly
3. Check for CUDA/Metal context errors in logs

## Benchmarking

### Running Benchmarks

```bash
# Using the benchmark module
cd backend
python -m mlx_manager.mlx_server.services.batching.benchmark \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --max-tokens 100
```

### Interpreting Results

```
BenchmarkResult(batched):
  Requests: 15
  Total tokens: 1500
  Time: 13.47s
  Throughput: 111.4 tok/s
  Latency (avg): 898.0ms
  Latency (p50): 890.0ms
  Latency (p99): 1200.0ms

Speedup: 2.5x
Result: MEETS TARGET (>= 2x improvement)
```

**Key metrics**:
- **Throughput (tok/s)**: Primary success metric
- **Speedup**: Batched / Single throughput ratio
- **P99 latency**: Worst-case latency for SLA

## Future Improvements

### Short-term

1. **Native batch KV cache**: When mlx-lm adds batched generation support (Issue #548)
2. **Vision model batching**: Extend scheduler to handle mlx-vlm
3. **Prefix cache warming**: Pre-populate cache with common prompts

### Medium-term

1. **Dynamic batch sizing**: Adjust batch size based on request lengths
2. **Speculative decoding**: Use smaller model for draft tokens
3. **Quantized KV cache**: Reduce memory per token

### Long-term

1. **Multi-model batching**: Share GPU across models
2. **Distributed inference**: Split large models across devices
3. **Request routing**: Load balance across multiple instances

## References

- [vLLM-MLX](https://github.com/vllm-project/vllm-mlx): Inspiration for batching architecture
- [PagedAttention](https://arxiv.org/abs/2309.06180): Memory-efficient attention mechanism
- [Continuous Batching](https://www.anyscale.com/blog/continuous-batching-llm-inference): Overview of technique
- [mlx-lm Issue #548](https://github.com/ml-explore/mlx-examples/issues/548): Batched generation support

## Appendix: Data Structures

### BatchRequest

```python
@dataclass
class BatchRequest:
    request_id: str
    prompt: str
    prompt_tokens: list[int]
    max_tokens: int
    priority: Priority = Priority.NORMAL
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    block_table: BlockTable | None = None
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
```

### BlockTable

```python
@dataclass
class BlockTable:
    request_id: str
    logical_to_physical: dict[int, int] = field(default_factory=dict)
    num_tokens: int = 0

    def get_physical_blocks(self) -> list[int]:
        """Return ordered list of physical block IDs."""
```

### KVBlock

```python
@dataclass
class KVBlock:
    block_id: int
    ref_count: int = 0
    tokens: list[int] = field(default_factory=list)
    is_full: bool = False
```
