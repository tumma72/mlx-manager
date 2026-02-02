# MLX Multi-Model Server Feasibility Study

**Date:** 2026-01-27
**Status:** Research Complete
**Author:** Claude (synthesized from 4 parallel research agents)

---

## Executive Summary

This feasibility study evaluates building a custom high-performance MLX-optimized model server that:
- Serves multiple models in parallel with dynamic loading
- Supports both OpenAI and Anthropic REST APIs natively
- Achieves 2-4x throughput improvement over current mlx-openai-server
- Leverages Apple Silicon's unified memory architecture

**Verdict: FEASIBLE with HIGH CONFIDENCE**

The MLX ecosystem provides solid foundations (mlx-lm, mlx-vlm, mlx-embeddings), and vLLM-MLX has already demonstrated 3.4x speedup with continuous batching on M4 Max. Key techniques (PagedAttention, continuous batching) are proven and adaptable.

---

## 1. Foundation Libraries Assessment

### 1.1 Required Libraries

| Library | Purpose | Maturity | Verdict |
|---------|---------|----------|---------|
| **mlx-lm** | Text generation | Production (Apple-maintained) | Use |
| **mlx-vlm** | Vision-language models | Production | Use |
| **mlx-embeddings** | Text embeddings | Stable | Use |
| **MLX Core** | Direct Metal access | Production (Apple) | Foundation |

### 1.2 mlx-lm Capabilities

**Core API:**
```python
from mlx_lm import load, generate, stream_generate

# Model loading with caching
model, tokenizer = load("mlx-community/Model-4bit")

# Streaming generation
for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    yield response.text
```

**Key Features:**
- Prompt caching for multi-turn dialogues
- Rotating KV cache (`--max-kv-size`) for memory management
- Custom sampler and logits processor support
- 4-bit/8-bit quantization

**Performance (Apple M-series):**
- ~230 tok/s single stream
- <10s TTFT for 14B dense models
- 24GB Mac supports 8B BF16 or 30B 4-bit MoE

### 1.3 mlx-vlm Capabilities

**Supported Model Families:**
- Qwen2-VL, Qwen2.5-VL (including video)
- LLaVA variants
- Idefics3
- Gemma-3n-E2B (audio + vision)

**API:**
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
output = generate(model, processor, formatted_prompt, images, audio=audio)
```

### 1.4 mlx-embeddings Capabilities

**Supported Architectures:**
- XLM-RoBERTa, BERT, ModernBERT
- Qwen3 embedding models
- SigLIP (vision embeddings)

**Batch Processing:**
```python
from mlx_embeddings.utils import load
model, tokenizer = load("mlx-community/all-MiniLM-L6-v2-4bit")
outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
embeddings = outputs.text_embeds  # normalized
```

### 1.5 Alternatives Considered

| Alternative | Assessment |
|-------------|------------|
| Raw MLX | Maximum control but ~200+ lines for basic Llama, no tokenizer handling |
| vLLM-MLX | Already integrates mlx-lm/mlx-vlm, proven batching |
| Candle (Rust) | Not MLX-optimized, different ecosystem |

**Recommendation:** Build on mlx-lm + mlx-vlm + mlx-embeddings. These provide:
- Apple-maintained code for mlx-lm
- Active community for mlx-vlm
- Proper Metal memory management
- Model family adapters already implemented

---

## 2. Performance Techniques Analysis

### 2.1 PagedAttention (Critical)

**Problem Solved:** Traditional LLM serving wastes 60-80% of KV cache memory due to fragmentation and over-reservation.

**How It Works:**
```
TRADITIONAL (Wastes 60-80% memory)
+------------------------------------------+
| Request 1: [########________] (50% used) |
| Request 2: [######__________] (37% used) |
|            Pre-allocated contiguous      |
+------------------------------------------+

PAGED ATTENTION (< 4% waste)
+------------------------------------------+
| Block Pool: [B0][B1][B2][B3][B4]...      |
| Request 1: Block 0->B7, Block 1->B1      |
| Request 2: Block 0->B3, Block 1->B5      |
|            Dynamic, non-contiguous       |
+------------------------------------------+
```

**Key Components:**
1. Fixed-size blocks (16-32 tokens, ~12.8KB for 13B model)
2. Block table maps logical -> physical blocks
3. Dynamic allocation on-demand
4. Copy-on-write for parallel sampling (55% memory reduction)

**Result:** 2-4x throughput improvement, >90% GPU utilization

### 2.2 Continuous Batching (Essential)

**Problem Solved:** Static batching waits for longest request, wasting GPU cycles.

**How It Works:**
```
STATIC BATCHING (GPU underutilized)
+------------------------------------------+
| R1: [########################]           |
| R2: [############]-> IDLE waiting...     |
| R3: [################]-> IDLE waiting... |
+------------------------------------------+

CONTINUOUS BATCHING (GPU always busy)
+------------------------------------------+
| R1: [########################]           |
| R2: [############]-> R5: [##############]|
| R3: [################]-> R6: [##########]|
+------------------------------------------+
```

**Implementation:**
- Iteration-level scheduling (per token generation step)
- Token-level batching (process one token per sequence per iteration)
- Dynamic replacement (completed sequences immediately freed)

**Result:** Up to 23x throughput vs naive static batching

### 2.3 vLLM-MLX Benchmarks (Proof of Concept)

| Model | Single Request | Batched (5 concurrent) | Speedup |
|-------|---------------|------------------------|---------|
| Qwen3-0.6B-8bit | 328 tok/s | 1,112 tok/s | **3.4x** |
| Llama-3.2-1B-4bit | 299 tok/s | 613 tok/s | **2.0x** |
| Llama-3.2-3B-4bit | 200 tok/s | - | - |

### 2.4 Additional Optimizations

| Technique | Impact | Complexity | MLX Status |
|-----------|--------|------------|------------|
| Prefix Caching | 87% cache hits, 88% faster TTFT | Medium | Supported |
| Chunked Prefill | +50% throughput | Medium | Not yet |
| Speculative Decoding | 1.5-2.8x latency | High | Needs draft model |
| Rotating KV Cache | Memory bounded | Low | Built-in |

---

## 3. Existing Server Analysis

### 3.1 mlx-lm Server (Official Apple)

**Architecture:** Surprisingly minimal - Python's `ThreadingHTTPServer`

**Limitations:**
- Single-threaded generation (requests queue)
- No batching
- No max-kv-size in HTTP server
- "Not recommended for production"

**Lesson:** Good for understanding mlx-lm API, not for production patterns.

### 3.2 mlx-openai-server (FastAPI)

**Architecture:** FastAPI with request queue

```bash
--max-concurrency: 1 (default)
--queue-timeout: 300s
--queue-size: 100
```

**Good Patterns:**
- Request queue with configurable concurrency
- Tool call parsers per model family
- OpenAI-compatible message format

**Limitations:**
- Still fundamentally single-model
- No continuous batching
- No paged KV cache

### 3.3 mlx-omni-server (Dual API)

**Why It's Interesting:** Both OpenAI AND Anthropic API support

```
OpenAI Request -> OpenAIAdapter -> ChatGenerator -> Model
Anthropic Request -> AnthropicMessagesAdapter -> ChatGenerator -> Model
```

**Known Stability Issues:**
- Token caching bug (first token prepended to next request)
- Memory not freed after concurrent testing
- "Insufficient Memory" errors on Mac Studio

**Lesson:** Cache/state isolation between requests is critical.

### 3.4 vLLM-MLX (Production-Grade)

**Architecture:**
```
vLLM API Layer -> MLXPlatform (plugin) -> {mlx-lm, mlx-vlm} -> MLX
```

**Key Differentiators:**
1. True continuous batching (3.4x throughput)
2. Paged KV cache with prefix sharing
3. API key authentication
4. MCP tool calling integration

**This is our primary reference implementation.**

### 3.5 LM Studio mlx-engine (Unified Pattern)

**Innovation:** VisionAddOn pattern

```python
class BaseVisionAddOn:
    """Generates image embeddings for text model input"""
    def generate_embeddings(self, images) -> Embeddings
```

Vision embeddings feed into mlx-lm's `stream_generate()` via `input_embeddings` argument. Text path stays unified.

**Lesson:** Treat vision/audio as modular add-ons to text generation.

---

## 4. Architecture Proposal

### 4.1 High-Level Design

```
+----------------------------------------------------------------------+
|                      MLX UNIFIED SERVER                              |
+----------------------------------------------------------------------+
|  API Layer (FastAPI + uvloop)                                        |
|  +------------------+  +------------------+  +--------------------+  |
|  | /v1/chat/        |  | /v1/messages     |  | /v1/embeddings     |  |
|  | completions      |  | (Anthropic)      |  |                    |  |
|  | (OpenAI)         |  |                  |  |                    |  |
|  +---------+--------+  +---------+--------+  +-----------+--------+  |
|            |                     |                       |           |
|  +---------v---------------------v-----------------------v--------+  |
|  |                    Protocol Translator                         |  |
|  |  OpenAI --+                                     +-- Anthropic  |  |
|  |           +-------> Internal Format <-----------+              |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    Request Scheduler                           |  |
|  |  +-----------+  +-----------+  +-----------+                   |  |
|  |  | Priority  |  | Priority  |  | Priority  |  Continuous       |  |
|  |  |  High     |  |  Normal   |  |   Low     |  Batching         |  |
|  |  +-----------+  +-----------+  +-----------+  Engine           |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    Model Pool Manager                          |  |
|  |  +----------------------------------------------------------+  |  |
|  |  | Hot Models (in unified memory)                           |  |  |
|  |  | +----------+ +----------+ +----------+ +----------+      |  |  |
|  |  | | Model A  | | Model B  | | Model C  | | Embed    |      |  |  |
|  |  | | (LLM)    | | (VLM)    | | (LLM)    | | Model    |      |  |  |
|  |  | +----------+ +----------+ +----------+ +----------+      |  |  |
|  |  +----------------------------------------------------------+  |  |
|  |  LRU Eviction | Memory Pressure Monitor | Hot-Swap Support    |  |
|  +---------------------------+------------------------------------+  |
|                              |                                       |
|  +---------------------------v------------------------------------+  |
|  |                    Inference Engine                            |  |
|  |  +----------------------------------------------------------+ |  |
|  |  |              Paged KV Cache Manager                       | |  |
|  |  |  Block Pool -> Block Tables -> Prefix Sharing             | |  |
|  |  +----------------------------------------------------------+ |  |
|  |  +----------------------------------------------------------+ |  |
|  |  |              Model Adapters                               | |  |
|  |  |  +--------+ +--------+ +--------+ +--------+              | |  |
|  |  |  | Llama  | | Qwen   | |Mistral | | Gemma  | ...          | |  |
|  |  |  |Adapter | |Adapter | |Adapter | |Adapter |              | |  |
|  |  |  +--------+ +--------+ +--------+ +--------+              | |  |
|  |  +----------------------------------------------------------+ |  |
|  |  +----------------------------------------------------------+ |  |
|  |  |              Modality Add-Ons                             | |  |
|  |  |  VisionAddOn (mlx-vlm) | AudioAddOn (future)              | |  |
|  |  +----------------------------------------------------------+ |  |
|  +---------------------------------------------------------------+  |
|                                                                      |
|  +---------------------------------------------------------------+  |
|  |                    Observability Layer (Pydantic LogFire)      |  |
|  |  Request Tracing | LLM Token Metrics | SQLite Spans | Alerts   |  |
|  +---------------------------------------------------------------+  |
+----------------------------------------------------------------------+
```

### 4.2 Core Components

#### Protocol Translator

```python
class ProtocolTranslator:
    def openai_to_internal(self, request: OpenAIRequest) -> InternalRequest:
        return InternalRequest(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream,
            tools=request.tools,
        )

    def anthropic_to_internal(self, request: AnthropicRequest) -> InternalRequest:
        messages = request.messages
        if request.system:
            messages = [{"role": "system", "content": request.system}] + messages
        return InternalRequest(
            model=request.model,
            messages=messages,
            max_tokens=request.max_tokens,  # Required in Anthropic
            temperature=request.temperature,
            stream=request.stream,
        )

    def internal_to_openai_chunk(self, token: str, ...) -> OpenAIChunk: ...
    def internal_to_anthropic_event(self, token: str, ...) -> AnthropicEvent: ...
```

#### Continuous Batching Scheduler

```python
class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size: int = 8, max_tokens_per_batch: int = 4096):
        self.running: List[ActiveRequest] = []
        self.waiting: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.max_batch_size = max_batch_size

    async def scheduling_loop(self):
        while True:
            # 1. Generate next token for all active sequences
            if self.running:
                tokens = await self.batch_generate_step(self.running)
                for request, token in zip(self.running, tokens):
                    await request.emit_token(token)
                    if request.is_complete():
                        self.running.remove(request)
                        await request.finalize()

            # 2. Fill batch with waiting requests
            while len(self.running) < self.max_batch_size:
                if self.waiting.empty():
                    break
                priority, request = await self.waiting.get()
                await request.start_prefill()
                self.running.append(request)

            # 3. Yield to event loop
            await asyncio.sleep(0)
```

#### Model Pool Manager

```python
class ModelPoolManager:
    def __init__(self, max_memory_gb: float = 48.0):
        self.hot_models: Dict[str, LoadedModel] = {}
        self.model_lock = asyncio.Lock()
        self.max_memory = max_memory_gb * 1024 ** 3

    async def get_model(self, model_id: str) -> LoadedModel:
        async with self.model_lock:
            if model_id in self.hot_models:
                self.hot_models[model_id].last_used = time.time()
                return self.hot_models[model_id]

            # Check memory pressure
            while self._current_memory() + self._estimate_model_size(model_id) > self.max_memory:
                await self._evict_lru_model()

            # Load model
            model = await self._load_model(model_id)
            self.hot_models[model_id] = model
            return model

    async def _evict_lru_model(self):
        lru = min(self.hot_models.values(), key=lambda m: m.last_used)
        del self.hot_models[lru.model_id]
        mx.metal.clear_cache()
```

#### Paged KV Cache Manager

```python
class PagedKVCacheManager:
    def __init__(self, block_size: int = 16, num_blocks: int = 1024):
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_tables: Dict[str, List[int]] = {}  # request_id -> blocks
        self.prefix_cache: Dict[int, int] = {}  # hash -> block_id

    def allocate_block(self, request_id: str) -> int:
        if not self.free_blocks:
            raise MemoryError("No free KV cache blocks")
        block = self.free_blocks.pop()
        self.block_tables.setdefault(request_id, []).append(block)
        return block

    def release_request(self, request_id: str):
        blocks = self.block_tables.pop(request_id, [])
        self.free_blocks.extend(blocks)

    def find_cached_prefix(self, token_ids: List[int]) -> Tuple[List[int], int]:
        """Returns (cached_blocks, num_cached_tokens)"""
        prefix_hash = hash(tuple(token_ids[:self.block_size]))
        if prefix_hash in self.prefix_cache:
            # Share existing blocks (copy-on-write)
            ...
```

### 4.3 Model Adapters

Each model family has specific handling for:
- Chat template formatting
- Tool call parsing
- Stop token detection
- Thinking/reasoning tokens

```python
class ModelAdapter(Protocol):
    model_family: str

    def apply_chat_template(
        self,
        tokenizer: Tokenizer,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        enable_thinking: bool = False,
    ) -> str: ...

    def parse_tool_calls(self, response: str) -> List[ToolCall]: ...

    def get_stop_tokens(self, tokenizer: Tokenizer) -> Set[int]: ...

    def handle_thinking_tokens(self, response: str) -> Tuple[str, Optional[str]]:
        """Returns (response, thinking_content)"""
        ...

# Registry
ADAPTERS: Dict[str, ModelAdapter] = {
    "llama": LlamaAdapter(),
    "qwen": QwenAdapter(),
    "mistral": MistralAdapter(),
    "gemma": GemmaAdapter(),
    ...
}

def get_adapter(model_id: str) -> ModelAdapter:
    family = detect_model_family(model_id)
    return ADAPTERS.get(family, DefaultAdapter())
```

---

## 5. Technology Stack Proposal

### 5.1 Core Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **HTTP Server** | FastAPI + uvloop | 2-4x perf vs asyncio, async-native |
| **Inference** | mlx-lm + mlx-vlm + mlx-embeddings | Apple-maintained, MLX-optimized |
| **Tokenization** | HuggingFace Tokenizers (Rust) | 43x faster than pure Python |
| **Streaming** | SSE (Server-Sent Events) | OpenAI-compatible, robust |
| **Validation** | Pydantic v2 (Rust core) | Type-safe, 5-50x faster than v1 |
| **Observability** | Pydantic LogFire | Native FastAPI/HTTPX/LLM integration |

### 5.2 Pydantic v2 + LogFire

**Pydantic v2 (Validation Layer)**

Pydantic v2 features a complete rewrite with Rust core (`pydantic-core`) providing 5-50x faster validation. Already used by FastAPI.

```python
from pydantic import BaseModel, Field

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    stream: bool = False
    tools: list[Tool] | None = None

# Automatic validation on instantiation
# Type coercion and error messages built-in
```

Benefits:
- Rust-powered validation (pydantic-core)
- Native support in FastAPI
- Strict mode for production safety
- JSON Schema generation for API docs

**Pydantic LogFire (Observability Layer)**

LogFire is built on OpenTelemetry and provides native instrumentation for our entire stack:

```python
import logfire
from fastapi import FastAPI

logfire.configure()  # Auto-detects environment
app = FastAPI()
logfire.instrument_fastapi(app)  # Request tracing
logfire.instrument_httpx()        # Outbound HTTP calls
logfire.instrument_aiosqlite()    # Database queries

# Custom spans for inference
with logfire.span("model_inference", model=model_id):
    response = await generate(...)
    logfire.info("tokens_generated", input_tokens=..., output_tokens=...)
```

Key Features:
- **Auto-instrumentation**: FastAPI, HTTPX, aiosqlite, asyncio
- **LLM tracking**: OpenAI/Anthropic client instrumentation with token counts
- **OpenTelemetry export**: Can send to Prometheus, Jaeger, or any OTLP backend
- **MIT license**: Open source SDK
- **Dashboard**: Optional LogFire cloud or self-hosted visualization

Why LogFire over raw Prometheus + OpenTelemetry:
1. **Native Python integration** - decorators, context managers, type-safe spans
2. **Automatic LLM metrics** - token usage, latency, costs per model
3. **Zero-config FastAPI** - single line adds full request tracing
4. **Pydantic integration** - validates log data, generates schemas

### 5.3 Rust Components (Performance Critical)

**Phase 1 - High Impact, Low Complexity:**

| Component | Benefit | Implementation |
|-----------|---------|----------------|
| **Tokenizer** | Already Rust (HuggingFace) | Just use it |
| **SSE Encoder** | Fast frame serialization | PyO3 extension |
| **JSON Validator** | Request validation | Pydantic v2 (Rust core) |

**Phase 2 - Advanced (Optional):**

| Component | Benefit | Complexity |
|-----------|---------|------------|
| Request Router | Low-latency routing | Medium |
| Connection Pool | Efficient connections | Medium |
| Metrics Aggregator | High-frequency metrics | Low |

### 5.4 Async Architecture

```python
# Main server loop
import uvloop
uvloop.install()

async def main():
    # Initialize components
    model_pool = ModelPoolManager(max_memory_gb=48.0)
    scheduler = ContinuousBatchingScheduler(max_batch_size=8)
    kv_cache = PagedKVCacheManager(block_size=16, num_blocks=2048)

    # Start background tasks
    asyncio.create_task(scheduler.scheduling_loop())
    asyncio.create_task(health_monitor.monitor_loop())

    # Start FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()
```

### 5.5 Memory Management Strategy

**Apple Silicon Unified Memory Advantages:**
- CPU and GPU share same memory pool
- Zero-copy operations between CPU/GPU
- No explicit memory transfers needed

**Management Approach:**
```python
# Set memory limits
mx.metal.set_memory_limit(max_memory_bytes, relaxed=True)
mx.metal.set_cache_limit(cache_limit_bytes)

# Monitor usage
def check_memory_pressure() -> float:
    active = mx.metal.get_active_memory()
    peak = mx.metal.get_peak_memory()
    return active / max_memory_bytes

# Cleanup after request
def cleanup_after_request():
    mx.synchronize()  # Wait for pending operations
    mx.metal.clear_cache()
```

---

## 6. Multi-Model Architecture

### 6.1 Concurrent Model Serving

**Design Goal:** Multiple models hot in memory, serving different request types.

```
+-------------------------------------------------------------+
|                    Request Router                           |
|                                                             |
|  model="llama-3.2-3b"  ------> Llama Model Instance        |
|  model="qwen-vl-2b"    ------> Qwen VL Model Instance      |
|  model="mistral-7b"    ------> Mistral Model Instance      |
|  model="embed-v3"      ------> Embedding Model Instance    |
|                                                             |
|  (All share unified memory pool)                           |
+-------------------------------------------------------------+
```

### 6.2 On-Demand Loading

```python
class OnDemandLoader:
    async def ensure_model_loaded(self, model_id: str) -> LoadedModel:
        if model_id in self.hot_models:
            return self.hot_models[model_id]

        # Queue request while loading
        if model_id in self.loading:
            await self.loading[model_id].wait()
            return self.hot_models[model_id]

        # Start loading
        self.loading[model_id] = asyncio.Event()
        try:
            model = await self._load_model(model_id)
            self.hot_models[model_id] = model
            self.loading[model_id].set()
            return model
        finally:
            del self.loading[model_id]
```

### 6.3 Memory Budget

**Example Configuration (64GB Mac):**

| Model Type | Size | Instances | Memory |
|------------|------|-----------|--------|
| 7B LLM (4-bit) | ~4GB | 2 | 8GB |
| 3B VLM (4-bit) | ~2GB | 1 | 2GB |
| Embedding | ~500MB | 1 | 0.5GB |
| KV Cache Pool | - | - | 8GB |
| **Total** | - | - | **~20GB** |

Leaves ~44GB for system and headroom.

---

## 7. API Design

### 7.1 OpenAI-Compatible Endpoints

```
POST /v1/chat/completions     # Chat completions
POST /v1/completions          # Legacy completions
POST /v1/embeddings           # Text embeddings
GET  /v1/models               # List available models
```

### 7.2 Anthropic-Compatible Endpoints

```
POST /v1/messages             # Messages API
POST /v1/complete             # Legacy (optional)
```

### 7.3 Management Endpoints

```
GET  /health                  # Liveness check
GET  /ready                   # Readiness check
GET  /metrics                 # Prometheus metrics
POST /admin/models/load       # Preload model
POST /admin/models/unload     # Unload model
GET  /admin/models/status     # Model pool status
```

### 7.4 Streaming Formats

**OpenAI SSE:**
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Anthropic SSE:**
```
event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}

event: message_stop
data: {"type":"message_stop"}
```

---

## 8. Performance Targets

### 8.1 Baseline vs Target

| Metric | Current (mlx-openai-server) | Target | Method |
|--------|----------------------------|--------|--------|
| Single request | 200-400 tok/s | 200-400 tok/s | Same |
| 5 concurrent | 200-400 tok/s | 800-1200 tok/s | Continuous batching |
| TTFT (cold) | 500-1000ms | 500-1000ms | Same |
| TTFT (cached prefix) | 500-1000ms | 50-100ms | Prefix caching |
| Memory efficiency | ~40% | >90% | Paged KV cache |
| Model hot-swap | N/A | <5s | LRU pool |

### 8.2 Expected Benchmarks

Based on vLLM-MLX results on M4 Max:

| Configuration | Throughput |
|---------------|------------|
| Single model, single request | 200-400 tok/s |
| Single model, 5 concurrent | 600-1100 tok/s |
| Multi-model (3 hot), 5 concurrent | 400-800 tok/s |

---

## 9. Implementation Roadmap

### Phase 1: Foundation (4-6 weeks)
- [ ] FastAPI + uvloop server skeleton
- [ ] OpenAI Chat Completions endpoint
- [ ] Single model inference (mlx-lm)
- [ ] SSE streaming
- [ ] Basic health checks

### Phase 2: Multi-Model (4-6 weeks)
- [ ] Model pool manager with LRU
- [ ] Request router by model parameter
- [ ] Vision model support (mlx-vlm)
- [ ] Embedding model support (mlx-embeddings)
- [ ] Hot-swap without downtime

### Phase 3: Performance (4-6 weeks)
- [ ] Paged KV cache manager
- [ ] Continuous batching scheduler
- [ ] Prefix caching
- [ ] Priority queues

### Phase 4: Dual Protocol (2-4 weeks)
- [ ] Anthropic Messages API
- [ ] Protocol translator
- [ ] Streaming format translation

### Phase 5: Production Hardening (4-6 weeks)
- [ ] Pydantic LogFire integration (auto-instrumentation for FastAPI, HTTPX, SQLite)
- [ ] LLM token tracking and cost metrics
- [ ] Circuit breaker pattern
- [ ] Rate limiting
- [ ] Authentication (AuthLib)

### Phase 6: Optimization (Ongoing)
- [ ] Rust tokenization integration
- [ ] Speculative decoding (optional)
- [ ] Chunked prefill (if MLX supports)

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MLX memory leaks | Medium | High | Explicit cleanup, monitoring |
| Batching complexity | Medium | Medium | Start with vLLM-MLX patterns |
| Model adapter sprawl | High | Low | Good abstraction, registry |
| KV cache state corruption | Medium | High | Per-request isolation |

### 10.2 Ecosystem Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| mlx-lm breaking changes | Low | Medium | Pin versions, track releases |
| MLX deprecation | Very Low | High | Apple commitment to ML ecosystem |
| vLLM-MLX abandonment | Medium | Low | We own our implementation |

---

## 11. Conclusion

Building a custom MLX multi-model server is **feasible and recommended**.

**Key Strengths:**
1. **Solid foundation:** mlx-lm, mlx-vlm, mlx-embeddings are mature
2. **Proven techniques:** vLLM-MLX demonstrates 3.4x batching speedup
3. **Unique advantages:** Apple Silicon unified memory simplifies architecture
4. **Clear path:** Existing implementations provide reference patterns

**Recommended Approach:**
1. Start with mlx-lm/mlx-vlm wrappers (not raw MLX)
2. Implement continuous batching early (biggest ROI)
3. Add paged KV cache for memory efficiency
4. Use LM Studio's VisionAddOn pattern for multimodal
5. Build protocol translator for dual API support
6. Use Pydantic v2 for all validation (Rust-powered, FastAPI-native)
7. Integrate Pydantic LogFire for observability from day one

**Expected Outcome:**
- 2-4x throughput improvement over current mlx-openai-server
- Native OpenAI + Anthropic API support
- Multi-model serving with dynamic loading
- Production-ready reliability and observability

---

## References

### Core Libraries
- [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm)
- [mlx-vlm GitHub](https://github.com/Blaizzy/mlx-vlm)
- [mlx-embeddings GitHub](https://github.com/Blaizzy/mlx-embeddings)
- [MLX Framework](https://ml-explore.github.io/mlx/)

### Performance References
- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [vLLM-MLX GitHub](https://github.com/waybarrios/vllm-mlx)
- [vLLM Continuous Batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)

### Server Implementations
- [mlx-openai-server](https://github.com/cubist38/mlx-openai-server)
- [mlx-omni-server](https://github.com/madroidmaq/mlx-omni-server)
- [LM Studio mlx-engine](https://github.com/lmstudio-ai/mlx-engine)

### Rust Integration
- [PyO3](https://github.com/PyO3/pyo3)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [uvloop](https://github.com/MagicStack/uvloop)

### Validation & Observability
- [Pydantic v2](https://docs.pydantic.dev/latest/)
- [Pydantic LogFire](https://logfire.pydantic.dev/)
- [pydantic-core (Rust)](https://github.com/pydantic/pydantic-core)

### Protocol Specifications
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/en/api)
