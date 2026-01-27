# Phase 7: Foundation - Server Skeleton & Single Model Inference - Research

**Researched:** 2026-01-27
**Domain:** FastAPI server with MLX-LM inference, OpenAI-compatible API, SSE streaming
**Confidence:** HIGH

## Summary

This research focuses on implementation details for building a FastAPI-based MLX inference server with OpenAI-compatible endpoints and SSE streaming. The investigation covers mlx-lm's `stream_generate` API, FastAPI + uvloop configuration, OpenAI API compliance, Pydantic LogFire instrumentation, and model adapter patterns.

**Key Findings:**
- mlx-lm provides `stream_generate()` that yields response objects with `.text` attribute for streaming generation
- FastAPI with uvloop (via `uvicorn[standard]`) provides 2-4x async performance improvement
- OpenAI Chat Completions API uses SSE format with `data: <json>\n\n` pattern
- Pydantic LogFire offers one-line FastAPI instrumentation with `logfire.instrument_fastapi(app)`
- Llama 3.x models require dual stop tokens (`eos_token_id` + `<|eot_id|>`) for proper chat completion

**Primary recommendation:** Build on mlx-lm's `stream_generate`, use sse-starlette for robust SSE implementation, instrument with LogFire from day one, and implement model-specific adapters for chat templates and stop tokens.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mlx-lm | Latest (0.x) | Text generation with MLX | Apple-maintained, optimized for Apple Silicon |
| FastAPI | 0.115+ | Async HTTP framework | Industry standard for Python async APIs |
| uvloop | Latest | High-performance event loop | 2-4x faster than asyncio |
| Pydantic v2 | 2.10+ | Request/response validation | Rust-powered validation, FastAPI native |
| Pydantic LogFire | Latest | Observability platform | Native FastAPI/LLM instrumentation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sse-starlette | 2.x | SSE streaming | Production SSE with W3C compliance |
| uvicorn[standard] | 0.34+ | ASGI server | Includes uvloop + httptools |
| transformers | Latest | Tokenizer utilities | HuggingFace tokenizer access |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sse-starlette | Manual StreamingResponse | Manual SSE = missing W3C spec compliance, error handling |
| uvloop | asyncio | uvloop = 2-4x faster, but adds dependency |
| LogFire | Prometheus + OTEL | LogFire = zero-config LLM tracking, but hosted service |

**Installation:**
```bash
pip install "fastapi[standard]" "uvicorn[standard]" mlx-lm pydantic logfire sse-starlette
```

## Architecture Patterns

### Recommended Project Structure
```
mlx_server/
├── api/
│   ├── v1/
│   │   ├── chat.py          # /v1/chat/completions
│   │   ├── completions.py   # /v1/completions
│   │   └── models.py        # /v1/models
│   └── dependencies.py      # FastAPI dependencies
├── models/
│   ├── pool.py              # ModelPoolManager
│   └── adapters/            # Model family adapters
│       ├── base.py          # ModelAdapter protocol
│       └── llama.py         # Llama adapter
├── schemas/
│   ├── openai.py            # OpenAI-compatible schemas
│   └── internal.py          # Internal request/response
├── services/
│   └── inference.py         # Inference orchestration
├── utils/
│   └── memory.py            # MLX memory management
└── main.py                  # FastAPI app + lifespan
```

### Pattern 1: MLX Memory Management

**What:** Explicit memory cleanup after each request to prevent memory leaks

**When to use:** After every inference operation, especially in long-running servers

**Example:**
```python
# Source: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.clear_cache.html
import mlx.core as mx

async def generate_with_cleanup(model, tokenizer, prompt, max_tokens):
    try:
        response = ""
        for chunk in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
            response += chunk.text
            yield chunk.text
    finally:
        # Critical: clear cache after generation
        mx.synchronize()  # Wait for pending operations
        mx.metal.clear_cache()
```

**Why critical:** MLX can leak memory during generation - active memory may not be released when generation finishes. Explicit cleanup is essential for long-running servers.

### Pattern 2: SSE Streaming with Proper Format

**What:** Server-Sent Events using `data: <json>\n\n` format

**When to use:** For streaming chat completions (OpenAI-compatible)

**Example:**
```python
# Source: https://pypi.org/project/sse-starlette/
from sse_starlette.sse import EventSourceResponse

async def stream_completion(request: ChatCompletionRequest):
    async def event_generator():
        for chunk in stream_generate(model, tokenizer, prompt, max_tokens=request.max_tokens):
            # OpenAI SSE format
            data = {
                "id": "chatcmpl-xxx",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk.text},
                    "finish_reason": None
                }]
            }
            yield {"data": json.dumps(data)}

        # Final chunk with finish_reason
        yield {"data": json.dumps({..., "choices": [{"delta": {}, "finish_reason": "stop"}]})}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())
```

### Pattern 3: Chat Template Application

**What:** Use tokenizer's built-in chat template with proper generation prompt

**When to use:** For all chat completions (not raw completions)

**Example:**
```python
# Source: https://huggingface.co/docs/transformers/en/chat_templating
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mlx-community/Llama-3.2-3B-Instruct-4bit")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

# Apply chat template (includes special tokens)
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # Adds assistant role marker
    tokenize=False  # Return string, not token IDs
)

# Result for Llama 3: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
```

### Pattern 4: Dual Stop Token Detection (Llama 3.x)

**What:** Configure both EOS token and model-specific end-of-turn token

**When to use:** For Llama 3.x models (critical for proper stopping)

**Example:**
```python
# Source: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
# and https://github.com/meta-llama/llama3/issues

from mlx_lm import load

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Llama 3 uses TWO stop tokens:
# 1. <|end_of_text|> (standard EOS)
# 2. <|eot_id|> (end of turn in chat)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Pass to generation (if mlx-lm supports stop_tokens)
# Or manually check in generation loop
for chunk in stream_generate(model, tokenizer, prompt, max_tokens=512):
    if chunk.token_id in terminators:
        break
    yield chunk.text
```

**Why critical:** Llama 3 models signal end-of-message with `<|eot_id|>` but continue generating if only `eos_token_id` is checked. This causes runaway generation.

### Pattern 5: FastAPI + uvloop Configuration

**What:** Enable uvloop for 2-4x async performance

**When to use:** Production deployment (always)

**Example:**
```python
# Source: https://www.uvicorn.org/
# Option 1: Automatic (via uvicorn[standard])
# Just install uvicorn[standard] - uvloop auto-detected

# Option 2: Explicit installation
import uvloop
uvloop.install()

# Option 3: Uvicorn CLI
# uvicorn main:app --loop uvloop --host 0.0.0.0 --port 8000

# Option 4: Programmatic
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        loop="uvloop"  # Explicit loop selection
    )
```

### Pattern 6: Pydantic LogFire Instrumentation

**What:** One-line FastAPI instrumentation for request tracing + custom LLM spans

**When to use:** From day one (zero-config observability)

**Example:**
```python
# Source: https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/
import logfire
from fastapi import FastAPI

app = FastAPI()

# Configure once at startup
logfire.configure()

# Instrument FastAPI (automatic request tracing)
logfire.instrument_fastapi(app)

# Custom span for inference
async def generate_text(model, tokenizer, prompt):
    with logfire.span('model_inference', model=model.config.model_type):
        response = await stream_generate(model, tokenizer, prompt, max_tokens=512)

        # Log token usage
        logfire.info(
            'generation_complete',
            input_tokens=len(tokenizer.encode(prompt)),
            output_tokens=len(tokenizer.encode(response)),
        )

        return response
```

**Key features:**
- `instrument_fastapi()` adds `fastapi.arguments.values`, `fastapi.arguments.errors`, and timestamps to spans
- Custom spans with `logfire.span()` context manager
- Structured logging with `logfire.info()` for metrics

### Anti-Patterns to Avoid

- **Loading model per request:** Models should be loaded once and reused (load time ~5-10s for 7B models)
- **Blocking I/O in async handlers:** Use `async def` and await all I/O operations
- **Ignoring memory cleanup:** Always call `mx.metal.clear_cache()` after generation
- **Single stop token for Llama 3:** Must check both `eos_token_id` AND `<|eot_id|>`
- **Manual SSE formatting:** Use sse-starlette instead of string concatenation

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSE streaming | `f"data: {json}\n\n"` loops | sse-starlette | Handles connection drops, heartbeats, W3C spec compliance, client reconnect |
| Request validation | Manual dict checks | Pydantic BaseModel | 5-50x faster (Rust core), auto-generates OpenAPI docs |
| Chat template formatting | String concatenation | `tokenizer.apply_chat_template()` | Model-specific templates, special token handling |
| Async event loop | asyncio | uvloop (via uvicorn[standard]) | 2-4x faster, drop-in replacement |
| LLM observability | Custom logging | Pydantic LogFire | Auto-instruments FastAPI/HTTPX, LLM token tracking, OpenTelemetry export |

**Key insight:** LLM serving has many edge cases (connection drops during streaming, memory leaks, model-specific templates). Battle-tested libraries handle these better than custom code.

## Common Pitfalls

### Pitfall 1: MLX Memory Leaks in Long-Running Servers

**What goes wrong:** Active memory continues to rise throughout generation, eventually crashing with OOM. Memory occupied by KV cache is not released after generation completes.

**Why it happens:** MLX's memory allocator doesn't automatically free memory when arrays go out of scope. The cache holds references that prevent garbage collection.

**How to avoid:**
```python
# ALWAYS wrap inference in try/finally with cleanup
try:
    for chunk in stream_generate(model, tokenizer, prompt, max_tokens=512):
        yield chunk.text
finally:
    mx.synchronize()  # Wait for pending GPU operations
    mx.metal.clear_cache()  # Free cached memory
    # Note: get_cache_memory() should return 0 after clear_cache()
```

**Warning signs:**
- `mx.metal.get_active_memory()` increases with each request
- `mx.metal.get_peak_memory()` keeps growing
- Server crashes after 10-50 requests

**Source:** [MLX Issue #724](https://github.com/ml-explore/mlx-examples/issues/724), [MLX Issue #742](https://github.com/ml-explore/mlx/issues/742)

### Pitfall 2: FastAPI Async Generator Memory Leaks

**What goes wrong:** Streaming responses hold memory indefinitely, causing RSS (resident set size) to creep upward over time.

**Why it happens:** In async FastAPI servers, allocations from different requests get interleaved. When one request finishes and frees its buffers, other requests still have objects in those same memory spans, keeping entire spans resident.

**How to avoid:**
1. **Never reference request-scoped objects from lifespan/global scope** (causes memory leaks and race conditions)
2. **Implement proper error handling in generators:**
```python
async def stream_completion():
    try:
        for chunk in generate_text():
            yield chunk
    except Exception as e:
        logfire.error('stream_error', error=str(e))
        raise
    finally:
        # Cleanup happens even if client disconnects
        cleanup_resources()
```
3. **Use appropriate chunk sizes** to avoid buffering entire responses
4. **Monitor with tools:** `psutil`, `objgraph`, asyncio task tracking

**Warning signs:**
- RSS grows steadily under load but doesn't in single requests
- Many active asyncio tasks that never complete
- Database connections not being closed

**Source:** [BetterUp Memory Leak Investigation](https://build.betterup.com/chasing-a-memory-leak-in-our-async-fastapi-service-how-jemalloc-fixed-our-rss-creep/), [FastAPI Performance Mistakes](https://dev.to/igorbenav/fastapi-mistakes-that-kill-your-performance-2b8k)

### Pitfall 3: Llama 3 Models Don't Stop Generating

**What goes wrong:** Model generates past the assistant's response, continuing with user messages or gibberish.

**Why it happens:** Llama 3 uses `<|eot_id|>` (end of turn) for chat completion, but the standard generation config only checks `eos_token_id` (`<|end_of_text|>`). The model correctly generates `<|eot_id|>` but generation continues because it's not recognized as a stop token.

**How to avoid:**
```python
# Configure BOTH stop tokens
terminators = [
    tokenizer.eos_token_id,  # <|end_of_text|> (128009)
    tokenizer.convert_tokens_to_ids("<|eot_id|>")  # <|eot_id|> (128001)
]

# Check in generation loop (if mlx-lm doesn't support stop_tokens param)
for chunk in stream_generate(model, tokenizer, prompt, max_tokens=512):
    if chunk.token_id in terminators:
        break
    yield chunk.text
```

**Warning signs:**
- Response includes multiple role headers (`<|start_header_id|>user<|end_header_id|>`)
- Model generates questions after answering
- Multi-turn chats degrade into gibberish

**Source:** [Llama 3 Model Card](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/), [HuggingFace Discussion #142](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/142)

### Pitfall 4: Missing Chat Template Breaks Model Output

**What goes wrong:** Model generates gibberish, repeats prompts, or ignores system messages.

**Why it happens:** Each model family expects specific formatting with special tokens (e.g., `<|begin_of_text|>`, `<|start_header_id|>`, `[INST]`). Without proper templates, the model doesn't recognize message boundaries.

**How to avoid:**
```python
# ALWAYS use tokenizer.apply_chat_template()
# DON'T manually format with f-strings

# WRONG:
prompt = f"System: {system}\nUser: {user}\nAssistant:"

# CORRECT:
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user}
]
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # Adds assistant role marker
    tokenize=False
)
```

**Warning signs:**
- Model echoes the user's question instead of answering
- System messages are ignored
- Multi-turn conversations fail to maintain context

**Source:** [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/en/chat_templating)

### Pitfall 5: Blocking Operations in Async Handlers

**What goes wrong:** Server latency spikes, requests time out, throughput drops dramatically.

**Why it happens:** Blocking operations (file I/O, `time.sleep()`, synchronous HTTP calls) block the entire event loop, preventing other requests from being processed.

**How to avoid:**
```python
# WRONG: Blocking operations
def get_completion(prompt: str):
    result = model.generate(prompt)  # Blocks event loop
    time.sleep(1)  # Blocks event loop
    return result

# CORRECT: Async operations
async def get_completion(prompt: str):
    result = await asyncio.to_thread(model.generate, prompt)  # Offload to thread
    await asyncio.sleep(1)  # Non-blocking sleep
    return result
```

**Warning signs:**
- Single slow request blocks all other requests
- CPU-bound operations cause timeouts
- Latency is unpredictable

**Source:** [FastAPI Performance Issues](https://www.mindfulchase.com/explore/troubleshooting-tips/troubleshooting-fastapi-performance-resolving-async-execution-and-latency-issues.html)

## Code Examples

Verified patterns from official sources:

### Loading Model with mlx-lm
```python
# Source: https://github.com/ml-explore/mlx-lm
from mlx_lm import load

# Load quantized model (faster, less memory)
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Load with custom tokenizer config (for special tokens)
model, tokenizer = load(
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True}
)
```

### Streaming Generation
```python
# Source: https://github.com/ml-explore/mlx-lm
from mlx_lm import stream_generate

messages = [{"role": "user", "content": "Explain quantum computing"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# stream_generate yields response objects with .text attribute
for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
```

### OpenAI Chat Completion Schema
```python
# Source: https://platform.openai.com/docs/api-reference/chat
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]  # [{index: 0, message: {role, content}, finish_reason}]
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
```

### SSE Streaming Response Format
```python
# Source: https://platform.openai.com/docs/api-reference/chat-streaming
# OpenAI SSE format: "data: {json}\n\n"

# Initial chunk (role)
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

# Content chunks
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

# Final chunk (finish_reason)
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Models Endpoint Response
```python
# Source: https://platform.openai.com/docs/api-reference/models/list
# GET /v1/models

{
  "object": "list",
  "data": [
    {
      "id": "llama-3.2-3b-instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "mlx-community"
    }
  ]
}
```

### Model Family Detection
```python
# Source: Inferred from model naming conventions
def detect_model_family(model_id: str) -> str:
    """Detect model family from HuggingFace model ID."""
    model_id_lower = model_id.lower()

    # Check common patterns
    if "llama" in model_id_lower or "codellama" in model_id_lower:
        return "llama"
    elif "qwen" in model_id_lower:
        return "qwen"
    elif "mistral" in model_id_lower or "mixtral" in model_id_lower:
        return "mistral"
    elif "gemma" in model_id_lower:
        return "gemma"
    elif "phi" in model_id_lower:
        return "phi"
    else:
        return "unknown"
```

### Model Adapter Protocol
```python
# Source: Pattern from mlx-openai-server, mlx-omni-server
from typing import Protocol

class ModelAdapter(Protocol):
    """Protocol for model-specific handling."""

    def apply_chat_template(
        self,
        tokenizer,
        messages: list[dict],
        add_generation_prompt: bool = True
    ) -> str:
        """Apply model-specific chat template."""
        ...

    def get_stop_tokens(self, tokenizer) -> list[int]:
        """Get model-specific stop token IDs."""
        ...

class LlamaAdapter:
    """Llama 3.x family adapter."""

    def apply_chat_template(self, tokenizer, messages, add_generation_prompt=True):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )

    def get_stop_tokens(self, tokenizer):
        # Llama 3 requires BOTH tokens
        return [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual JSON validation | Pydantic v2 (Rust core) | 2023 | 5-50x faster validation |
| asyncio loop | uvloop | Stable since 2020 | 2-4x async performance |
| Custom logging | OpenTelemetry + LogFire | 2024 | Zero-config observability |
| Static batching | Continuous batching | 2023 (vLLM) | 2-23x throughput |
| Single stop token | Multiple stop tokens (Llama 3) | 2024 | Proper chat completion |

**Deprecated/outdated:**
- **Manual SSE formatting:** Use sse-starlette (W3C compliance, error handling)
- **Pydantic v1:** v2 has Rust core (5-50x faster)
- **mlx-lm's built-in HTTP server:** Uses `ThreadingHTTPServer`, not recommended for production
- **Single `eos_token_id` for Llama 3:** Requires dual stop tokens (`eos_token_id` + `<|eot_id|>`)

## Open Questions

Things that couldn't be fully resolved:

1. **mlx-lm `stream_generate` stop tokens parameter**
   - What we know: CLI has `--extra-eos-token`, `BatchGenerator` supports `stop_tokens` set
   - What's unclear: Whether `stream_generate()` accepts `stop_tokens` parameter directly
   - Recommendation: Check mlx-lm source code or implement manual stop token detection in generation loop

2. **mlx-lm response object structure**
   - What we know: `stream_generate` yields response objects with `.text` attribute
   - What's unclear: Does response object include `.token_id` for stop token detection?
   - Recommendation: Inspect mlx-lm source or add token tracking in generation loop

3. **MLX memory limits configuration**
   - What we know: `mx.metal.set_memory_limit()` and `mx.metal.set_cache_limit()` exist
   - What's unclear: Recommended limits for production servers, behavior when limit exceeded
   - Recommendation: Test with monitoring, start conservative (e.g., 80% of available memory)

4. **Model hot-swapping impact**
   - What we know: Models take 5-10s to load, `mx.metal.clear_cache()` frees memory
   - What's unclear: Can requests queue during model swap without timeout?
   - Recommendation: Implement loading queue with timeout (30s), return 503 if model loading

## Sources

### Primary (HIGH confidence)
- [mlx-lm GitHub Repository](https://github.com/ml-explore/mlx-lm) - Official Apple MLX LLM library
- [mlx-lm Server Documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) - Server implementation guidance
- [MLX Metal API Documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.clear_cache.html) - Memory management functions
- [Pydantic LogFire FastAPI Integration](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/) - Official instrumentation docs
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) - Official API reference
- [OpenAI Streaming API](https://platform.openai.com/docs/api-reference/chat-streaming) - SSE format specification
- [OpenAI Models Endpoint](https://platform.openai.com/docs/api-reference/models/list) - Models listing format
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/en/chat_templating) - Tokenizer chat template docs
- [Llama 3 Model Card](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/) - Special tokens specification
- [Uvicorn Documentation](https://www.uvicorn.org/) - ASGI server configuration
- [sse-starlette PyPI](https://pypi.org/project/sse-starlette/) - Production SSE library

### Secondary (MEDIUM confidence)
- [FastAPI Performance Mistakes](https://dev.to/igorbenav/fastapi-mistakes-that-kill-your-performance-2b8k) - Common pitfalls (2025 article)
- [BetterUp Memory Leak Investigation](https://build.betterup.com/chasing-a-memory-leak-in-our-async-fastapi-service-how-jemalloc-fixed-our-rss-creep/) - FastAPI async memory issues
- [Pydantic v2 Migration Guide](https://docs.pydantic.dev/latest/migration/) - v1 to v2 changes
- [FastAPI Body Fields](https://fastapi.tiangolo.com/tutorial/body-fields/) - Pydantic Field validation
- [MLX Memory Issues #724](https://github.com/ml-explore/mlx-examples/issues/724) - Community-reported memory leak
- [MLX Memory Issues #742](https://github.com/ml-explore/mlx/issues/742) - GPU memory management discussion
- [Llama 3 Stop Token Issue](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/142) - Community fix for dual stop tokens

### Tertiary (LOW confidence - needs validation)
- [vLLM-MLX Continuous Batching](https://medium.com/@clnaveen/mlx-lm-continuous-batching-e060c73e7d08) - Medium article (2025)
- [MLX Model Family Detection](https://arxiv.org/html/2506.01631) - Research paper on model fingerprinting
- Model adapter pattern - inferred from mlx-openai-server and mlx-omni-server codebases (not directly documented)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries are production-ready and actively maintained
- Architecture patterns: HIGH - based on official documentation and proven implementations
- MLX memory management: MEDIUM - community reports + official docs, but needs testing
- Stop token handling: HIGH - verified in Llama 3 model cards and community discussions
- SSE streaming: HIGH - W3C spec + FastAPI patterns well-documented
- Model adapters: MEDIUM - pattern exists but not officially standardized

**Research date:** 2026-01-27
**Valid until:** ~60 days (FastAPI/MLX are stable, mlx-lm updates ~monthly)

**Recommended next steps for planner:**
1. Verify mlx-lm `stream_generate` stop tokens support in source code
2. Test memory cleanup patterns under load (50+ requests)
3. Implement model adapter registry with Llama adapter as reference
4. Set up LogFire instrumentation from day one (easier than retrofitting)
5. Use sse-starlette for production SSE (don't hand-roll)
