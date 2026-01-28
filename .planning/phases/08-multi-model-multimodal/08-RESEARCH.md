# Phase 8: Multi-Model & Multimodal Support - Research

**Researched:** 2026-01-28
**Domain:** MLX multi-model pool manager, vision model integration (mlx-vlm), embeddings API (mlx-embeddings), model family adapters (Qwen, Mistral, Gemma)
**Confidence:** HIGH (core libraries verified via official repos and docs)

## Summary

Phase 8 extends the Phase 7 single-model server into a multi-model serving platform. The research covers five key areas: (1) the Model Pool Manager with LRU eviction and memory pressure detection, (2) vision model integration via mlx-vlm with image preprocessing, (3) embeddings API via mlx-embeddings with batch support, (4) model family adapters for Qwen/Mistral/Gemma, and (5) a unified model type detection system.

The Phase 7 codebase already provides strong foundations: `ModelPoolManager` with async loading, `LoadedModel` dataclass with `touch()` for LRU, adapter registry with `register_adapter()`, memory utilities (`get_memory_usage`, `clear_cache`, `set_memory_limit`), and the queue-based threading pattern for MLX Metal thread affinity. The existing `model_detection.py` utility already detects multimodal models via `detect_multimodal()` and normalizes architecture families -- this should be reused and extended for the MLX server's detection chain.

**Primary recommendation:** Build LRU eviction and memory pressure detection directly into the existing `ModelPoolManager`. Add `mlx-vlm` and `mlx-embeddings` as separate model type handlers that share the same pool infrastructure. Implement Qwen/Mistral/Gemma adapters following the exact pattern of the existing `LlamaAdapter`. Use `httpx` for async image URL fetching and `Pillow` for resize/base64 decode. Keep the OpenAI embeddings response format as the contract for the embeddings endpoint.

## Standard Stack

The established libraries/tools for this domain:

### Core (already in Phase 7, verified)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mlx-lm | Latest (0.x) | Text generation with MLX | Apple-maintained, Phase 7 foundation |
| mlx-vlm | 0.3.9+ | Vision-language model inference | Only MLX-native VLM library |
| mlx-embeddings | Latest | Text embedding generation | Only MLX-native embeddings library |
| FastAPI | 0.115+ | Async HTTP framework | Phase 7 foundation |
| Pydantic v2 | 2.10+ | Request/response validation | Phase 7 foundation |
| sse-starlette | 2.x | SSE streaming | Phase 7 foundation |

### Supporting (new for Phase 8)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pillow (PIL) | 10.x+ | Image resize, base64 decode | Processing vision inputs |
| httpx | 0.27+ | Async HTTP client for image URLs | Fetching images from URLs |
| mlx.core | Latest | Memory management APIs | LRU eviction memory checks |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| httpx (async) | requests (sync) | httpx is already a FastAPI ecosystem pattern, async-native, avoids blocking event loop |
| Pillow | opencv-python | Pillow is lighter, sufficient for resize/decode, no need for CV2's complexity |
| mlx-vlm.generate() | VisionAddOn pattern (LM Studio) | VisionAddOn is more performant (prompt caching for text-only) but less mature, only Gemma3/Pixtral supported. Use mlx-vlm.generate() for Phase 8 reliability, consider VisionAddOn migration in Phase 9+ |

**Installation:**
```bash
pip install mlx-vlm mlx-embeddings Pillow httpx
```

## Architecture Patterns

### Recommended Project Structure (extending Phase 7)
```
mlx_server/
├── api/
│   ├── v1/
│   │   ├── chat.py          # Existing -- extend for vision content blocks
│   │   ├── completions.py   # Existing (unchanged)
│   │   ├── models.py        # Existing -- extend with model type info
│   │   ├── embeddings.py    # NEW: /v1/embeddings endpoint
│   │   └── admin.py         # NEW: /admin/models/* management endpoints
│   └── dependencies.py      # Future
├── models/
│   ├── pool.py              # EXTEND: add LRU eviction, preload protection, memory pressure
│   ├── adapters/
│   │   ├── base.py          # Existing -- extend ModelAdapter protocol for model_type
│   │   ├── llama.py         # Existing (reference implementation)
│   │   ├── qwen.py          # NEW: Qwen family adapter
│   │   ├── mistral.py       # NEW: Mistral family adapter
│   │   ├── gemma.py         # NEW: Gemma family adapter
│   │   └── registry.py      # Existing -- register new adapters
│   ├── detection.py         # NEW: model type detection (text-gen / vision / embeddings)
│   └── types.py             # NEW: ModelType enum, LoadedVisionModel, LoadedEmbeddingModel
├── schemas/
│   ├── openai.py            # Existing -- extend for vision content blocks + embeddings
│   └── internal.py          # Future
├── services/
│   ├── inference.py         # Existing -- extend for vision generation
│   ├── embeddings.py        # NEW: embeddings generation service
│   └── image_processor.py   # NEW: image fetch, decode, resize
├── utils/
│   └── memory.py            # Existing -- extend with model size estimation
└── main.py                  # Existing -- register new routers
```

### Pattern 1: LRU Eviction with Preload Protection

**What:** Models are evicted least-recently-used, but preloaded models are protected. Eviction triggers BEFORE loading a new model when memory pressure is detected.

**When to use:** Every time `get_model()` is called and model is not hot.

**Example (extending existing pool.py):**
```python
# Source: Extends existing ModelPoolManager from Phase 7
import dataclasses

@dataclasses.dataclass
class LoadedModel:
    model_id: str
    model: Any
    tokenizer: Any
    model_type: str  # NEW: "text-gen" | "vision" | "embeddings"
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    size_gb: float = 0.0
    preloaded: bool = False  # NEW: protected from eviction

    def touch(self) -> None:
        self.last_used = time.time()


class ModelPoolManager:
    def __init__(self, max_memory_gb: float = 48.0, max_models: int = 4,
                 memory_limit_pct: float | None = None):
        self._models: dict[str, LoadedModel] = {}
        self._loading: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        # Memory limit: either absolute GB or percentage of system memory
        self.max_memory_gb = max_memory_gb
        self.memory_limit_pct = memory_limit_pct  # e.g., 0.75 = 75%
        self.max_models = max_models

    def _get_effective_memory_limit(self) -> float:
        """Resolve memory limit from percentage or absolute GB."""
        if self.memory_limit_pct is not None:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024**3)
            return total_gb * self.memory_limit_pct
        return self.max_memory_gb

    def _current_memory_gb(self) -> float:
        """Sum of all loaded model sizes."""
        return sum(m.size_gb for m in self._models.values())

    async def get_model(self, model_id: str) -> LoadedModel:
        async with self._lock:
            if model_id in self._models:
                self._models[model_id].touch()
                return self._models[model_id]

        # Evict if needed BEFORE loading
        await self._ensure_memory_for_load(model_id)
        return await self._load_model(model_id)

    async def _ensure_memory_for_load(self, model_id: str) -> None:
        """Evict LRU models if needed to make room."""
        limit = self._get_effective_memory_limit()
        estimated_size = self._estimate_model_size(model_id)  # From config or name heuristic

        async with self._lock:
            while (self._current_memory_gb() + estimated_size > limit
                   and self._evictable_models()):
                await self._evict_lru()

            # After eviction, check if we have room
            if self._current_memory_gb() + estimated_size > limit:
                available = limit - self._current_memory_gb()
                raise HTTPException(
                    status_code=503,
                    detail=f"Insufficient memory: need {estimated_size:.1f}GB, "
                           f"only {available:.1f}GB available after eviction"
                )

    def _evictable_models(self) -> list[LoadedModel]:
        """Models that can be evicted (not preloaded)."""
        return [m for m in self._models.values() if not m.preloaded]

    async def _evict_lru(self) -> None:
        """Evict the least-recently-used non-preloaded model."""
        evictable = self._evictable_models()
        if not evictable:
            return
        lru = min(evictable, key=lambda m: m.last_used)
        del self._models[lru.model_id]
        clear_cache()
        logger.info(f"Evicted LRU model: {lru.model_id}")

    async def preload_model(self, model_id: str) -> LoadedModel:
        """Load a model and mark it as protected from eviction."""
        loaded = await self.get_model(model_id)
        loaded.preloaded = True
        return loaded
```

### Pattern 2: Vision Model Loading and Image Processing

**What:** Vision models use mlx-vlm which returns a (model, processor) tuple -- not (model, tokenizer). Image preprocessing happens BEFORE model inference.

**When to use:** When model_type is detected as "vision".

**Example:**
```python
# Source: https://github.com/Blaizzy/mlx-vlm
from mlx_vlm import load as load_vlm, generate as vlm_generate
from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
from mlx_vlm.utils import load_config

# Loading vision model -- NOTE: returns (model, processor), not (model, tokenizer)
model, processor = load_vlm("mlx-community/Qwen2-VL-2B-Instruct-4bit")
config = load_config("mlx-community/Qwen2-VL-2B-Instruct-4bit")

# Chat template for VLM requires num_images parameter
formatted_prompt = vlm_apply_chat_template(
    processor, config, prompt_text, num_images=len(images)
)

# Generate with images (accepts local paths, URLs, or PIL Images)
output = vlm_generate(
    model, processor, formatted_prompt, images,
    max_new_tokens=512, temperature=0.7, verbose=False
)
```

### Pattern 3: Image Preprocessing Service

**What:** Images arrive as base64 inline, URLs, or local paths. Preprocess: fetch URLs, decode base64, resize large images, validate format.

**When to use:** Before passing images to mlx-vlm generate.

**Example:**
```python
# Source: Pillow + httpx patterns
import base64
from io import BytesIO
from PIL import Image
import httpx

MAX_IMAGE_DIMENSION = 2048  # Configurable per Claude's Discretion
URL_FETCH_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
MAX_URL_RETRIES = 3

async def preprocess_image(
    image_input: str,  # base64 data URI, URL, or file path
    client: httpx.AsyncClient,
) -> Image.Image:
    """Fetch, decode, and resize image."""
    if image_input.startswith("data:"):
        # Base64 data URI: "data:image/png;base64,<data>"
        _, data = image_input.split(",", 1)
        img_bytes = base64.b64decode(data)
        img = Image.open(BytesIO(img_bytes))
    elif image_input.startswith(("http://", "https://")):
        # URL fetch with retry
        img = await _fetch_image_from_url(image_input, client)
    else:
        # Local file path
        img = Image.open(image_input)

    # Auto-resize if exceeds max dimension
    max_dim = max(img.size)
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
        logger.warning(
            f"Image resized from {max_dim}px to {max(new_size)}px "
            f"(max dimension: {MAX_IMAGE_DIMENSION}px)"
        )

    return img


async def _fetch_image_from_url(
    url: str, client: httpx.AsyncClient
) -> Image.Image:
    """Fetch image from URL with retry and timeout."""
    for attempt in range(MAX_URL_RETRIES):
        try:
            response = await client.get(url, timeout=URL_FETCH_TIMEOUT)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt == MAX_URL_RETRIES - 1:
                raise ValueError(f"Failed to fetch image from {url}: {e}") from e
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    # Should not reach here, but satisfy type checker
    raise ValueError(f"Failed to fetch image after {MAX_URL_RETRIES} attempts")
```

### Pattern 4: Embeddings Generation (Batch, L2-Normalized)

**What:** Accept array of strings, return array of embeddings in OpenAI format. mlx-embeddings already returns L2-normalized embeddings via `text_embeds`.

**When to use:** For the /v1/embeddings endpoint.

**Example:**
```python
# Source: https://github.com/Blaizzy/mlx-embeddings
from mlx_embeddings.utils import load as load_embeddings

# Load embedding model -- NOTE: different API from mlx-lm
model, tokenizer = load_embeddings("mlx-community/all-MiniLM-L6-v2-4bit")

# Batch encoding
texts = ["Hello world", "Machine learning is great"]
inputs = tokenizer.batch_encode_plus(
    texts,
    return_tensors="mlx",
    padding=True,
    truncation=True,
    max_length=512,
)

# Forward pass
outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

# text_embeds are ALREADY L2-normalized (mean pooled + normalized)
embeddings = outputs.text_embeds  # shape: (batch_size, embedding_dim)

# Convert to list for JSON response
embeddings_list = embeddings.tolist()
```

### Pattern 5: OpenAI Embeddings Response Format

**What:** Return embeddings in exact OpenAI format with data array and usage stats.

**When to use:** /v1/embeddings endpoint response.

**Example (Pydantic schemas):**
```python
# Source: https://platform.openai.com/docs/api-reference/embeddings
from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    input: str | list[str]  # Single string or batch
    model: str
    # encoding_format: Literal["float"] = "float"  # Only float for now

class EmbeddingData(BaseModel):
    embedding: list[float]
    index: int
    object: str = "embedding"

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    data: list[EmbeddingData]
    model: str
    object: str = "list"
    usage: EmbeddingUsage
```

### Pattern 6: Vision Content Blocks in Chat Messages

**What:** OpenAI-compatible vision chat uses content arrays with text and image_url blocks.

**When to use:** Extending the existing ChatCompletionRequest for multimodal.

**Example (extending existing schemas):**
```python
from typing import Union

class ImageURL(BaseModel):
    url: str  # Can be data:image/... base64 URI or http(s) URL

class ImageContentBlock(BaseModel):
    type: str = "image_url"
    image_url: ImageURL

class TextContentBlock(BaseModel):
    type: str = "text"
    text: str

ContentBlock = Union[TextContentBlock, ImageContentBlock]

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentBlock]  # String OR array of content blocks
```

### Anti-Patterns to Avoid

- **Loading mlx-vlm as mlx-lm:** `mlx_vlm.load()` returns `(model, processor)` while `mlx_lm.load()` returns `(model, tokenizer)`. Different APIs, different chat template functions. Never mix them.
- **Ignoring model type for adapter selection:** Must first determine model_type (text-gen/vision/embeddings), THEN select family adapter. A Qwen text model and Qwen VL model need different loading paths.
- **Evicting preloaded models:** Admin preload is a signal the user wants this model always available. Never evict preloaded models; if memory is insufficient, return 503.
- **Skipping image resize:** Large images (>2048px) slow down vision models significantly and can OOM. Always cap max dimension.
- **Blocking URL fetch in async handler:** Use `httpx.AsyncClient`, not `requests.get()`. The event loop serves other requests.
- **Not validating model type against capability:** Text-only model receiving image request must return 400, not silently ignore images.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Image resize + format handling | Custom resize logic | Pillow (`Image.resize`, `LANCZOS`) | Handles all image formats, anti-aliasing, color space conversion |
| Async HTTP client with retries | Custom retry loop | httpx `AsyncClient` + transport retries | Connection pooling, TLS reuse, timeout granularity |
| Embedding normalization | Custom numpy L2 norm | mlx-embeddings `text_embeds` | Already returns mean-pooled + L2-normalized |
| Vision chat template formatting | Manual string concatenation | `mlx_vlm.prompt_utils.apply_chat_template()` | Handles vision token injection, model-specific formatting |
| Memory usage tracking | Custom memory polling | `mx.get_active_memory()`, `mx.get_cache_memory()` | Accurate Metal memory stats (existing in memory.py) |
| Model type detection from config | Custom pattern matching | Extend existing `detect_multimodal()` from model_detection.py | Already handles vision_config, image_token_id, architecture patterns |
| LRU ordering | Custom sorted structure | Python `time.time()` + `min(key=last_used)` on dict values | For 4-10 models, linear scan is faster than maintaining a heap |

**Key insight:** The Phase 7 codebase already contains significant infrastructure (model_detection.py with `detect_multimodal`, adapter registry with `register_adapter`, memory.py with all MLX memory APIs). Phase 8 should EXTEND these, not rebuild.

## Common Pitfalls

### Pitfall 1: MLX-VLM vs MLX-LM API Confusion

**What goes wrong:** Calling `mlx_lm.load()` on a vision model, or passing a vision model's `processor` to `mlx_lm.stream_generate()`.

**Why it happens:** Both libraries have similar names and `load()` functions but completely different return types and generation APIs.

**How to avoid:**
- Model type detection MUST run BEFORE loading. Use the detection chain: config.json `vision_config` key -> model name patterns (`-VL`, `-vlm`) -> default to text-gen.
- mlx-vlm uses `(model, processor)` and `mlx_vlm.generate()`. mlx-lm uses `(model, tokenizer)` and `mlx_lm.stream_generate()`.
- Store `model_type` in `LoadedModel` dataclass and validate at the API boundary.

**Warning signs:**
- `TypeError` about unexpected arguments to `generate()`
- Model producing empty or garbage output (wrong chat template applied)
- Memory errors (vision models are typically larger than text models at same parameter count)

### Pitfall 2: Qwen2-VL Chat Template Missing Vision Tokens

**What goes wrong:** Qwen2-VL/Qwen2.5-VL chat template in `tokenizer_config.json` does NOT include vision token handling. Using it directly produces broken prompts.

**Why it happens:** Qwen models ship two chat templates -- one in `tokenizer_config.json` (text-only) and one in `chat_template.json` (full multimodal). The default tokenizer loads the wrong one.

**How to avoid:**
```python
# WRONG: Using default tokenizer chat template for vision
prompt = tokenizer.apply_chat_template(messages, ...)

# CORRECT: Use mlx-vlm's apply_chat_template which handles vision tokens
from mlx_vlm.prompt_utils import apply_chat_template
formatted = apply_chat_template(processor, config, text, num_images=len(images))
```

**Warning signs:**
- Vision model ignores images completely (just does text completion)
- Qwen2.5-VL reporting "missing vision tokens" errors

### Pitfall 3: Qwen Family Stop Token Configuration

**What goes wrong:** Qwen models use `<|im_end|>` as the end-of-turn token, not `<|eot_id|>`. Using Llama's stop tokens with Qwen causes runaway generation.

**Why it happens:** Each model family has its own special token vocabulary. Qwen uses ChatML format with `<|im_start|>` / `<|im_end|>` markers.

**How to avoid:**
```python
class QwenAdapter:
    @property
    def family(self) -> str:
        return "qwen"

    def get_stop_tokens(self, tokenizer: Any) -> list[int]:
        stop_tokens = [tokenizer.eos_token_id]
        # Qwen uses <|im_end|> as the end-of-turn token (ChatML format)
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
                stop_tokens.append(im_end_id)
        except Exception:
            pass
        return stop_tokens
```

### Pitfall 4: Mistral System Message Handling

**What goes wrong:** Passing system messages to older Mistral models causes template errors because Mistral v1/v2 doesn't natively support system role.

**Why it happens:** Mistral's instruct format uses `[INST] ... [/INST]` which has no system message slot in older versions.

**How to avoid:**
```python
class MistralAdapter:
    def apply_chat_template(self, tokenizer, messages, add_generation_prompt=True):
        # Mistral v1/v2: prepend system message to first user message
        processed = list(messages)
        if processed and processed[0].get("role") == "system":
            system_content = processed[0]["content"]
            processed = processed[1:]
            if processed and processed[0].get("role") == "user":
                processed[0] = {
                    "role": "user",
                    "content": f"{system_content}\n\n{processed[0]['content']}"
                }
        # Fall through to tokenizer template (works for v3+)
        return tokenizer.apply_chat_template(
            processed, add_generation_prompt=add_generation_prompt, tokenize=False
        )

    def get_stop_tokens(self, tokenizer):
        # Mistral uses </s> as EOS
        return [tokenizer.eos_token_id]
```

### Pitfall 5: Memory Estimation Before Load

**What goes wrong:** Eviction decisions are based on total active memory, but model size is unknown until AFTER loading. Loading a larger model than expected triggers OOM.

**Why it happens:** MLX active memory only reports what's currently allocated. Pre-load estimation requires heuristics.

**How to avoid:**
- Estimate model size from config.json `num_parameters` * bytes_per_param (4-bit = 0.5 bytes, 8-bit = 1 byte, bf16 = 2 bytes)
- Fall back to name-based heuristics: `3B-4bit` = ~2GB, `7B-4bit` = ~4GB, `13B-4bit` = ~8GB
- Record actual size after first load and cache it for subsequent load cycles
- Use `relaxed=True` on memory limit to allow temporary overages during loading

### Pitfall 6: Concurrent Model Load Race Condition

**What goes wrong:** Two requests for the same unloaded model trigger two parallel loads, wasting memory and potentially corrupting state.

**Why it happens:** Phase 7's `_loading` dict with `asyncio.Event` handles this for text models. The pattern must extend to vision and embedding models.

**How to avoid:**
- The existing `_loading` Event pattern in pool.py is correct -- ensure ALL model types (text, vision, embeddings) use the same pool and the same `_loading` coordination.
- Never bypass the pool for any model type.

## Code Examples

Verified patterns from official sources:

### Model Type Detection Chain
```python
# Source: Extends existing detect_multimodal() from backend/mlx_manager/utils/model_detection.py
from enum import Enum

class ModelType(Enum):
    TEXT_GEN = "text-gen"
    VISION = "vision"
    EMBEDDINGS = "embeddings"

def detect_model_type(model_id: str, config: dict | None = None) -> ModelType:
    """Detect model type using the decision chain:
    1. config.json fields (most reliable)
    2. Model name patterns (fallback)
    3. Default to text-gen
    """
    if config is None:
        config = read_model_config(model_id)  # Reuse existing utility

    if config:
        # Vision: has vision_config or image_token_id
        if "vision_config" in config or "image_token_id" in config:
            return ModelType.VISION
        # Check model_type for vision indicators
        model_type = config.get("model_type", "").lower()
        if any(ind in model_type for ind in ("vl", "vision", "multimodal")):
            return ModelType.VISION
        # Embeddings: specific architectures
        arch = config.get("architectures", [""])[0].lower()
        if any(ind in arch for ind in ("embedding", "sentence", "bert", "roberta")):
            return ModelType.EMBEDDINGS

    # Name-based fallback
    name_lower = model_id.lower()
    if any(ind in name_lower for ind in ("-vl", "vlm", "vision", "qwen2-vl", "qwen2.5-vl")):
        return ModelType.VISION
    if any(ind in name_lower for ind in ("embed", "minilm", "sentence", "e5")):
        return ModelType.EMBEDDINGS

    return ModelType.TEXT_GEN  # Safe default
```

### Admin Endpoints for Preload/Unload
```python
# Source: FastAPI pattern extending existing admin structure
from fastapi import APIRouter, HTTPException
from mlx_manager.mlx_server.models.pool import get_model_pool

admin_router = APIRouter(prefix="/admin", tags=["admin"])

@admin_router.post("/models/load/{model_id:path}")
async def preload_model(model_id: str) -> dict:
    """Preload a model into the pool (protected from eviction)."""
    pool = get_model_pool()
    try:
        loaded = await pool.preload_model(model_id)
        return {
            "status": "loaded",
            "model_id": model_id,
            "model_type": loaded.model_type,
            "size_gb": loaded.size_gb,
            "preloaded": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.post("/models/unload/{model_id:path}")
async def unload_model(model_id: str) -> dict:
    """Unload a model from the pool."""
    pool = get_model_pool()
    success = await pool.unload_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model not loaded: {model_id}")
    return {"status": "unloaded", "model_id": model_id}

@admin_router.get("/models/status")
async def pool_status() -> dict:
    """Get current model pool status."""
    pool = get_model_pool()
    from mlx_manager.mlx_server.utils.memory import get_memory_usage
    models = {
        model_id: {
            "model_type": m.model_type,
            "size_gb": m.size_gb,
            "preloaded": m.preloaded,
            "last_used": m.last_used,
        }
        for model_id, m in pool._models.items()
    }
    return {
        "loaded_models": models,
        "memory": get_memory_usage(),
        "max_memory_gb": pool.max_memory_gb,
    }
```

### Embedding Endpoint Implementation Skeleton
```python
# Source: OpenAI Embeddings API format
# https://platform.openai.com/docs/api-reference/embeddings
from fastapi import APIRouter, HTTPException

embeddings_router = APIRouter(prefix="/v1", tags=["embeddings"])

@embeddings_router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for input text(s)."""
    # Normalize input to list
    inputs = [request.input] if isinstance(request.input, str) else request.input

    pool = get_model_pool()
    loaded = await pool.get_model(request.model)

    if loaded.model_type != "embeddings":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is type '{loaded.model_type}', not 'embeddings'"
        )

    # Run embedding in thread (MLX Metal thread affinity)
    embeddings, token_count = await _generate_embeddings_threaded(
        loaded.model, loaded.tokenizer, inputs
    )

    return EmbeddingResponse(
        data=[
            EmbeddingData(embedding=emb, index=i)
            for i, emb in enumerate(embeddings)
        ],
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=token_count, total_tokens=token_count),
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate model pools per type | Single unified pool with model_type field | 2024-2025 (LM Studio pattern) | Simpler memory management, single LRU policy |
| VLM as standalone server | VisionAddOn modular pattern | 2025 (LM Studio mlx-engine) | Enables prompt caching for text-only queries to VLMs |
| Manual chat template per family | tokenizer.apply_chat_template() + family adapters | 2024 | Maintainable, handles new models without code changes |
| requests (sync) for image fetch | httpx AsyncClient | 2024 | Non-blocking, connection pooling |
| Raw numpy embedding normalization | mlx-embeddings text_embeds (pre-normalized) | 2024 | Zero-copy, Metal-accelerated |

**Deprecated/outdated:**
- **requests.get() for image URLs in async handlers:** Use httpx.AsyncClient to avoid blocking the event loop.
- **Manual SSE for streaming vision responses:** Use sse-starlette (Phase 7 already established this pattern).
- **Loading vision models with mlx-lm load():** Must use mlx-vlm.load() for VLMs.
- **Qwen tokenizer_config.json chat template for vision:** Use mlx-vlm's apply_chat_template() which loads the correct template.

## Open Questions

Things that couldn't be fully resolved:

1. **mlx-vlm streaming generation**
   - What we know: mlx-vlm has a `generate()` function (non-streaming). The FastAPI server it ships supports streaming.
   - What's unclear: Whether mlx-vlm exposes a `stream_generate()` equivalent at the Python API level, or whether streaming must be implemented manually by iterating token-by-token.
   - Recommendation: Check mlx-vlm source for stream_generate. If absent, use the existing queue-based threading pattern from Phase 7 to stream tokens from a single `generate()` call, or implement token-by-token generation using mlx-vlm internals.

2. **Model size estimation accuracy**
   - What we know: `mx.get_active_memory()` reports total active, not per-model. Config.json has `num_parameters` for some models.
   - What's unclear: Reliable pre-load size estimation without downloading config.json first.
   - Recommendation: Record actual size deltas (active_memory_before vs after load) on first load. Cache in a simple JSON file for subsequent estimations. Fall back to name-based heuristics.

3. **VisionAddOn vs mlx-vlm for Phase 8**
   - What we know: LM Studio's VisionAddOn pattern (image embeddings fed into mlx-lm text model) enables prompt caching. Currently supports only Gemma3/Pixtral. Qwen2.5VL extension is in progress but subpar OCR.
   - What's unclear: Maturity for production use across model families.
   - Recommendation: Use mlx-vlm.generate() for Phase 8 (reliable, all VLM families supported). Consider VisionAddOn migration in Phase 9 or 10 for performance optimization.

4. **Embeddings model token counting**
   - What we know: mlx-embeddings tokenizer has `encode()` and `batch_encode_plus()`.
   - What's unclear: Whether token count from `batch_encode_plus` accounts for padding tokens or only meaningful tokens per input.
   - Recommendation: Count tokens per input individually (not from padded batch) to report accurate usage stats.

## Sources

### Primary (HIGH confidence)
- [mlx-vlm GitHub](https://github.com/Blaizzy/mlx-vlm) - Official VLM library, verified load/generate API
- [mlx-embeddings GitHub](https://github.com/Blaizzy/mlx-embeddings) - Official embeddings library, verified batch API and normalization
- [OpenAI Embeddings API Reference](https://platform.openai.com/docs/api-reference/embeddings) - Response format specification
- [MLX Memory APIs](https://ml-explore.github.io/mlx/build/html/python/) - get_active_memory, clear_cache, set_memory_limit
- [Mistral Tokenization Cookbook](https://docs.mistral.ai/cookbooks/concept-deep-dive-tokenization-chat_templates) - Chat template format per version
- [LM Studio Unified MLX Engine](https://lmstudio.ai/blog/unified-mlx-engine) - VisionAddOn pattern
- Phase 7 codebase: pool.py, inference.py, adapters/, memory.py, model_detection.py - Verified directly

### Secondary (MEDIUM confidence)
- [Qwen2.5-VL Chat Template Issue](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/discussions/11) - Missing vision tokens in tokenizer_config.json
- [Qwen2.5-VL failing in mlx-vlm](https://github.com/Blaizzy/mlx-vlm/issues/192) - Compatibility issues with apply_chat_template
- [mlx-engine VisionAddOn Qwen Extension](https://github.com/lmstudio-ai/mlx-engine/issues/167) - Pattern extension status
- [httpx Async Patterns](https://www.python-httpx.org/async/) - Official async client documentation
- [Gemma 3 Model Docs](https://huggingface.co/blog/gemma3) - Stop tokens and chat format

### Tertiary (LOW confidence - needs validation)
- mlx-vlm streaming generation at Python API level (not confirmed in docs, inferred from FastAPI server behavior)
- Pre-load model size estimation accuracy (heuristic-based, needs empirical validation)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries verified via official repos
- Architecture patterns: HIGH - pool.py extension is mechanical, adapter pattern is proven
- Vision integration: MEDIUM - mlx-vlm API verified, streaming path needs validation
- Embeddings integration: HIGH - batch API and normalization confirmed in official docs
- Model family adapters: HIGH - stop tokens and templates verified per family
- LRU eviction logic: HIGH - straightforward extension of existing pool with well-understood semantics

**Research date:** 2026-01-28
**Valid until:** ~30 days (mlx-vlm updates frequently, check for breaking changes)

**Recommended next steps for planner:**
1. Extend LoadedModel with model_type field and preloaded flag
2. Implement LRU eviction in ModelPoolManager (the skeleton is already there)
3. Add model type detection in the pool's _load_model (reuse detect_multimodal from model_detection.py)
4. Create QwenAdapter, MistralAdapter, GemmaAdapter following LlamaAdapter pattern
5. Add EmbeddingRequest/Response schemas to openai.py
6. Extend ChatMessage.content to support content blocks (vision)
7. Create image_processor.py service for URL fetch + base64 decode + resize
8. Add /v1/embeddings and /admin/models/* routers
9. Validate mlx-vlm streaming capability before implementing vision streaming
