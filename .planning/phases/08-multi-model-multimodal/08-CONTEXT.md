# Phase 8: Multi-Model & Multimodal Support - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend the Phase 7 single-model server to support multiple hot models with LRU eviction, vision models via mlx-vlm, embeddings via mlx-embeddings, and additional model family adapters (Qwen, Mistral, Gemma). Admin endpoints for explicit model preload/unload.

Model types for this phase: Text-generation, Vision, Embeddings. Audio and other types are future phases.

</domain>

<decisions>
## Implementation Decisions

### Memory Eviction Policy
- Eviction triggers **before loading a new model** — check available memory, evict LRU models if needed to fit
- Preloaded models are **protected from eviction** — only on-demand loaded models can be evicted
- If eviction can't free enough memory: **return 503 error** with "need X GB, only Y GB available after eviction"
- Memory limits configurable as **either percentage OR absolute GB** — user chooses which format

### Vision Model Handling
- Images accepted as **both base64 inline AND URL** — server fetches URLs when provided
- Large images are **auto-resized** to max dimension (e.g., 2048px) with warning logged
- **Multiple images per message** supported — array of images in content, processed in order
- Text-only model receiving image request: **return 400 error** with "Model X does not support vision"

### Embeddings API
- **Batch requests supported** — accept array of strings, return array of embeddings
- Embeddings are **L2-normalized by default** — ready for cosine similarity
- Response format: **OpenAI-compatible JSON** — `{data: [{embedding: [...], index: 0}], usage: {...}}`
- Dimension reduction: **not for now** — return full dimensions, defer truncation support

### Model Family Detection
- Detection chain: **config.json → model name parsing → default**
- First determine **model type** (text-gen, vision, embeddings), then detect **family within type**
- For text-generation models, Llama-style is reasonable default
- Architecture must support future model types (audio, image manipulation) even if not implemented now

### Claude's Discretion
- Exact max image dimension for resize (2048px is suggestion)
- URL fetch timeout and retry behavior
- config.json field names to check for architecture detection
- Model name patterns for family inference

</decisions>

<specifics>
## Specific Ideas

- Model detection needs a fallback chain strategy because there's no standard — some models might not have config.json
- The detection architecture should be extensible for future model types (t2s, s2t, audio manipulation, image manipulation) even though Phase 8 only implements text-gen, vision, embeddings

</specifics>

<deferred>
## Deferred Ideas

- Audio model support (text-to-speech, speech-to-text) — future phase
- Image manipulation models — future phase
- Matryoshka/dimension reduction for embeddings — future enhancement

</deferred>

---

*Phase: 08-multi-model-multimodal*
*Context gathered: 2026-01-28*
