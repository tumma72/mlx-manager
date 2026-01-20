# Phase 4: Model Discovery & Badges - Research

**Researched:** 2026-01-20
**Domain:** MLX model metadata extraction, HuggingFace API, frontend badge UI
**Confidence:** HIGH

## Summary

This phase adds model characteristic detection and visual badges to help users understand model capabilities at a glance. The implementation leverages the existing `model_detection.py` infrastructure which already reads local `config.json` files, extending it to extract additional metadata (context window, multimodal support, quantization). For HuggingFace results, configs will be fetched lazily via the `/resolve/` endpoint.

The codebase already has a Badge component with multiple variants and the ModelCard component displays model tags. The architecture change involves: (1) extending backend to return model characteristics, (2) adding a new API endpoint for fetching remote configs, (3) creating badge-specific components, and (4) refactoring the models page UI per CONTEXT.md decisions.

**Primary recommendation:** Extend existing `read_model_config()` to return structured characteristics, add backend endpoint for lazy config fetching, implement badge components that consume this data.

## Standard Stack

### Core (Already in Codebase)
| Library | Purpose | Notes |
|---------|---------|-------|
| FastAPI | Backend API | Existing router at `routers/models.py` |
| httpx | Async HTTP client | Already used in `hf_api.py` for HuggingFace API calls |
| Svelte 5 | Frontend reactivity | Using `$state`, `$derived` runes |
| bits-ui | UI components | Existing Badge component at `components/ui/badge.svelte` |
| lucide-svelte | Icons | Already used throughout the app |

### Supporting
| Library | Purpose | When to Use |
|---------|---------|-------------|
| Tailwind CSS | Styling | Semantic color classes for badge types |

### No New Dependencies Required
The implementation uses existing libraries. No new packages needed.

## Architecture Patterns

### Backend: Model Characteristics Schema

Based on analysis of real MLX model `config.json` files, here's the normalized schema:

```python
# New type in types.py
class ModelCharacteristics(TypedDict, total=False):
    """Extracted model characteristics from config.json."""

    # Architecture
    model_type: str          # e.g., "qwen2", "llama", "mistral", "gemma2"
    architectures: list[str] # e.g., ["LlamaForCausalLM"]
    architecture_family: str # Normalized: "llama", "qwen", "mistral", etc.

    # Context & Size
    max_position_embeddings: int  # Context window (e.g., 32768, 131072)
    num_hidden_layers: int        # Number of transformer layers
    hidden_size: int              # Model dimension
    vocab_size: int               # Vocabulary size

    # Attention
    num_attention_heads: int      # Attention heads
    num_key_value_heads: int      # GQA heads (if present)

    # Quantization (MLX-specific)
    quantization_bits: int | None   # 2, 3, 4, 8, or None for fp16/bf16
    quantization_group_size: int | None

    # Multimodal Detection
    is_multimodal: bool           # True if vision/audio capabilities
    multimodal_type: str | None   # "vision", "vision-audio", None

    # KV Cache
    use_cache: bool               # Whether KV caching is enabled
```

### Config.json Field Mapping

| Characteristic | Config.json Field(s) | Detection Logic |
|----------------|---------------------|-----------------|
| Architecture Family | `model_type`, `architectures[0]` | Extract base name (llama, qwen, mistral, etc.) |
| Context Window | `max_position_embeddings` | Direct mapping |
| Quantization Bits | `quantization.bits` | Direct mapping, null if absent |
| Quantization Group | `quantization.group_size` | Direct mapping |
| Multimodal | `vision_config`, `image_token_id`, model_type contains "vl"/"vision" | Multiple signals |
| KV Cache | `use_cache` | Direct mapping, default true |
| Parameter Count | Estimate from `hidden_size * num_hidden_layers` | Rough approximation |

### Multimodal Detection Logic (HIGH confidence)

Based on analysis of vision models (Qwen2-VL, Llama-3.2-Vision):

```python
def detect_multimodal(config: dict) -> tuple[bool, str | None]:
    """Detect if model supports multimodal input."""

    # Strongest signals
    if "vision_config" in config:
        return True, "vision"

    # Token-based signals
    vision_tokens = [
        "image_token_id", "image_token_index",
        "vision_token_id", "video_token_id"
    ]
    if any(token in config for token in vision_tokens):
        return True, "vision"

    # Architecture-based signals
    model_type = config.get("model_type", "").lower()
    architectures = [a.lower() for a in config.get("architectures", [])]

    vl_patterns = ["vl", "vision", "multimodal"]
    if any(p in model_type for p in vl_patterns):
        return True, "vision"
    if any(any(p in arch for p in vl_patterns) for arch in architectures):
        return True, "vision"

    return False, None
```

### Architecture Family Normalization (HIGH confidence)

```python
ARCHITECTURE_FAMILIES = {
    # model_type -> display family
    "llama": "Llama",
    "qwen2": "Qwen",
    "qwen2_vl": "Qwen",
    "qwen2_moe": "Qwen",
    "mistral": "Mistral",
    "gemma2": "Gemma",
    "phi3": "Phi",
    "mllama": "Llama",  # Llama Vision
    "starcoder2": "StarCoder",
    "deepseek": "DeepSeek",
    "glm": "GLM",
    # Add more as needed
}

def normalize_architecture(config: dict) -> str:
    """Get normalized architecture family name."""
    model_type = config.get("model_type", "").lower()

    # Direct match
    if model_type in ARCHITECTURE_FAMILIES:
        return ARCHITECTURE_FAMILIES[model_type]

    # Partial match
    for key, family in ARCHITECTURE_FAMILIES.items():
        if key in model_type:
            return family

    # Fallback to architectures field
    archs = config.get("architectures", [])
    if archs:
        arch_lower = archs[0].lower()
        for key, family in ARCHITECTURE_FAMILIES.items():
            if key in arch_lower:
                return family

    return "Unknown"
```

### Frontend: Badge Component Structure

```
src/lib/components/models/
├── ModelCard.svelte           # Existing - add badges
├── ModelBadges.svelte         # NEW: Container for all badges
├── badges/
│   ├── ArchitectureBadge.svelte   # Blue badge for architecture family
│   ├── MultimodalBadge.svelte     # Green badge for vision/audio
│   └── QuantizationBadge.svelte   # Purple badge for quantization
└── ModelSpecs.svelte          # NEW: Expandable specs section
```

### Lazy Loading Pattern for HuggingFace Configs

```typescript
// In models page or a dedicated store
interface ModelConfig {
    characteristics: ModelCharacteristics | null;
    loading: boolean;
    error: string | null;
}

class ModelConfigStore {
    configs = $state<Map<string, ModelConfig>>(new Map());

    async fetchConfig(modelId: string): Promise<void> {
        if (this.configs.has(modelId)) return;

        // Set loading state
        this.configs.set(modelId, { characteristics: null, loading: true, error: null });

        try {
            const response = await fetch(`/api/models/config/${encodeURIComponent(modelId)}`);
            const characteristics = await response.json();
            this.configs.set(modelId, { characteristics, loading: false, error: null });
        } catch (e) {
            this.configs.set(modelId, {
                characteristics: null,
                loading: false,
                error: e.message
            });
        }
    }
}
```

### Backend Endpoint for Remote Config Fetching

```python
@router.get("/config/{model_id:path}")
async def get_model_config(
    current_user: Annotated[User, Depends(get_current_user)],
    model_id: str,
):
    """
    Get model characteristics from config.json.

    For local models: reads from HuggingFace cache.
    For remote models: fetches via HuggingFace resolve API.

    Returns normalized ModelCharacteristics.
    """
    # Try local first
    characteristics = extract_characteristics_local(model_id)
    if characteristics:
        return characteristics

    # Fetch remote
    characteristics = await fetch_characteristics_remote(model_id)
    return characteristics
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Badge styling | Custom CSS classes | Existing Badge component variants | Already has `default`, `secondary`, `outline`, `success`, `warning` variants |
| Icon selection | Custom SVGs | lucide-svelte | Consistent with existing icons (Eye, Cpu, Layers, etc.) |
| Expandable sections | Custom accordion | Svelte transitions + CSS | Simple show/hide with `slide` transition is sufficient |
| HTTP requests | fetch wrapper | Existing `handleResponse` utility | Already handles auth, errors, JSON parsing |

## Common Pitfalls

### Pitfall 1: Missing or Incomplete config.json
**What goes wrong:** Some models may not have config.json or may have non-standard fields.
**Why it happens:** MLX models come from various sources with different conventions.
**How to avoid:**
- Always provide fallback values
- Use `total=False` in TypedDict
- Display "Unknown" badges gracefully or hide them
**Warning signs:** Empty badge rows, broken UI

### Pitfall 2: HuggingFace Rate Limiting
**What goes wrong:** 429 errors when fetching many configs quickly.
**Why it happens:** HuggingFace limits API requests (500/5min anonymous, 1000/5min free user).
**How to avoid:**
- Use `/resolve/` endpoint (higher limits: 3000-5000/5min)
- Implement request queuing with delays
- Cache responses aggressively (configs rarely change)
- Consider batching visible models only
**Warning signs:** Console errors, badges never loading

### Pitfall 3: Architecture Name Inconsistency
**What goes wrong:** Same model family shows different badge names.
**Why it happens:** model_type varies (e.g., "qwen2", "qwen2_vl", "qwen2_moe").
**How to avoid:** Normalize to family names using mapping table.
**Warning signs:** Too many unique architecture badges

### Pitfall 4: Large Config Files
**What goes wrong:** Some configs include tokenizer or extra metadata, making them large.
**Why it happens:** config.json format varies.
**How to avoid:** Only fetch config.json, not tokenizer_config.json. Set reasonable timeout.
**Warning signs:** Slow badge loading for certain models

### Pitfall 5: Reactive State for Lazy Loading
**What goes wrong:** Badges don't update when config loads.
**Why it happens:** Map mutations don't trigger Svelte reactivity.
**How to avoid:** Create new Map on each update (as done in downloadsStore).
**Warning signs:** Badges stuck on "loading" state

## Code Examples

### Fetching Remote config.json (Backend)

```python
# Source: HuggingFace resolve API pattern
async def fetch_remote_config(model_id: str) -> dict | None:
    """Fetch config.json from HuggingFace Hub."""
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch config for {model_id}: {e}")

    return None
```

### Badge Component Example

```svelte
<!-- ArchitectureBadge.svelte -->
<script lang="ts">
    import { Badge } from '$components/ui';
    import { Cpu } from 'lucide-svelte';

    interface Props {
        family: string;
    }

    let { family }: Props = $props();
</script>

<Badge variant="outline" class="bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-900/30 dark:text-blue-300">
    <Cpu class="w-3 h-3 mr-1" />
    {family}
</Badge>
```

### Expandable Specs Section

```svelte
<!-- ModelSpecs.svelte -->
<script lang="ts">
    import { slide } from 'svelte/transition';
    import { ChevronDown, ChevronUp } from 'lucide-svelte';

    interface Props {
        characteristics: ModelCharacteristics;
    }

    let { characteristics }: Props = $props();
    let expanded = $state(false);
</script>

<button
    class="text-sm text-muted-foreground hover:text-foreground"
    onclick={() => expanded = !expanded}
>
    {#if expanded}
        <ChevronUp class="w-4 h-4 inline" /> Hide specs
    {:else}
        <ChevronDown class="w-4 h-4 inline" /> Show specs
    {/if}
</button>

{#if expanded}
    <div transition:slide class="mt-2 text-sm space-y-1">
        {#if characteristics.max_position_embeddings}
            <div>Context: {characteristics.max_position_embeddings.toLocaleString()} tokens</div>
        {/if}
        {#if characteristics.num_hidden_layers}
            <div>Layers: {characteristics.num_hidden_layers}</div>
        {/if}
        <!-- More fields -->
    </div>
{/if}
```

### Filter Modal Component Structure

```svelte
<!-- FilterModal.svelte -->
<script lang="ts">
    import * as Dialog from 'bits-ui/dialog';
    import { Badge } from '$components/ui';

    interface Props {
        open: boolean;
        onClose: () => void;
        onApply: (filters: FilterState) => void;
    }

    interface FilterState {
        architectures: string[];
        multimodal: boolean | null;  // null = any, true = only multimodal, false = text-only
        quantization: number[];       // [4, 8] = 4bit or 8bit
    }

    let filters = $state<FilterState>({
        architectures: [],
        multimodal: null,
        quantization: []
    });
</script>

<Dialog.Root bind:open>
    <Dialog.Content>
        <Dialog.Title>Filter Models</Dialog.Title>

        <!-- Architecture section -->
        <section>
            <h4>Architecture</h4>
            <!-- Checkbox list -->
        </section>

        <!-- Multimodal section -->
        <section>
            <h4>Capabilities</h4>
            <!-- Radio: Any / Text-only / Multimodal -->
        </section>

        <!-- Quantization section -->
        <section>
            <h4>Quantization</h4>
            <!-- Checkbox: 2-bit, 3-bit, 4-bit, 8-bit, fp16 -->
        </section>

        <Dialog.Close>Apply</Dialog.Close>
    </Dialog.Content>
</Dialog.Root>
```

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Manual model type selection | Auto-detect from config.json | Reduces user error |
| No size/architecture visibility | Badges at glance | Better model selection |
| Static checkbox filters | Modal with grouped filters | Cleaner UI, more options |

## Open Questions

1. **Caching Strategy for Remote Configs**
   - What we know: HuggingFace has rate limits (3000 resolves/5min for free users)
   - What's unclear: Should we cache in backend (DB/memory) or just frontend?
   - Recommendation: Frontend cache (sessionStorage) for session, no backend persistence. Configs rarely change.

2. **Filter Persistence**
   - What we know: CONTEXT.md says toggle defaults to "My Models"
   - What's unclear: Should filter selections persist across sessions?
   - Recommendation: Start without persistence, add localStorage later if requested.

3. **Unknown Architecture Handling**
   - What we know: Not all models fit known families
   - What's unclear: Show "Unknown" badge or hide architecture badge entirely?
   - Recommendation: Show "Unknown" - transparency helps users.

## Sources

### Primary (HIGH confidence)
- HuggingFace Hub config.json files:
  - `mlx-community/Qwen2.5-7B-Instruct-4bit`
  - `mlx-community/Qwen2-VL-7B-Instruct-4bit` (vision model)
  - `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit` (vision model)
  - `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
  - `mlx-community/gemma-2-9b-it-4bit`
- HuggingFace Rate Limits documentation: https://huggingface.co/docs/hub/en/rate-limits

### Secondary (MEDIUM confidence)
- Existing codebase analysis:
  - `/backend/mlx_manager/utils/model_detection.py` - existing config reading
  - `/backend/mlx_manager/services/hf_api.py` - existing HF API patterns
  - `/frontend/src/lib/components/ui/badge.svelte` - existing badge component
  - `/frontend/src/lib/stores/downloads.svelte.ts` - Map reactivity pattern

## Metadata

**Confidence breakdown:**
- Config.json schema: HIGH - Verified with multiple real model configs
- Multimodal detection: HIGH - Verified vision_config pattern in multiple VL models
- Architecture normalization: HIGH - model_type field is reliable
- HuggingFace API: HIGH - Documented rate limits, existing httpx usage
- Frontend patterns: HIGH - Existing codebase patterns confirmed

**Research date:** 2026-01-20
**Valid until:** 2026-03-20 (60 days - config.json schema is stable)
