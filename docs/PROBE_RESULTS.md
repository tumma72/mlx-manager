# Model Probe Results

*Generated: 2026-02-08 | Probe system: v2 (strategy pattern)*

## Summary

Probed 40 cached models using `mlx-manager probe --all`. Results reveal **5 categories of issues** that need architectural attention.

## Full Model Inventory

### Text Generation Models (19 models)

| Model | Size | Context | Thinking | Native Tools | Load | Notes |
|-------|------|---------|----------|-------------|------|-------|
| `mlx-community/Qwen3-0.6B-4bit-DWQ` | 0.4GB | 40,960 | Yes | Yes | OK | Reference quick model |
| `mlx-community/Qwen3-4B-4bit` | 2.3GB | 40,960 | Yes | Yes | OK | |
| `mlx-community/Qwen3-30B-A3B-4bit` | 17.2GB | 40,960 | Yes* | Yes* | OK* | MoE, large |
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 0.3GB | 32,768 | No | Yes | OK | |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 3.6GB | 131,072 | No | No | OK | Reasoning distill |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | 0.7GB | 131,072 | No | Yes | OK | |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 1.8GB | 131,072 | No | Yes* | OK* | |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 4.1GB | 32,768 | No* | No* | OK* | |
| `mlx-community/SmolLM-135M-Instruct-4bit` | 0.1GB | 2,048 | No | No | OK | Tiny model |
| `mlx-community/EuroLLM-22B-Instruct-2512-mlx-4bit` | 12.8GB | 32,768 | No* | No* | OK* | |
| `mlx-community/IQuest-Coder-V1-40B-Instruct-4bit` | 22.4GB | 131,072 | No* | No* | OK* | |
| `mlx-community/IQuest-Coder-V1-40B-Loop-Instruct-4bit` | 22.4GB | 131,072 | No* | No* | OK* | |
| `mlx-community/LFM2-2.6B-Exp-4bit` | 1.5GB | 128,000 | No* | Yes* | OK* | Liquid model |
| `lmstudio-community/LFM2-1.2B-MLX-8bit` | 1.2GB | 128,000 | No | Yes | OK | Liquid model |
| `lmstudio-community/LFM2-1.2B-MLX-bf16` | 2.3GB | 128,000 | No | Yes | OK | Liquid model |
| `lmstudio-community/Qwen3-Coder-Next-MLX-4bit` | 44.9GB | 262,144 | No | Yes | OK | Very large MoE |
| **`mlx-community/Llama-3-Groq-8B-Tool-Use-4bit`** | 4.5GB | 8,192 | No | **No** | OK | **Issue #3** |
| **`mlx-community/Llama-3-Groq-70B-Tool-Use-4bit`** | 39.7GB | 8,192 | No* | **No*** | OK* | **Issue #3** |
| **`mlx-community/GLM-4.7-Flash-4bit`** | 33.7GB | 202,752 | ?* | ?* | ?* | **Issue #5**: Misdetected as AUDIO |

*\* = not live-probed (inferred from config or too large to load in CLI session)*

### Vision Models (7 models)

| Model | Size | Context | Multi-Image | Video | Load | Notes |
|-------|------|---------|------------|-------|------|-------|
| `mlx-community/gemma-3-12b-it-qat-4bit` | 8.1GB | 131,072 | ?* | ?* | OK* | |
| `mlx-community/gemma-3-27b-it-4bit-DWQ` | 16.0GB | ?* | ?* | ?* | OK* | Reference full model |
| `mlx-community/gemma-3-27b-it-qat-4bit` | 16.9GB | 131,072 | ?* | ?* | OK* | |
| `mistralai/Devstral-Small-2-24B-Instruct-2512` | 31.8GB | 393,216 | ? | ? | Slow | 24GB load |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit` | 14.1GB | 393,216 | ?* | ?* | OK* | |
| **`lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit`** | 18.3GB | 262,144 | ? | Yes* | **FAIL** | **Issue #4** |
| **`lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit`** | 5.8GB | 262,144 | ? | Yes* | **FAIL** | **Issue #4** |
| **`mlx-community/Qwen3-VL-8B-Instruct-4bit`** | 5.8GB | 262,144 | ? | Yes* | **FAIL** | **Issue #4** |

### Embeddings Models (2 models)

| Model | Size | Context | Dimensions | Normalized | Load | Notes |
|-------|------|---------|-----------|-----------|------|-------|
| `mlx-community/all-MiniLM-L6-v2-4bit` | 0.0GB | 512 | 384 | Yes | OK | Reference model |
| `sentence-transformers/all-MiniLM-L6-v2` | 0.2GB | 512 | 384* | Yes* | OK* | Non-quantized |

### Audio Models (5 models, 2 misdetected)

| Model | Size | Actual Type | Detected Type | TTS | STT | Load | Notes |
|-------|------|------------|--------------|-----|-----|------|-------|
| `mlx-community/Kokoro-82M-4bit` | 0.6GB | Audio | Audio | Yes* | No* | **FAIL** | Pool load issue |
| `prince-canuma/Kokoro-82M` | 0.4GB | Audio | Audio | Yes* | No* | **FAIL** | Pool load issue |
| `mlx-community/whisper-large-v3-turbo` | 1.6GB | Audio | Audio | No* | Yes* | ?* | Whisper STT |
| `mlx-community/Dia-1.6B-4bit` | 3.2GB | Audio | Audio | Yes* | No* | ?* | Dialogue TTS |
| **`mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit`** | 17.8GB | **TEXT_GEN** | **Audio** | - | - | - | **Issue #5** |

### Non-MLX / Unclassifiable Models (5 models)

| Model | Size | Notes |
|-------|------|-------|
| `bert-base-uncased` | 0.0GB | No config.json (non-MLX) |
| `dhairyashil/FLUX.1-schnell-mflux-4bit` | 9.6GB | Image generation (Flux), no config.json |
| `lucasnewman/f5-tts-mlx` | 1.4GB | F5-TTS, no config.json (custom format) |
| `lucasnewman/vocos-mel-24khz` | 0.1GB | Vocoder, no config.json |
| `mlx-community/descript-audio-codec-44khz` | 0.3GB | Audio codec, misdetected as text-gen |
| `mlx-community/snac_24khz` | 0.1GB | Audio codec, misdetected as text-gen |

---

## Issues Found

### Issue #1: Probe UI should use modal dialog
**Severity**: UX
**Status**: Planned (Task #6)

The probe button currently shows progress inline. Users can interact with other parts of the UI while a probe is running, which can cause confusion. The probe progress should be displayed in a modal dialog that blocks interaction until completion.

### Issue #2: Probe button appears for all model types
**Severity**: UX
**Status**: Fixed (strategy pattern implemented)

The probe button now only appears for model types that have a registered probe strategy (text-gen, vision, embeddings, audio). The backend exposes `GET /api/models/probe/supported-types` for the frontend to check.

### Issue #3: Groq Tool-Use models report `native_tools=False`
**Severity**: Functional — false negative
**Affected models**: `Llama-3-Groq-8B-Tool-Use-4bit`, `Llama-3-Groq-70B-Tool-Use-4bit`

**Root cause**: `has_native_tool_support()` in `utils/template_tools.py` checks whether the tokenizer's `apply_chat_template()` accepts a `tools=` parameter. Groq Tool-Use models use the **Hermes function-calling format** (special tokens like `<tool_call>` in the chat template) rather than the native `tools=` parameter. The probe correctly detects they don't support the native tools API, but it doesn't detect Hermes-style tool support.

**Architectural fix needed**: Distinguish between:
- `supports_native_tools`: Tokenizer accepts `tools=` parameter (current check)
- `supports_tool_use`: Broader detection including Hermes format, tool-related special tokens in vocabulary, or model name/tag heuristics

**Detection approach for Hermes format**:
1. Check if chat template contains `<tool_call>`, `<function_call>`, or similar markers
2. Check if tokenizer vocabulary contains tool-related special tokens
3. Check HuggingFace model tags for "tool-use", "function-calling", etc.

### Issue #4: Qwen3-VL models fail to load
**Severity**: Blocking — cannot probe or serve
**Affected models**: All Qwen3-VL variants (3 models)

**Error**: `TypeError: argument of type 'NoneType' is not iterable` in `transformers/models/auto/video_processing_auto.py:96`

**Root cause**: The installed `transformers` version doesn't have `Qwen3VLVideoProcessor` registered in `VIDEO_PROCESSOR_MAPPING_NAMES`. The `video_processor_class_from_name()` function receives `extractors=None` because the mapping doesn't exist yet for this model class.

**Fix options**:
1. **Upgrade transformers**: `pip install transformers>=4.53.0` (or whichever version adds Qwen3-VL video processor support)
2. **Workaround in pool.py**: Catch this specific error and retry with `use_fast=False` or without video processor
3. **Pin to compatible version**: Add minimum transformers version to `pyproject.toml`

### Issue #5: Model type detection false positives
**Severity**: Functional — wrong type assigned
**Affected models**: GLM-4.7-Flash-4bit (text-gen misdetected as audio), NVIDIA-Nemotron-3-Nano (text-gen misdetected as audio), descript-audio-codec (audio codec misdetected as text-gen), snac_24khz (audio codec misdetected as text-gen)

**Root cause — GLM-4.7-Flash**: The `audio_model_type_indicators` tuple contains `"glm"`, which matches `glm4_moe_lite` (a text-gen MoE model). The indicator was intended for GLM audio models but is too broad.

**Root cause — NVIDIA-Nemotron**: The `audio_name_patterns` tuple contains `"dia-"` (for the Dia audio model), which matches `nvi**dia-**nemotron` as a substring.

**Root cause — audio codecs**: `descript-audio-codec-44khz` and `snac_24khz` have no config.json audio indicators and their architectures don't match any audio patterns, so they fall through to the TEXT_GEN default. These are audio infrastructure models (codecs/vocoders) not meant to be served directly.

**Fixes needed**:
1. Change `"glm"` to `"glm4_audio"` or `"glm_tts"` / `"glm_asr"` in `audio_model_type_indicators`
2. Change `"dia-"` to `"/dia-"` or `"dia-1"` in `audio_name_patterns` to avoid matching `nvidia-`
3. Consider adding a `CODEC` or `INFRASTRUCTURE` model type for non-servable audio models, or at minimum exclude them from the probe

### Issue #6: Audio model loading not implemented in pool
**Severity**: Blocking for audio probes
**Affected models**: Kokoro-82M-4bit, prince-canuma/Kokoro-82M

**Root cause**: The model pool's `_load_model()` method handles TEXT_GEN, VISION, and EMBEDDINGS but the AUDIO path (added in 15-13) may not properly handle all audio model formats. The Kokoro models fail during the load phase.

**Status**: Part of Phase 15-13 (Audio integration), tracked separately.

---

## Detection Decision Chain

Current priority order in `detect_model_type()`:

```
1. config.json → detect_multimodal() → VISION (if vision_config present)
2. config.json → audio_config_indicators → AUDIO (audio_config, tts_config, etc.)
3. config.json → audio_arch_indicators → AUDIO (kokoro, whisper, bark, etc.)
4. config.json → audio_model_type_indicators → AUDIO (glm*, dia, etc.)  ← BUG
5. config.json → embedding_indicators → EMBEDDINGS (bert, roberta, etc.)
6. Name patterns → audio_name_patterns → AUDIO (dia-*, etc.)  ← BUG
7. Name patterns → vision_patterns → VISION (-vl, llava, gemma-3, etc.)
8. Name patterns → embed_patterns → EMBEDDINGS (minilm, e5-, bge-, etc.)
9. Default → TEXT_GEN
```

## Recommendations

1. **Fix detection bugs first** (Issue #5) — straightforward string fixes, high impact
2. **Upgrade transformers** (Issue #4) — unblocks all Qwen3-VL models
3. **Add Hermes tool detection** (Issue #3) — improves probe accuracy for tool-use models
4. **Frontend modal** (Issue #1) — UX improvement, already planned
5. **Audio pool integration** (Issue #6) — tracked in Phase 15-13
