---
status: complete
phase: 08-multi-model-multimodal
source: [08-01-SUMMARY.md, 08-02-SUMMARY.md, 08-03-SUMMARY.md, 08-04-SUMMARY.md, 08-05-SUMMARY.md, 08-06-SUMMARY.md]
started: 2026-01-28T13:00:00Z
updated: 2026-01-28T15:30:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Multi-Model Hot-Swapping
expected: Load two different models via /v1/chat/completions. Server holds both in memory simultaneously. Both respond correctly.
result: pass

### 2. LRU Eviction Under Memory Pressure
expected: When memory limit is reached, least-recently-used non-preloaded model is automatically evicted to make room for new model.
result: skipped
reason: Could not reach memory limit with available models to trigger eviction. Note: Separate issue observed - models already in HF cache are being re-downloaded (Phase 7 issue, not Phase 8).

### 3. Qwen Model Stop Tokens
expected: Qwen model (e.g., Qwen2.5) generates complete responses and stops properly without runaway generation.
result: pass

### 4. Mistral Model Stop Tokens
expected: Mistral model generates complete responses and stops properly without runaway generation.
result: pass

### 5. Gemma Model Stop Tokens
expected: Gemma model generates complete responses and stops properly without runaway generation.
result: issue
reported: "Gemma 3 vision model (gemma-3-12b-it-qat-4bit) returns 500 error: 'Gemma3Processor' object has no attribute 'eos_token_id'. Model detected as type=vision, adapter expects tokenizer but gets Processor."
severity: major

### 6. Vision Model Chat Request
expected: Vision model (e.g., LLaVA, Qwen-VL) accepts image input via base64 or URL in chat request and generates description or answer about the image.
result: issue
reported: "Two issues: (1) Wikipedia image URL returns 403 Forbidden - missing User-Agent header. (2) Qwen3VLProcessor has no eos_token_id - same Processor vs Tokenizer issue as test 5."
severity: major

## Summary

total: 6
passed: 3
issues: 2
pending: 0
skipped: 1

## Gaps

- truth: "Vision models work with all adapters (Gemma, Qwen, etc.)"
  status: failed
  reason: "All vision models use Processor objects instead of Tokenizers. Adapters assume tokenizer.eos_token_id exists but Processors wrap tokenizers differently."
  severity: major
  test: 5, 6
  root_cause: "All adapter get_stop_tokens() methods assume tokenizer has eos_token_id, but vision models use Processor which wraps tokenizer"
  artifacts:
    - path: "backend/mlx_manager/mlx_server/models/adapters/gemma.py"
      issue: "Line 40: tokenizer.eos_token_id fails for Processor objects"
    - path: "backend/mlx_manager/mlx_server/models/adapters/qwen.py"
      issue: "Line 41: tokenizer.eos_token_id fails for Processor objects"
    - path: "backend/mlx_manager/mlx_server/models/adapters/llama.py"
      issue: "Similar issue likely exists"
    - path: "backend/mlx_manager/mlx_server/models/adapters/mistral.py"
      issue: "Similar issue likely exists"
  missing:
    - "Handle Processor objects in all adapters by accessing nested tokenizer (e.g., getattr(tokenizer, 'tokenizer', tokenizer).eos_token_id)"
  debug_session: ""

- truth: "Image URLs can be fetched from common sources"
  status: failed
  reason: "Wikipedia returns 403 Forbidden - httpx client missing User-Agent header"
  severity: minor
  test: 6
  root_cause: "httpx client in image_processor.py doesn't set User-Agent header, Wikipedia blocks bare requests"
  artifacts:
    - path: "backend/mlx_manager/mlx_server/services/image_processor.py"
      issue: "httpx.AsyncClient() created without headers"
  missing:
    - "Add User-Agent header to httpx client in image_processor.py"
  debug_session: ""

## Notes

- Port 8000 conflict with Docker container caused false positives
- Consider changing default MLX server port from 8000 to avoid common conflicts
- Debug agent findings about MLX operations blocking event loop are valid but separate concern
