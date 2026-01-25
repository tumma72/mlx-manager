---
phase: 06
plan: 18
type: gap-closure
subsystem: model-detection
tags: [tool-use, badges, console-errors, model-characteristics, huggingface]
requires: [06-15, 06-17]
provides: [reliable-tool-use-badge, clean-browser-console]
affects: [models-panel-ux]
tech-stack:
  added: []
  patterns: [family-based-detection, graceful-degradation]
decisions:
  - title: "204 No Content for Missing Configs"
    rationale: "Browser console logs 404 errors before JS catch handler runs. 204 indicates expected absence without error logging."
    alternatives: "Keep 404, accept console noise"
    tradeoffs: "Less semantic (404 is technically correct) but better UX"
  - title: "Model Family Allowlist for Tool-Use"
    rationale: "Many model creators don't set HuggingFace tags correctly. Qwen/GLM/MiniMax/DeepSeek/Hermes families have well-documented tool support."
    alternatives: "Rely only on tags, accept false negatives"
    tradeoffs: "Requires maintenance as new families emerge"
key-files:
  created: []
  modified:
    - backend/mlx_manager/routers/models.py
    - backend/mlx_manager/utils/model_detection.py
    - frontend/src/lib/stores/models.svelte.ts
metrics:
  duration: 1m 51s
  tasks: 3
  commits: 3
  completed: 2026-01-25
---

# Phase 6 Plan 18: Tool-Use Badge & Console Errors Summary

**One-liner:** Model family allowlist for reliable tool-use detection + 204 response eliminates console errors

## What Was Built

Fixed two UX issues in model browsing:

1. **Browser console 404 errors eliminated** — Backend now returns `204 No Content` instead of `404 Not Found` when model config.json is unavailable. The browser's network layer logs 404s before JavaScript catch handlers run, creating console noise. 204 indicates "expected absence" without error logging.

2. **Reliable tool-use badge on known families** — Added `TOOL_CAPABLE_FAMILIES` allowlist to both backend and frontend. Models from Qwen, GLM, MiniMax, DeepSeek, Hermes, Command-R, and Mistral families now show tool-use badge even when HuggingFace tags are missing. Many model creators don't tag correctly despite native tool support.

## Implementation Details

### Backend Changes

**models.py:**
- Import `Response` from fastapi
- Replace `HTTPException(404)` with `Response(status_code=204)` in get_model_config endpoint
- Updated docstring to document 204 behavior

**model_detection.py:**
- Added `TOOL_CAPABLE_FAMILIES` constant with 12 known tool-capable model type strings
- Updated `detect_tool_use()` with three-tier detection:
  1. Tag-based (primary): Check HuggingFace tags for tool indicators
  2. Family-based (secondary): Check model_type against TOOL_CAPABLE_FAMILIES
  3. Config-based (fallback): Check for tool_call_parser in config.json
- Removed complex nested string value search (replaced by simpler family check)

### Frontend Changes

**models.svelte.ts:**
- Added `TOOL_CAPABLE_FAMILIES` Set with 7 normalized family names (matches backend ARCHITECTURE_FAMILIES mapping)
- Updated `parseCharacteristicsFromName()` to check family after tag-based detection
- Falls back to family allowlist when tags don't indicate tool support

## Deviations from Plan

None - plan executed exactly as written.

## Testing Notes

### Manual Verification Needed

1. **Browser console clean** — Browse models panel, open DevTools console, verify no 404 errors for `/api/models/config/*` requests
2. **Qwen models show tool badge** — Search "Qwen3", verify tool-use badge appears even without explicit tags
3. **GLM-4 shows tool badge** — Search "GLM", verify badge on GLM-4 models
4. **MiniMax shows tool badge** — Search "MiniMax", verify badge appears
5. **DeepSeek shows tool badge** — Search "DeepSeek", verify badge appears
6. **Tagged models still work** — Verify models with explicit tool-use tags still detected correctly

### Type Safety

- Frontend: `npm run check` passes with 0 errors, 0 warnings
- Backend: Python imports work without errors

## Technical Notes

### Why 204 Instead of 404?

Technically, 404 is correct — the config doesn't exist. But browsers log 404s at the network layer before JavaScript `try/catch` can handle them. This creates console noise for expected behavior (many models don't have configs yet).

`204 No Content` semantically means "request succeeded, but there's no content to return" — a better match for "config not downloaded yet" vs "config URL is wrong".

### Family Allowlist Maintenance

The allowlist will need updates as new tool-capable model families emerge. Consider these candidates for future additions:

- Aya (Cohere's multilingual family)
- Granite (IBM's tool-capable models)
- Phi (Microsoft Phi-3 supports tools in some variants)

Current strategy: Conservative (only add families with well-documented tool support).

### Backend/Frontend Consistency

Backend uses lowercase model_type values (`qwen`, `glm`), frontend uses normalized family names (`Qwen`, `GLM`). The mapping is:

- Backend `model_type` → `ARCHITECTURE_FAMILIES` → Frontend `architecture_family`
- Backend `TOOL_CAPABLE_FAMILIES` contains model_type values
- Frontend `TOOL_CAPABLE_FAMILIES` contains normalized family names

This separation is correct — backend reads raw config.json, frontend uses normalized display names.

## Next Phase Readiness

**Status:** Ready

**Integration points:**
- Badge display logic in frontend models panel already exists (06-15)
- Model characteristics API already supports tool-use field (06-15)
- No breaking changes to API contracts

**Blockers:** None

**Concerns:** None — changes are backwards compatible and isolated to detection logic.

## Files Changed

### Created
None.

### Modified
- `backend/mlx_manager/routers/models.py` — 204 response for missing configs
- `backend/mlx_manager/utils/model_detection.py` — TOOL_CAPABLE_FAMILIES + 3-tier detection
- `frontend/src/lib/stores/models.svelte.ts` — Frontend family allowlist + fallback logic

## Decisions Made

1. **204 No Content for Missing Configs** — Browser console logs 404 before JS catches. 204 indicates expected absence without error noise. Tradeoff: Less semantic but better UX.

2. **Model Family Allowlist** — Qwen/GLM/MiniMax/DeepSeek/Hermes families have well-documented tool support. Tags unreliable. Tradeoff: Requires maintenance as families emerge.

## Commits

- `999c8d8` — fix(06-18): return 204 instead of 404 for missing model configs
- `91b30b5` — feat(06-18): add model family allowlist for backend tool-use detection
- `63e85a3` — feat(06-18): add model family allowlist for frontend tool-use detection
