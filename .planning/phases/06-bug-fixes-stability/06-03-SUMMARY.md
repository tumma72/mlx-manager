---
phase: 06-bug-fixes-stability
plan: 03
subsystem: model-discovery
tags: [badges, ui, model-detection, tool-use, function-calling]
requires:
  - phase: 04
    plan: 01
    provides: badge-system
  - phase: 04
    plan: 03
    provides: model-characteristics
provides:
  - tool-use-badge
  - tool-use-detection
affects:
  - models-page
tech-stack:
  added: []
  patterns:
    - dual-detection-strategy
    - tag-based-detection
    - config-based-fallback
key-files:
  created:
    - frontend/src/lib/components/models/badges/ToolUseBadge.svelte
  modified:
    - backend/mlx_manager/types.py
    - backend/mlx_manager/utils/model_detection.py
    - frontend/src/lib/api/types.ts
    - frontend/src/lib/components/models/ModelBadges.svelte
decisions:
  - what: Dual detection strategy for tool-use capability
    why: HuggingFace tags are authoritative when available, but config-based fallback ensures local models can be detected
    alternatives: Tag-only detection would miss local models without network access
  - what: Amber color scheme for tool-use badge
    why: Distinguishes from existing badges (blue=architecture, purple=multimodal, green=quantization) while maintaining visual consistency
    alternatives: Orange or yellow considered but amber provides better contrast in both light/dark modes
  - what: Tool-use indicators in tags
    why: Common patterns across HuggingFace model tags include "tool-use", "function-calling", "tools"
    alternatives: Single indicator would miss models using different naming conventions
metrics:
  duration: 4 minutes
  completed: 2026-01-24
---

# Phase 6 Plan 03: Tool-Use Badge Detection & Display Summary

**One-liner:** Dual-strategy tool-use detection (tags + config) with amber badge for function-calling capable models

## What Was Built

### Tool-Use Detection (Backend)
Added `is_tool_use` field to `ModelCharacteristics` with dual detection strategy:

1. **Tag-based detection (primary)**: Scans HuggingFace model tags for indicators:
   - "tool-use", "tool_use"
   - "function-calling", "function_calling"
   - "tool-calling", "tools"

2. **Config-based detection (fallback)**: Checks config.json for:
   - `tool_call_parser` field presence
   - Known tool-capable architectures (Qwen, GLM, MiniMax) with parser configs

### Tool-Use Badge (Frontend)
Created amber-themed badge component following established pattern:
- **Icon**: Wrench (from lucide-svelte)
- **Color**: Amber (bg-amber-100/dark:bg-amber-900/30)
- **Integration**: Conditional render in ModelBadges when `is_tool_use: true`

## Technical Implementation

### Backend Changes

**types.py:**
```python
class ModelCharacteristics(TypedDict, total=False):
    # ... existing fields
    is_tool_use: bool  # Tool-use / function-calling support
```

**model_detection.py:**
```python
def detect_tool_use(config: dict[str, Any], tags: list[str] | None = None) -> bool:
    # Tag-based detection (primary)
    if tags:
        tool_indicators = ["tool-use", "tool_use", "function-calling", ...]
        if any(indicator in tag.lower() for tag in tags for indicator in tool_indicators):
            return True

    # Config-based detection (fallback)
    if "tool_call_parser" in config:
        return True

    return False
```

### Frontend Changes

**ToolUseBadge.svelte:**
```svelte
<div class="... bg-amber-100 text-amber-800 ...">
  <Wrench class="w-3 h-3" />
  Tool Use
</div>
```

**ModelBadges.svelte:**
```svelte
{#if characteristics.is_tool_use}
  <ToolUseBadge />
{/if}
```

## Quality Verification

**Backend:**
- ✅ Ruff linting: All checks passed
- ✅ Type checking: Success (mypy)
- ✅ Tests: 532 passed

**Frontend:**
- ✅ Type checking: 0 errors, 0 warnings
- ✅ Linting: ESLint passed
- ✅ Tests: 544 passed

## User Impact

**Before:** No way to identify models with tool-use/function-calling support from the UI

**After:** Models with tool-use capability display a clear amber "Tool Use" badge:
- Helps users select appropriate models for agentic workflows
- Visible on both search results and local models pages
- Works offline (config-based fallback for downloaded models)

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

**Dependencies satisfied:**
- Badge system (Phase 04-01) ✅
- Model characteristics extraction (Phase 04-03) ✅

**New capabilities:**
- Tool-use detection available for profile creation recommendations
- Badge pattern can be extended for additional model capabilities

**No blockers identified.**
