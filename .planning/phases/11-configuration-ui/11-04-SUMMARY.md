---
phase: 11
plan: 04
title: "Routing Rules UI"
subsystem: frontend-settings
tags: [svelte, drag-drop, routing, ui-components]

dependency-graph:
  requires: ["11-01"]
  provides:
    - RoutingRulesSection component
    - RuleCard component
    - RuleForm component
    - RuleTestInput component
    - Drag-drop priority reordering
  affects: ["11-05"]

tech-stack:
  added:
    - "@rodrigodagostino/svelte-sortable-list@^2.1.10"
  patterns:
    - SortableList for drag-drop reordering
    - Optimistic UI updates with rollback on error

key-files:
  created:
    - frontend/src/lib/components/settings/RoutingRulesSection.svelte
    - frontend/src/lib/components/settings/RuleCard.svelte
    - frontend/src/lib/components/settings/RuleForm.svelte
    - frontend/src/lib/components/settings/RuleTestInput.svelte
  modified:
    - frontend/package.json
    - frontend/src/lib/components/settings/index.ts
    - frontend/src/lib/api/client.ts

decisions:
  - id: svelte-sortable-list
    choice: "@rodrigodagostino/svelte-sortable-list v2"
    rationale: "Svelte 5 compatible, accessibility-first, zero dependencies"
  - id: optimistic-reorder
    choice: "Optimistic UI updates for drag-drop"
    rationale: "Better UX - instant feedback, rollback on error"
  - id: derived-by-pattern
    choice: "Use $derived.by() for type narrowing"
    rationale: "Maintains TypeScript type narrowing in complex derivations"

metrics:
  duration: 7m
  completed: 2026-01-29
---

# Phase 11 Plan 04: Routing Rules UI Summary

**One-liner:** Drag-drop sortable routing rules UI with card display, creation form, and rule testing functionality.

## What Was Built

### Components Created

1. **RuleCard.svelte** (82 lines)
   - Individual rule display with drag handle
   - Pattern type badge (exact/prefix/regex) with color coding
   - Warning badge for unconfigured providers
   - Delete button with confirmation

2. **RuleForm.svelte** (160 lines)
   - Form for creating new routing rules
   - Pattern type selector with contextual placeholders
   - Backend selection (Local/OpenAI/Anthropic)
   - Optional backend model override
   - Optional fallback backend
   - Warning when selected backend is not configured

3. **RuleTestInput.svelte** (109 lines)
   - Test input to check which rule matches a model name
   - Real-time feedback showing matched rule and target backend
   - Success/error states with visual indicators

4. **RoutingRulesSection.svelte** (196 lines)
   - Main container integrating all rule components
   - Drag-drop sortable list using @rodrigodagostino/svelte-sortable-list
   - Optimistic priority updates on reorder
   - Automatic rollback on API error
   - Loading states and error handling

### API Integration

- Fixed `settings.testRule()` to use query parameter instead of JSON body
- All CRUD operations working: create, list, delete, update priorities

## Key Implementation Details

### Drag-Drop Reordering

The SortableList component from @rodrigodagostino/svelte-sortable-list provides:
- Mouse, keyboard, and touch support
- Accessibility-first with screen reader announcements
- Smooth transitions during drag

On drag end:
1. Optimistically update local state with `sortItems()` helper
2. Calculate new priorities (top = highest)
3. Batch update via `settings.updateRulePriorities()`
4. On error, reload from server to restore correct order

### Warning Badge Logic

Rules show a warning badge when:
- Backend type is not "local" AND
- Backend type is not in configured providers list

This helps users identify rules that won't work until the provider is configured.

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 89d5e80 | feat | Install drag-drop library and create RuleCard |
| 0715012 | feat | Create RuleForm and RuleTestInput components |
| 47da023 | feat | Create RoutingRulesSection with drag-drop |

## Deviations from Plan

None - plan executed exactly as written.

## Success Criteria Met

- [x] svelte-sortable-list installed
- [x] RuleCard displays pattern, type badge, backend, warning if needed
- [x] Drag handle allows reordering cards
- [x] Priority updates persist to backend on reorder
- [x] RuleForm creates new rules with all fields
- [x] RuleTestInput tests model names and shows results
- [x] Warning badge shows for unconfigured providers
- [x] All TypeScript compiles, ESLint passes

## Next Phase Readiness

Plan 11-04 completes the routing rules UI. The settings page now has:
- Provider configuration (11-03)
- Model pool settings (11-03)
- Routing rules management (11-04)

Ready for Plan 11-05: Settings page integration and final polish.
