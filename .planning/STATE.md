# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-17)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** Phase 2 Complete — Ready for Phase 3 (User-Based Authentication)

## Current Position

Phase: 2 of 6 (Server Panel Redesign) - COMPLETE
Plan: 5 of 5 complete (including gap closure)
Status: Phase complete
Last activity: 2026-01-19 — Completed 02-05-PLAN.md (Gap Closure: Restart tile disappearing)

Progress: ███░░░░░░░ 33% (2/6 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: ~5 min
- Total execution time: ~30 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1/1 | — | — |
| 2 | 5/5 | ~27 min | ~5 min |

**Recent Trend:**
- Last 5 plans: 02-01, 02-02, 02-03, 02-04, 02-05
- Trend: Fast execution, consistent velocity

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Search now finds all MLX models (filter=mlx) instead of only mlx-community
- Model sizes use usedStorage API for accuracy (not safetensors.total)
- bits-ui Combobox for searchable profile dropdown (accessible, keyboard-nav)
- Profile filter searches both name and model_path for flexibility
- Stabilize ProfileSelector profiles via ID comparison to prevent polling flicker
- Separate StartingTile component for starting/failed states
- Use servers list directly for running state detection in chat page
- Container-scoped scroll over window scroll for reliability
- Use $effect.pre for pre-update capture, $effect for post-update restore
- Track restarting state separately from starting state (restartingProfiles SvelteSet)

### Pending Todos

1. **Restore test coverage to 95%+** — Backend at 88%, 33 skipped tests need audit

### Known Gaps

1. **Throughput metrics not available** — User requested tokens/s and message count per server. Backend API does not expose this data. Would require mlx-openai-server changes to expose /v1/stats or similar.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-19
Stopped at: Completed 02-05-PLAN.md (Gap Closure: Restart tile disappearing)
Resume file: None
Next plan: Phase 3 planning (if exists)
