# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-17)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** Phase 2 — Server Panel Redesign

## Current Position

Phase: 2 of 4 (Server Panel Redesign)
Plan: 3 of 4 complete
Status: In progress
Last activity: 2026-01-17 — Completed 02-03-PLAN.md

Progress: ███▒░░░░░░ 35% (1/4 phases complete, 3/4 phase 2 plans done)

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: ~6 min
- Total execution time: ~23 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1/1 | — | — |
| 2 | 3/4 | ~20 min | ~7 min |

**Recent Trend:**
- Last 5 plans: 01-01, 02-01, 02-02, 02-03
- Trend: Fast execution with one user feedback iteration

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

### Pending Todos

1. **Restore test coverage to 95%+** — Backend at 88%, 33 skipped tests need audit

### Known Gaps

1. **Throughput metrics not available** — User requested tokens/s and message count per server. Backend API does not expose this data. Would require mlx-openai-server changes to expose /v1/stats or similar.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-17
Stopped at: Completed 02-03-PLAN.md (ServerTile with metrics gauges)
Resume file: None
Next plan: 02-04-PLAN.md (if exists, otherwise phase 2 complete)
