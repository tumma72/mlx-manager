# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-17)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** Phase 3 (User-Based Authentication) - Complete

## Current Position

Phase: 3 of 6 (User-Based Authentication)
Plan: 5 of 5 complete
Status: Phase complete
Last activity: 2026-01-20 — Completed 03-05-PLAN.md (Admin UI Pages)

Progress: ████░░░░░░ 50% (3 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: ~4 min
- Total execution time: ~49 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1/1 | — | — |
| 2 | 5/5 | ~27 min | ~5 min |
| 3 | 5/5 | ~19 min | ~4 min |

**Recent Trend:**
- Last 5 plans: 03-01, 03-02, 03-03, 03-04, 03-05
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
- PyJWT over python-jose (abandoned) per FastAPI official recommendation
- pwdlib[argon2] over passlib for Python 3.13+ compatibility
- UserStatus enum for approval workflow (PENDING -> APPROVED -> DISABLED)
- OAuth2PasswordRequestForm for login (username field contains email)
- First user auto-approved and admin, subsequent users pending
- Admin endpoint pattern: Depends(get_admin_user) for admin-only routes
- Prevent last admin from demoting/deleting self
- All endpoints require get_current_user dependency (no unauthenticated access)
- Class-based auth store with Svelte 5 runes (consistent with existing stores)
- Auto-clear auth and redirect to /login on 401 response
- onMount redirect for admin-only page guards (client-side, consistent pattern)
- 30-second polling for pending count in navbar (freshness vs load balance)

### Pending Todos

1. **Restore test coverage to 95%+** — Backend at 88%, 33 skipped tests need audit

### Known Gaps

1. **Throughput metrics not available** — User requested tokens/s and message count per server. Backend API does not expose this data. Would require mlx-openai-server changes to expose /v1/stats or similar.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-20
Stopped at: Completed 03-05-PLAN.md (Admin UI Pages)
Resume file: None
Next plan: Phase 3 complete - ready for Phase 4 planning
