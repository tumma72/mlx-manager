# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-17)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** Phase 4 In Progress — Model Discovery & Badges

## Current Position

Phase: 4 of 6 (Model Discovery & Badges) - IN PROGRESS
Plan: 1 of 3 complete
Status: In progress
Last activity: 2026-01-20 — Completed 04-01-PLAN.md (Backend Characteristics API)

Progress: ████░░░░░░ 53% (3 phases + 1 plan complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 12
- Average duration: ~4 min
- Total execution time: ~57 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1/1 | — | — |
| 2 | 5/5 | ~27 min | ~5 min |
| 3 | 5/5 | ~19 min | ~4 min |
| 4 | 1/3 | ~8 min | ~8 min |

**Recent Trend:**
- Last 5 plans: 03-03, 03-04, 03-05, 04-01
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
- TypedDict with total=False for optional model characteristics
- Architecture family normalized to display names (qwen2 -> Qwen)
- Multimodal detection via vision_config, token IDs, or type keywords
- Local cache read first, HF API fallback for remote config

### Pending Todos

1. **Standardize profile/server tile buttons** — Use smaller button style from ServerTile across both ProfileTile and ServerTile (UI polish)

### Known Gaps

1. **Throughput metrics not available** — User requested tokens/s and message count per server. Backend API does not expose this data. Would require mlx-openai-server changes to expose /v1/stats or similar.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-20
Stopped at: Completed 04-01-PLAN.md (Backend Characteristics API)
Resume file: None
Next plan: 04-02-PLAN.md (Frontend Badge Components)
