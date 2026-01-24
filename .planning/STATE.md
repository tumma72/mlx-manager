# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-17)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** Phase 6 (Bug Fixes & Stability) — Server metrics accuracy complete, 6 plans remaining

## Current Position

Phase: 6 of 6 (Bug Fixes & Stability) - IN PROGRESS
Plan: 1 of 7 complete
Status: Phase in progress
Last activity: 2026-01-24 — Completed 06-04-PLAN.md (Server metrics accuracy & log cleanup)

Progress: ████████░░ 77% (20 of 26 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 20
- Average duration: ~3.6 min
- Total execution time: ~90 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1/1 | — | — |
| 2 | 5/5 | ~27 min | ~5 min |
| 3 | 5/5 | ~19 min | ~4 min |
| 4 | 3/3 | ~12 min | ~4 min |
| 5 | 5/5 | ~18 min | ~4 min |
| 6 | 1/7 | ~2 min | ~2 min |

**Recent Trend:**
- Last 5 plans: 05-03, 05-04, 05-05, 06-04
- Trend: Excellent velocity, sub-5-min execution

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- OpenAI ContentPart format for multimodal messages (mlx-openai-server compatible)
- Max 3 attachments per chat message (memory and UI considerations)
- Video duration limit of 2 minutes for attachments
- Use connection attempt as server check (httpx.ConnectError is appropriate check)
- Character-level streaming for thinking/response (matches mlx-openai-server granularity)
- Auto-expand ThinkingBubble during streaming, auto-collapse when done
- "Thought for Xs" label after thinking completes (shows thinking duration)
- FileReader.readAsDataURL() for client-side base64 encoding (no backend changes needed)
- ContentPart[] for multimodal, string for text-only API messages
- Strip <think>/<\/think> tags from reasoning_content (server may include them)
- Read both reasoning_content and content fields from SSE delta (dual detection)
- ErrorMessage component with collapsible details and copy-to-clipboard
- Inline error display in chat messages area (not banner)
- CHAT-04, DISC-04, PRO-01, PRO-02 deferred to Phase 6
- Text file MIME detection: text/*, application/json, application/xml, application/x-yaml, application/x-sh, application/sql
- Attachment button visible for all model types (not just multimodal)
- Text-only models accept only text files; multimodal models accept images, videos, and text files
- Use 100ms CPU measurement interval for accuracy (acceptable latency for status endpoint)
- Sum metrics across entire process tree (parent + children) for accurate model resource usage
- Centralize log file cleanup in helper method to prevent handle leaks

### Pending Todos

None.

### Known Gaps

1. **Throughput metrics not available** — User requested tokens/s and message count per server. Backend API does not expose this data. Would require mlx-openai-server changes to expose /v1/stats or similar.
2. **mlx-openai-server v1.5.0 regression** — GLM-4.7-Flash and Gemma VLM fail to load in dev (v1.5.0) but work in released v1.0.4. Upstream issue, not our code. Consider pinning `<1.5.0` or waiting for fix.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-24
Stopped at: Completed 06-04-PLAN.md (Server metrics accuracy & log cleanup)
Resume file: None
Next plan: Continue Phase 6 (6 plans remaining)
