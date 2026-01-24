# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-17)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** Phase 6 (Bug Fixes & Stability) — Error handling improvements complete, 3 plans remaining

## Current Position

Phase: 6 of 6 (Bug Fixes & Stability) - IN PROGRESS
Plan: 5 of 7 complete
Status: Phase in progress
Last activity: 2026-01-24 — Completed 06-07-PLAN.md (MCP mock service)

Progress: █████████░ 92% (24 of 26 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 24
- Average duration: ~3.5 min
- Total execution time: ~105 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1/1 | — | — |
| 2 | 5/5 | ~27 min | ~5 min |
| 3 | 5/5 | ~19 min | ~4 min |
| 4 | 3/3 | ~12 min | ~4 min |
| 5 | 5/5 | ~18 min | ~4 min |
| 6 | 5/7 | ~17 min | ~3.4 min |

**Recent Trend:**
- Last 5 plans: 06-07, 06-04, 06-02, 06-03, 06-01
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
- Remove all console.log debug statements from production code (keep console.error for legitimate errors)
- Add early-exit logic in state transition methods to prevent unnecessary reactive updates
- Only update error state when value actually changes (prevents triggering reactivity)
- Dual detection strategy for tool-use capability (tags + config fallback)
- Amber color scheme for tool-use badge (distinguishes from existing badges)
- Use debug level for non-critical failures (cache checks, fallbacks, optional deps)
- Use warning level for health check failures
- Use error level for database transaction failures
- Replace assertions with HTTPException(400) for state validation in routers
- Use AST parsing for safe calculator (no eval/exec code injection)
- Deterministic mock weather based on location hash for reproducible tests
- OpenAI function-calling format for tool definitions (compatible with mlx-openai-server)
- Tool errors returned as {error: string} not HTTP errors (tool execution succeeded, tool logic failed)

### Pending Todos

None.

### Known Gaps

1. **Throughput metrics not available** — User requested tokens/s and message count per server. Backend API does not expose this data. Would require mlx-openai-server changes to expose /v1/stats or similar.
2. **mlx-openai-server v1.5.0 regression** — GLM-4.7-Flash and Gemma VLM fail to load in dev (v1.5.0) but work in released v1.0.4. Upstream issue, not our code. Consider pinning `<1.5.0` or waiting for fix.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-24
Stopped at: Completed 06-07-PLAN.md (MCP mock service)
Resume file: None
Next plan: Continue Phase 6 (2 plans remaining)
