# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-17)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** All phases complete and verified — Milestone v1.1 ready for audit

## Current Position

Phase: 6 of 6 (Bug Fixes & Stability) - COMPLETE
Plan: 18 of 18 gap closure (completed)
Status: Phase 6 fully verified (23/23 must-haves) — Milestone v1.1 complete
Last activity: 2026-01-25 — Phase 6 executed and verified, all 23 success criteria passed

Progress: ██████████ 100% (all 6 phases complete, gap closure 18/18 complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 34
- Average duration: ~3.7 min
- Total execution time: ~128 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1/1 | — | — |
| 2 | 5/5 | ~27 min | ~5 min |
| 3 | 5/5 | ~19 min | ~4 min |
| 4 | 3/3 | ~12 min | ~4 min |
| 5 | 5/5 | ~18 min | ~4 min |
| 6 | 18/18 | ~38 min | ~2 min |

**Recent Trend:**
- Last 7 plans: 06-11, 06-12, 06-13, 06-14, 06-16, 06-17, 06-18
- Trend: Fast execution, gap closure COMPLETE (18/18)

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- All silent exception handlers logged at appropriate level (debug for non-critical, warning for operational, error for critical)
- Assertions replaced with HTTPException(400) in routers, ValueError in services
- JSON.stringify comparison for store deduplication (prevents re-renders on unchanged data)
- Tool-use detection uses dual strategy: HuggingFace tags (primary) + config.json fields (fallback)
- Amber color scheme for ToolUseBadge (distinguishes from blue/purple/green)
- CPU metrics use interval=0.1 for parent, interval=0 for children
- Memory metrics sum RSS across parent + all child processes (recursive)
- _cleanup_log_file() helper centralized across all exit paths
- Profile system_prompt field (nullable TEXT, auto-migrated)
- System prompt shown as pinned italic message in chat
- System prompt sent as first message (role: 'system') in API calls
- Chat retry: 3 attempts, linear backoff (2s, 4s, 6s), only on 5xx/network errors
- MCP mock uses safe AST-based calculator (no eval/exec)
- MCP mock weather uses deterministic hash-based values
- Text files read as plain text (FileReader.readAsText), images/videos as base64 (readAsDataURL)
- Text file content prefixed with [File: name] header for model context
- ChatRequest forwards tools array to mlx-openai-server when present (tool_choice defaults to "auto")
- Tool call chunks emitted as SSE events (type: tool_call) for streaming UX
- finish_reason "tool_calls" triggers tool_calls_done SSE event
- Log first chunk delta keys and finish_reason for diagnosing thinking issues (debug level)
- Dual-mechanism thinking detection: reasoning_content field + <think> tag parsing
- Acceptable fallback: thinking without tags/parser shown as regular response text
- formatBytes() used for memory display to show GB for values >= 1024 MB
- Health check polling uses backend API (servers.health) instead of direct fetch to eliminate console errors
- Chat textarea auto-grows up to 150px, Enter submits, Shift+Enter inserts newline
- processSSEStream extracts SSE reading logic for reuse (initial + follow-up requests)
- Max 3 tool-call rounds to prevent infinite loops (hard limit with user warning)
- Tool calls rendered in collapsible ToolCallBubble panel (not inline markdown) with wrench icon
- Tool call data stored as structured ToolCallData[] metadata on messages
- Amber border for tool call panel (distinguishes from thinking, success, info)
- JSON arguments auto-formatted with pretty-print, green background for results
- Tool results sent as role:tool messages in follow-up requests
- Extension-based text file detection replaces mime-type checking for reliability
- is_tool_use included in hasAnyCharacteristic fallback validation
- KNOWN_TEXT_FILENAMES allowlist for extensionless text files (README, Makefile, Dockerfile, LICENSE)
- hasExtension flag distinguishes extension-based vs filename-based detection
- Dotfiles (.gitignore, .env variants) included in extensionless allowlist
- 204 No Content response for missing model configs (prevents browser console errors)
- Model family allowlist for tool-use detection (Qwen, GLM, MiniMax, DeepSeek, Hermes, Command-R, Mistral)
- Three-tier tool-use detection: tags (primary) → family (secondary) → config (fallback)

### Pending Todos

None.

### Known Gaps

1. **Throughput metrics not available** — User requested tokens/s and message count per server. Backend API does not expose this data. Would require mlx-openai-server changes to expose /v1/stats or similar.
2. **mlx-openai-server v1.5.0 regression** — GLM-4.7-Flash and Gemma VLM fail to load in dev (v1.5.0) but work in released v1.0.4. Upstream issue, not our code. Consider pinning `<1.5.0` or waiting for fix.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-25
Stopped at: Phase 6 execution complete (23/23 must-haves verified)
Resume file: None
Next: Milestone audit and completion (/gsd:audit-milestone) — all 6 phases verified
