# MLX Model Manager

## What This Is

A web application for managing MLX-optimized language models on Apple Silicon Macs. Provides a UI for browsing/downloading models from HuggingFace's mlx-community, managing server profiles, controlling server instances, and configuring launchd services. Built for local-first AI development with a focus on ease of use.

## Core Value

Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Model browsing/search from HuggingFace mlx-community — existing
- ✓ Model download with progress tracking and resume capability — existing
- ✓ Server profile management (create, edit, delete configurations) — existing
- ✓ Server instance control (start/stop/restart) — existing
- ✓ Server health monitoring with status indicators — existing
- ✓ macOS launchd service integration for auto-start — existing
- ✓ Menubar app for quick access — existing
- ✓ Basic chat interface for testing models — existing
- ✓ Dark mode support — v1.0.3

### Active

<!-- Current scope. Building toward these. -->

**v1.1 — UX Polish & Stability**

- [ ] BUG: Fix server page scroll reset (5s polling causes re-render and scroll jump)
- [ ] Models panel: Anchor search/filter to top, scrollable model list below
- [ ] Models panel: Consolidate download state — hide original tile when download starts, show only download tile at top
- [ ] Models panel: Remove progress bar from normal model tiles (simplification)
- [ ] Server panel: Replace profile list with searchable dropdown + Start button
- [ ] Server panel: Running servers appear as rich tiles at top with:
  - Profile name + model name
  - Memory consumption (graphical)
  - GPU/CPU usage (graphical)
  - Uptime
  - Tokens/second throughput
  - Total tokens generated
  - Stop/Restart buttons

**v1.2 — Enhanced Chat & Model Discovery**

- [ ] Chat: Proper thinking model support (collapsible thinking panel, handle various formats)
- [ ] Chat: Multimodal support — attach images via button and drag-drop
- [ ] Chat: Persist chat history to database
- [ ] Chat: Left sidebar showing chat history per server
- [ ] Chat: Server-scoped chats (switching server loads relevant history)
- [ ] Chat: Create new / delete existing chats
- [ ] Models: Detect model characteristics from config.json or metadata
- [ ] Models: Visual badges for capabilities (multimodal, tool support, MCP, thinking, architecture)
- [ ] Models: Filter/search by model characteristics

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Multi-user authentication — local-only application, binds to 127.0.0.1
- Cloud deployment — designed for local Apple Silicon use
- Non-MLX models — focused on mlx-community ecosystem
- Windows/Linux support — macOS-specific (launchd, menubar)

## Context

**Technical Environment:**
- Backend: FastAPI + SQLModel + aiosqlite (async SQLite)
- Frontend: SvelteKit 2 + Svelte 5 (runes) + TailwindCSS + bits-ui
- Distribution: Python package with embedded static frontend
- Target: macOS on Apple Silicon (M1/M2/M3/M4)

**Known Issues (from codebase analysis):**
- 5-second polling interval causes constant re-renders (source of scroll bug)
- ProfileCard has complex polling state machine (fragile)
- Silent exception swallowing in multiple services
- No frontend component tests for critical UI

**Current State:**
- v1.0.3 released with dark mode support
- 67% backend test coverage
- Codebase mapped in `.planning/codebase/`

## Constraints

- **Tech stack**: Must maintain FastAPI/SvelteKit architecture — invested and working
- **Distribution**: Single Python package with embedded frontend — current packaging approach
- **Compatibility**: macOS only, Apple Silicon required for MLX
- **Backend metrics**: Server memory/GPU usage requires new instrumentation — may need `psutil` or similar

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix scroll bug via scroll preservation, not polling refactor | Polling refactor is larger scope; preserve scroll is targeted fix | — Pending |
| Server panel redesign separates "profiles" (config) from "servers" (running instances) | Current UI conflates configuration and runtime state | — Pending |
| Chat persistence in SQLite | Consistent with existing data layer; no new dependencies | — Pending |
| Model characteristics via background enrichment | Don't block search; fetch metadata async and update UI | — Pending |

---
*Last updated: 2026-01-17 after initialization*
