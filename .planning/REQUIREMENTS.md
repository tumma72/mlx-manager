# Requirements: MLX Model Manager

**Defined:** 2026-01-17
**Core Value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.

## v1 Requirements

Requirements for v1.1 release. Each maps to roadmap phases.

### Models Panel

- [x] **MODELS-01**: Search/filter bar anchored to top of panel with scrollable model list below
- [x] **MODELS-02**: When download starts, original model tile is hidden and download tile appears at top
- [x] **MODELS-03**: Normal model tiles do not show progress bars (progress only on download tiles)

### Server Panel

- [x] **SERVER-01**: Profile selection via searchable dropdown with Start button (replaces profile list)
- [x] **SERVER-02**: Running servers display as rich tiles showing: profile name, model name, memory consumption (graphical), GPU/CPU usage (graphical), uptime, stop/restart buttons (Note: tokens/s and total tokens require mlx-openai-server changes - tracked as gap)
- [x] **SERVER-03**: Scroll position preserved across 5-second polling updates (no scroll jump)

### Authentication

- [ ] **AUTH-01**: API key authentication via `MLX_MANAGER_API_KEY` environment variable
- [ ] **AUTH-02**: When API key is set, all API endpoints require `Authorization: Bearer <key>` header
- [ ] **AUTH-03**: When API key is not set, API endpoints are open (current behavior for localhost)
- [ ] **AUTH-04**: Frontend prompts for API key and stores in localStorage when auth is enabled

### Bug Fixes

- [ ] **BUGFIX-01**: Silent exception swallowing — add logging to all `except Exception: pass` blocks
- [ ] **BUGFIX-02**: Server process log file cleanup — guaranteed cleanup on crash/exit
- [ ] **BUGFIX-03**: Assertion-based validation — replace `assert` with proper HTTPException in routers
- [ ] **BUGFIX-04**: Frontend polling interval — adaptive polling or WebSocket for server status
- [ ] **BUGFIX-05**: Console.log pollution — remove debug logging from production code

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Chat

- **CHAT-01**: Thinking model support (collapsible thinking panel, handle various formats)
- **CHAT-02**: Multimodal support — attach images via button and drag-drop
- **CHAT-03**: Persist chat history to database
- **CHAT-04**: Chat history sidebar per server

### Model Discovery

- **DISC-01**: Detect model characteristics from config.json or metadata
- **DISC-02**: Visual badges for capabilities (multimodal, tool support, MCP, thinking, architecture)
- **DISC-03**: Filter/search by model characteristics

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| JWT/user management | Simple API key sufficient for single-user local app |
| Cloud deployment | Designed for local Apple Silicon use |
| Non-MLX models | Focused on mlx-community ecosystem |
| Windows/Linux support | macOS-specific (launchd, menubar) |
| Multi-user authentication | Local-only application, API key covers network access |

## Traceability

Which phases cover which requirements. Updated by create-roadmap.

| Requirement | Phase | Status |
|-------------|-------|--------|
| MODELS-01 | Phase 1 | Complete |
| MODELS-02 | Phase 1 | Complete |
| MODELS-03 | Phase 1 | Complete |
| SERVER-01 | Phase 2 | Complete |
| SERVER-02 | Phase 2 | Complete |
| SERVER-03 | Phase 2 | Complete |
| AUTH-01 | Phase 3 | Pending |
| AUTH-02 | Phase 3 | Pending |
| AUTH-03 | Phase 3 | Pending |
| AUTH-04 | Phase 3 | Pending |
| BUGFIX-01 | Phase 4 | Pending |
| BUGFIX-02 | Phase 4 | Pending |
| BUGFIX-03 | Phase 4 | Pending |
| BUGFIX-04 | Phase 4 | Pending |
| BUGFIX-05 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15 ✓
- Unmapped: 0

---
*Requirements defined: 2026-01-17*
*Last updated: 2026-01-17 after Phase 2 completion*
