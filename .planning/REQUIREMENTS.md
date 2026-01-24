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

- [x] **AUTH-01**: User registration with email + password (first user becomes admin)
- [x] **AUTH-02**: Subsequent users request accounts, admin must approve before active
- [x] **AUTH-03**: JWT-based authentication with 7-day session duration
- [x] **AUTH-04**: Unauthenticated requests redirect to /login page
- [x] **AUTH-05**: Admin user management page (approve, delete, reset passwords)
- [x] **AUTH-06**: Pending approval badge count in nav (admin only)

### Model Discovery

- [x] **DISC-01**: Detect model characteristics from config.json or metadata (architecture, context window, multimodal, KV cache)
- [x] **DISC-02**: Visual badges for capabilities (text-only vs multimodal, architecture type, quantization)
- [x] **DISC-03**: Filter/search by model characteristics
- [x] **DISC-04**: Best effort to determine model "tool-use" capability and add a badge to the model (tip: check in the front matter tags for "tool-use" or similar)

### Profile Enhancements

- [x] **PRO-01**: The model description in a profile should be a text area not a simple input field
- [x] **PRO-02**: The Profile should also have a default *system prompt* field which is used when starting the server to set the system prompt

### Chat Multimodal

- [x] **CHAT-01**: Thinking model support (collapsible thinking panel, handle various formats)
- [x] **CHAT-02**: Multimodal support — attach images via button and drag-drop
- [x] **CHAT-03**: Video attachments (frame extraction or direct if model supports)
- [x] **CHAT-04**: Integrate small MCP mock for get_weather or calculator (with aritmetic operations) to test if models use tools correctly

### Bug Fixes

- [x] **BUGFIX-01**: Silent exception swallowing — add logging to all `except Exception: pass` blocks
- [x] **BUGFIX-02**: Server process log file cleanup — guaranteed cleanup on crash/exit
- [x] **BUGFIX-03**: Assertion-based validation — replace `assert` with proper HTTPException in routers
- [x] **BUGFIX-04**: Frontend polling interval — adaptive polling or WebSocket for server status
- [x] **BUGFIX-05**: Console.log pollution — remove debug logging from production code
- [x] **BUGFIX-06**: Sometimes models are marked as already started in Servers, but they fail to load when starting a chat — needs move investigation
- [x] **BUGFIX-07**: The Servers CPU gauge doesn't change it is always showing 0% the memory gauge also doesn't seem to work is showing 387Mb for Qwen3-Coder which is a model of 16Gb size

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Server Enhancements

- **SERVER-04**: Extend the wrapped mlx-openai-server to support also Anthropic APIs while reusing the rest of the infrastructure
- **SERVER-05**: Extend mlx-manager to act as proxy and expose a single URL:PORT and internally reroute to model running on mlx-openai-server instances

### Chat History

- **CHAT-05**: Persist chat history to database
- **CHAT-06**: Chat history sidebar per server
- **CHAT-07**: Server-scoped chats (switching server loads relevant history)
- **CHAT-08**: Create new / delete existing chats

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| OAuth2/SSO | User-based auth with email+password covers local network use |
| Cloud deployment | Designed for local Apple Silicon use |
| Non-MLX models | Focused on mlx-community ecosystem |
| Windows/Linux support | macOS-specific (launchd, menubar) |
| Email-based password recovery | Requires SMTP setup, out of scope for local-first tool |

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
| AUTH-01 | Phase 3 | Complete |
| AUTH-02 | Phase 3 | Complete |
| AUTH-03 | Phase 3 | Complete |
| AUTH-04 | Phase 3 | Complete |
| AUTH-05 | Phase 3 | Complete |
| AUTH-06 | Phase 3 | Complete |
| DISC-01 | Phase 4 | Complete |
| DISC-02 | Phase 4 | Complete |
| DISC-03 | Phase 4 | Complete |
| DISC-04 | Phase 6 | Complete |
| PRO-01 | Phase 6 | Complete |
| PRO-02 | Phase 6 | Complete |
| CHAT-01 | Phase 5 | Complete |
| CHAT-02 | Phase 5 | Complete |
| CHAT-03 | Phase 5 | Complete |
| CHAT-04 | Phase 6 | Complete |
| BUGFIX-01 | Phase 6 | Complete |
| BUGFIX-02 | Phase 6 | Complete |
| BUGFIX-03 | Phase 6 | Complete |
| BUGFIX-04 | Phase 6 | Complete |
| BUGFIX-05 | Phase 6 | Complete |
| BUGFIX-06 | Phase 6 | Complete |
| BUGFIX-07 | Phase 6 | Complete |

**Coverage:**
- v1.1 requirements: 29 total
- Mapped to phases: 29 ✓
- Unmapped: 0

---
*Requirements defined: 2026-01-17*
*Last updated: 2026-01-24 — Phase 6 complete: all BUGFIX, DISC-04, PRO-01/02, CHAT-04 marked Complete*
