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

- [ ] **DISC-01**: Detect model characteristics from config.json or metadata (architecture, context window, multimodal, KV cache)
- [ ] **DISC-02**: Visual badges for capabilities (text-only vs multimodal, architecture type, quantization)
- [ ] **DISC-03**: Filter/search by model characteristics

### Chat Multimodal

- [ ] **CHAT-01**: Thinking model support (collapsible thinking panel, handle various formats)
- [ ] **CHAT-02**: Multimodal support — attach images via button and drag-drop
- [ ] **CHAT-03**: Video attachments (frame extraction or direct if model supports)

### Bug Fixes

- [ ] **BUGFIX-01**: Silent exception swallowing — add logging to all `except Exception: pass` blocks
- [ ] **BUGFIX-02**: Server process log file cleanup — guaranteed cleanup on crash/exit
- [ ] **BUGFIX-03**: Assertion-based validation — replace `assert` with proper HTTPException in routers
- [ ] **BUGFIX-04**: Frontend polling interval — adaptive polling or WebSocket for server status
- [ ] **BUGFIX-05**: Console.log pollution — remove debug logging from production code

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Chat History

- **CHAT-04**: Persist chat history to database
- **CHAT-05**: Chat history sidebar per server
- **CHAT-06**: Server-scoped chats (switching server loads relevant history)
- **CHAT-07**: Create new / delete existing chats

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
| DISC-01 | Phase 4 | Pending |
| DISC-02 | Phase 4 | Pending |
| DISC-03 | Phase 4 | Pending |
| CHAT-01 | Phase 5 | Pending |
| CHAT-02 | Phase 5 | Pending |
| CHAT-03 | Phase 5 | Pending |
| BUGFIX-01 | Phase 6 | Pending |
| BUGFIX-02 | Phase 6 | Pending |
| BUGFIX-03 | Phase 6 | Pending |
| BUGFIX-04 | Phase 6 | Pending |
| BUGFIX-05 | Phase 6 | Pending |

**Coverage:**
- v1.1 requirements: 23 total
- Mapped to phases: 23 ✓
- Unmapped: 0

---
*Requirements defined: 2026-01-17*
*Last updated: 2026-01-20 — added auth, model discovery, chat multimodal to v1.1*
