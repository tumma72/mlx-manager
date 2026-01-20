# Roadmap: MLX Model Manager v1.1

## Overview

Polish the UX for models and server panels, add optional API key authentication for network security, and fix accumulated technical debt. Four focused phases moving from user-facing improvements to infrastructure hardening.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Models Panel UX** - Anchored search, consolidated downloads
- [x] **Phase 2: Server Panel Redesign** - Dropdown selection, rich server tiles with metrics
- [x] **Phase 3: User-Based Authentication** - Email/password login with JWT, admin approval flow
- [x] **Phase 4: Model Discovery & Badges** - Detect characteristics, visual badges, filter by capabilities
- [ ] **Phase 5: Chat Multimodal Support** - Image/video attachments, thinking model display
- [ ] **Phase 6: Bug Fixes & Stability** - Logging, cleanup, validation, polling

## Phase Details

### Phase 1: Models Panel UX ✓
**Goal**: Clean up models panel layout so search stays visible and downloads are consolidated
**Depends on**: Nothing (first phase)
**Requirements**: MODELS-01, MODELS-02, MODELS-03
**Success Criteria** (what must be TRUE):
  1. Search/filter bar stays visible when scrolling model list ✓
  2. When download starts, only download tile visible at top (original tile hidden) ✓
  3. Non-downloading model tiles show no progress bar ✓
**Research**: None needed
**Completed**: 2026-01-17

Plans:
- [x] 01-01: Anchor search bar and consolidate download UX

### Phase 2: Server Panel Redesign ✓
**Goal**: Replace profile list with searchable dropdown and show running servers as rich metric tiles
**Depends on**: Phase 1
**Requirements**: SERVER-01, SERVER-02, SERVER-03
**Success Criteria** (what must be TRUE):
  1. User can search and select a profile from dropdown ✓
  2. User can click Start button to launch selected profile ✓
  3. Running servers display as tiles with real-time metrics (memory, CPU/GPU, uptime, throughput) ✓
  4. Stop/Restart buttons work on running server tiles ✓
  5. Scrolling the server list doesn't jump during polling updates ✓
**Research**: Completed (psutil for memory/CPU)
**Completed**: 2026-01-19

Plans:
- [x] 02-01: Backend metrics API (memory, CPU, GPU, throughput)
- [x] 02-02: Profile dropdown with search and Start button
- [x] 02-03: Running server tiles with real-time metrics display
- [x] 02-04: Scroll preservation during polling
- [x] 02-05: Gap closure - restart tile disappearing fix

### Phase 3: User-Based Authentication ✓
**Goal**: User registration/login with email+password, JWT sessions, admin approval flow
**Depends on**: Phase 2
**Requirements**: AUTH-01, AUTH-02, AUTH-03, AUTH-04, AUTH-05, AUTH-06
**Success Criteria** (what must be TRUE):
  1. First user to register becomes admin automatically ✓
  2. Subsequent users can request accounts, admin must approve ✓
  3. Authenticated users access app via JWT (7-day sessions) ✓
  4. Unauthenticated requests redirect to /login page ✓
  5. Admin can manage users from dedicated /users page ✓
  6. Pending approval requests show badge count in nav (admin only) ✓
**Research**: Completed (PyJWT + pwdlib, FastAPI dependencies, SvelteKit SPA auth)
**Completed**: 2026-01-20

Plans:
- [x] 03-01-PLAN.md — Backend auth foundation (User model, password hashing, JWT, dependencies)
- [x] 03-02-PLAN.md — Auth API endpoints (register, login, user management)
- [x] 03-03-PLAN.md — Secure existing APIs and frontend auth infrastructure
- [x] 03-04-PLAN.md — Login/register page and protected route structure
- [x] 03-05-PLAN.md — Admin user management page and nav badge

### Phase 4: Model Discovery & Badges ✓
**Goal**: Detect model characteristics and display visual badges for capabilities
**Depends on**: Phase 3
**Requirements**: DISC-01, DISC-02, DISC-03
**Success Criteria** (what must be TRUE):
  1. Model config.json parsed to extract: architecture, context window, multimodal support, KV cache ✓
  2. Visual badges displayed on model tiles (text-only vs multimodal, architecture type) ✓
  3. Technical specs shown: context window, parameters, quantization level ✓
  4. Filter/search by model characteristics works ✓
**Research**: Completed (HuggingFace config.json schema, MLX model metadata patterns)
**Completed**: 2026-01-20

Plans:
- [x] 04-01-PLAN.md — Backend model characteristics extraction (types, extraction logic, API endpoint)
- [x] 04-02-PLAN.md — Model tile badges and specs display (badge components, expandable specs, lazy loading)
- [x] 04-03-PLAN.md — Search UX refactor with filters (toggle switch, filter modal, filter chips)

### Phase 5: Chat Multimodal Support
**Goal**: Support image/video attachments for testing multimodal models
**Depends on**: Phase 4
**Requirements**: CHAT-01, CHAT-02
**Success Criteria** (what must be TRUE):
  1. Users can attach images via button and drag-drop
  2. Attached images display in chat and are sent to model
  3. Video attachments supported (frame extraction or direct if model supports)
  4. Thinking models show collapsible thinking panel
**Research**: Likely (MLX multimodal API, image encoding patterns)
**Plans**: TBD

Plans:
- [ ] 05-01: Image attachment UI and encoding
- [ ] 05-02: Multimodal message API integration
- [ ] 05-03: Thinking model display

### Phase 6: Bug Fixes & Stability
**Goal**: Clean up technical debt: logging, cleanup, validation, polling
**Depends on**: Phase 5
**Requirements**: BUGFIX-01, BUGFIX-02, BUGFIX-03, BUGFIX-04, BUGFIX-05
**Success Criteria** (what must be TRUE):
  1. Silent exceptions are logged (no more `except: pass`)
  2. Server log files are cleaned up on crash/exit
  3. API validation uses HTTPException (no assertions)
  4. Server status polling doesn't cause excessive re-renders
  5. No console.log debug statements in production
**Research**: Unlikely (standard fixes)
**Plans**: TBD

Plans:
- [ ] 06-01: Add logging to silent exception handlers
- [ ] 06-02: Server process cleanup and log file management
- [ ] 06-03: Replace assertions with HTTPException
- [ ] 06-04: Fix polling re-renders and remove debug logs

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Models Panel UX | 1/1 | ✓ Complete | 2026-01-17 |
| 2. Server Panel Redesign | 5/5 | ✓ Complete | 2026-01-19 |
| 3. User-Based Authentication | 5/5 | ✓ Complete | 2026-01-20 |
| 4. Model Discovery & Badges | 3/3 | ✓ Complete | 2026-01-20 |
| 5. Chat Multimodal Support | 0/3 | Not started | - |
| 6. Bug Fixes & Stability | 0/4 | Not started | - |
