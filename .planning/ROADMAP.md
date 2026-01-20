# Roadmap: MLX Model Manager v1.1

## Overview

Polish the UX for models and server panels, add optional API key authentication for network security, and fix accumulated technical debt. Four focused phases moving from user-facing improvements to infrastructure hardening.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Models Panel UX** - Anchored search, consolidated downloads
- [x] **Phase 2: Server Panel Redesign** - Dropdown selection, rich server tiles with metrics
- [ ] **Phase 3: User-Based Authentication** - Email/password login with JWT, admin approval flow
- [ ] **Phase 4: Bug Fixes & Stability** - Logging, cleanup, validation, polling

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

### Phase 3: User-Based Authentication
**Goal**: User registration/login with email+password, JWT sessions, admin approval flow
**Depends on**: Phase 2
**Requirements**: AUTH-01, AUTH-02, AUTH-03, AUTH-04, AUTH-05
**Success Criteria** (what must be TRUE):
  1. First user to register becomes admin automatically
  2. Subsequent users can request accounts, admin must approve
  3. Authenticated users access app via JWT (7-day sessions)
  4. Unauthenticated requests redirect to /login page
  5. Admin can manage users from dedicated /users page
  6. Pending approval requests show badge count in nav (admin only)
**Research**: Likely (FastAPI JWT patterns, password hashing, SQLModel user schema)
**Plans**: TBD

Plans:
- [ ] 03-01: User database schema and password hashing
- [ ] 03-02: Registration and login API endpoints
- [ ] 03-03: JWT middleware and auth guards
- [ ] 03-04: Frontend login/register pages
- [ ] 03-05: Admin user management page

### Phase 4: Bug Fixes & Stability
**Goal**: Clean up technical debt: logging, cleanup, validation, polling
**Depends on**: Phase 3
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
- [ ] 04-01: Add logging to silent exception handlers
- [ ] 04-02: Server process cleanup and log file management
- [ ] 04-03: Replace assertions with HTTPException
- [ ] 04-04: Fix polling re-renders and remove debug logs

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Models Panel UX | 1/1 | ✓ Complete | 2026-01-17 |
| 2. Server Panel Redesign | 5/5 | ✓ Complete | 2026-01-19 |
| 3. User-Based Authentication | 0/5 | Not started | - |
| 4. Bug Fixes & Stability | 0/4 | Not started | - |
