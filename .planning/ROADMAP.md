# Roadmap: MLX Model Manager v1.1

## Overview

Polish the UX for models and server panels, add optional API key authentication for network security, and fix accumulated technical debt. Four focused phases moving from user-facing improvements to infrastructure hardening.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Models Panel UX** - Anchored search, consolidated downloads
- [ ] **Phase 2: Server Panel Redesign** - Dropdown selection, rich server tiles with metrics
- [ ] **Phase 3: API Key Authentication** - Optional bearer token auth
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

### Phase 2: Server Panel Redesign
**Goal**: Replace profile list with searchable dropdown and show running servers as rich metric tiles
**Depends on**: Phase 1
**Requirements**: SERVER-01, SERVER-02, SERVER-03
**Success Criteria** (what must be TRUE):
  1. User can search and select a profile from dropdown
  2. User can click Start button to launch selected profile
  3. Running servers display as tiles with real-time metrics (memory, CPU/GPU, uptime, throughput)
  4. Stop/Restart buttons work on running server tiles
  5. Scrolling the server list doesn't jump during polling updates
**Research**: Likely (server metrics instrumentation)
**Research topics**: psutil for memory/CPU, GPU metrics on Apple Silicon, real-time throughput tracking
**Plans**: TBD

Plans:
- [ ] 02-01: Backend metrics API (memory, CPU, GPU, throughput)
- [ ] 02-02: Profile dropdown with search and Start button
- [ ] 02-03: Running server tiles with real-time metrics display
- [ ] 02-04: Scroll preservation during polling

### Phase 3: API Key Authentication
**Goal**: Optional bearer token authentication for network security
**Depends on**: Phase 2
**Requirements**: AUTH-01, AUTH-02, AUTH-03, AUTH-04
**Success Criteria** (what must be TRUE):
  1. When `MLX_MANAGER_API_KEY` is set, unauthenticated requests return 401
  2. When API key is set, valid Bearer token grants access
  3. When API key is not set, all endpoints are open (backwards compatible)
  4. Frontend prompts for API key when auth is enabled and stores in localStorage
**Research**: Unlikely (FastAPI auth patterns, localStorage)
**Plans**: TBD

Plans:
- [ ] 03-01: Backend auth middleware with env-based toggle
- [ ] 03-02: Frontend auth prompt and localStorage storage

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
| 2. Server Panel Redesign | 0/4 | Not started | - |
| 3. API Key Authentication | 0/2 | Not started | - |
| 4. Bug Fixes & Stability | 0/4 | Not started | - |
