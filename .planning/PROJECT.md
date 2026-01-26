# MLX Model Manager

## What This Is

A web application for managing MLX-optimized language models on Apple Silicon Macs. Provides a UI for browsing/downloading models from HuggingFace's mlx-community, managing server profiles, controlling server instances, and configuring launchd services. Built for local-first AI development with a focus on ease of use.

## Core Value

Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.

## Shipped Releases

### v1.1.0 — UX Polish, Auth & Chat Enhancements (2026-01-26)

**Milestone:** 29/29 requirements, 6 phases, 37 plans executed

Key features shipped:
- **Models Panel UX**: Anchored search, consolidated download tiles
- **Server Panel Redesign**: Searchable dropdown, rich metric tiles with memory/CPU/uptime
- **User Authentication**: Email/password login, JWT sessions, admin approval flow
- **Model Discovery**: Characteristic detection, visual badges, filtering
- **Chat Multimodal**: Image/video attachments, thinking models, MCP tool integration
- **Bug Fixes**: 7 stability fixes including exception logging, validation, polling optimization

Archive: `.planning/milestones/v1.1-ROADMAP.md`, `.planning/milestones/v1.1-REQUIREMENTS.md`

### v1.0.3 — Dark Mode (2026-01-15)

- Dark mode support across entire UI
- Initial chat interface for testing models

## Current Milestone: v1.2 — Unified API Gateway

**Goal:** Transform mlx-manager into a unified API gateway that routes requests to local or cloud backends based on model name.

**Target features:**
- Backend abstraction layer (adapter pattern for mlx-openai-server, vLLM-MLX, OpenAI cloud, Anthropic cloud)
- Unified proxy endpoint exposing both OpenAI and Anthropic compatible APIs
- Model name → backend routing with UI configuration
- On-demand local model auto-start when requests arrive
- Secure in-app API key storage for cloud providers

## Current State

**Status:** Milestone v1.2 active — defining requirements
**Last release:** v1.1.0 (2026-01-26)

## v2 Requirements (Deferred)

Tracked for future releases:

### AI Proxy Agent (v1.3+)
- LLM-based intelligent routing (orchestrator model routes requests based on natural language rules)
- Configurable routing criteria per model (e.g., "Thinking Agent" for brainstorming, "Worker Agent" for coding)

### Chat History
- **CHAT-05**: Persist chat history to database
- **CHAT-06**: Chat history sidebar per server
- **CHAT-07**: Server-scoped chats (switching server loads relevant history)
- **CHAT-08**: Create new / delete existing chats

## Out of Scope

| Feature | Reason |
|---------|--------|
| OAuth2/SSO | User-based auth with email+password covers local network use |
| Cloud deployment | Designed for local Apple Silicon use |
| Non-MLX models | Focused on mlx-community ecosystem |
| Windows/Linux support | macOS-specific (launchd, menubar) |
| Email-based password recovery | Requires SMTP setup, out of scope for local-first tool |

## Technical Context

**Tech Stack:**
- Backend: FastAPI + SQLModel + aiosqlite (async SQLite)
- Frontend: SvelteKit 2 + Svelte 5 (runes) + TailwindCSS + bits-ui
- Distribution: Python package with embedded static frontend
- Target: macOS on Apple Silicon (M1/M2/M3/M4)

**Quality Metrics (v1.1.0):**
- Backend: 97% test coverage, 550 tests
- Frontend: 544 tests, eslint/svelte-check clean
- Code: Zero silent exception handlers, zero assertions in routes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Scroll preservation via JSON.stringify comparison | Prevents re-renders on unchanged data | Shipped v1.1 |
| JWT + admin approval flow | Simple but secure for local network use | Shipped v1.1 |
| Model characteristics via background enrichment | Don't block search; fetch metadata async | Shipped v1.1 |
| MCP mock for tool-use testing | Validate tool-capable models without external deps | Shipped v1.1 |
| Backend-mediated health polling | Eliminates browser console errors | Shipped v1.1 |
| Three-tier tool-use detection | Tags → family → config fallback chain | Shipped v1.1 |

## Known Tech Debt

| Item | Severity | Notes |
|------|----------|-------|
| Throughput metrics unavailable | Info | Requires mlx-openai-server changes |
| GPU metrics unavailable | Info | psutil limitation on macOS |
| Download completion UX | Warning | Doesn't auto-refresh local models list |

---

*Last updated: 2026-01-26 — Milestone v1.2 started*
