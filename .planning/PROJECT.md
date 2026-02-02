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

## Current Milestone: v1.2 — MLX Unified Server

**Goal:** Build our own high-performance MLX inference server with multi-model support, continuous batching, and dual API compatibility.

**Pivot:** After feasibility research confirmed viability, we're building our own server instead of adapters for external backends (mlx-openai-server, vLLM-MLX). Benefits: full control, proven 2-4x throughput gains from batching, single codebase.

**Target features:**
- FastAPI + uvloop server built on mlx-lm/mlx-vlm/mlx-embeddings
- Multi-model serving with LRU eviction and memory pressure monitoring
- Continuous batching scheduler for 2-4x throughput improvement (proven by vLLM-MLX)
- Paged KV cache for memory efficiency (<4% waste vs 60-80%)
- OpenAI and Anthropic compatible REST APIs
- Cloud fallback routing (OpenAI/Anthropic APIs) when local unavailable
- Pydantic v2 validation + Pydantic LogFire observability

## Current State

**Status:** Milestone v1.2 active — roadmap revised, ready to plan Phase 7
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
| AuthLib for unified auth/crypto | Consolidates JWT, password hashing, API key encryption; OAuth2-ready | v1.2 Phase 7 |
| Build own MLX server | Full control, proven 2-4x batching gains, single codebase vs adapter sprawl | v1.2 Pivot |
| mlx-lm/mlx-vlm/mlx-embeddings | Apple-maintained, mature, proven foundation libraries | v1.2 Phase 7 |
| Pydantic v2 for validation | Rust core (5-50x faster), native FastAPI integration | v1.2 Phase 7 |
| Pydantic LogFire for observability | Native FastAPI/HTTPX/LLM instrumentation, built on OpenTelemetry | v1.2 Phase 7 |
| Continuous batching from Phase 9 | vLLM-MLX proved 3.4x throughput (328→1112 tok/s on M4 Max) | v1.2 Phase 9 |

## Known Tech Debt

| Item | Severity | Notes |
|------|----------|-------|
| Throughput metrics unavailable | Info | Will be solved by v1.2 server with LogFire metrics |
| GPU metrics unavailable | Info | psutil limitation on macOS |
| Download completion UX | Warning | Doesn't auto-refresh local models list |

---

*Last updated: 2026-01-27 — Milestone v1.2 pivoted to MLX Unified Server*
