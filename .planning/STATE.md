# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-27)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** Phase 7 - Foundation (Server Skeleton & Single Model Inference)

## Current Position

Phase: 7 of 12 (Foundation - Server Skeleton & Single Model Inference)
Plan: 4 of 6 complete
Status: In progress
Last activity: 2026-01-27 — Completed 07-04-PLAN.md (Model adapters)

Progress: [██████░░░░] 67% (4/6 plans complete in Phase 7)

## Milestone v1.2 Summary

**Goal:** MLX Unified Server (pivoted from adapter/proxy approach)
**Status:** Phase 7 in progress (4/6 plans complete)
**Phases:** 6 phases (7-12)
**Requirements:** 28 total
- Server Foundation: SRV-01 to SRV-05 (5 requirements)
- Continuous Batching: BATCH-01 to BATCH-04 (4 requirements)
- API Layer: API-01 to API-05 (5 requirements)
- Model Adapters: ADAPT-01 to ADAPT-05 (5 requirements)
- Cloud Fallback: CLOUD-01 to CLOUD-04 (4 requirements)
- Configuration: CONF-01 to CONF-04 (4 requirements)
- Production: PROD-01 to PROD-04 (4 requirements)

## v1.2 Pivot Rationale

**Previous approach:** Build adapters/proxies for mlx-openai-server, vLLM-MLX, and cloud backends.

**New approach:** Build our own high-performance MLX inference server directly on mlx-lm/mlx-vlm/mlx-embeddings.

**Why:**
1. Research confirmed feasibility — mature foundation libraries (Apple-maintained mlx-lm)
2. vLLM-MLX proved batching gains: 328→1112 tok/s (3.4x) on M4 Max
3. Full control over inference stack enables optimizations impossible with external servers
4. Removes dependency on external server projects (mlx-openai-server stability issues, vLLM-MLX early stage)
5. Unified codebase — one project to maintain instead of adapter sprawl

**Key technologies:**
- mlx-lm + mlx-vlm + mlx-embeddings for inference
- Pydantic v2 (Rust core) for validation
- Pydantic LogFire for observability
- Continuous batching + paged KV cache for throughput

## Milestone v1.1 Summary

**Shipped:** 2026-01-26
**Version:** v1.1.0
**Stats:**
- Requirements: 29/29 complete
- Phases: 6/6 complete
- Plans: 37 executed
- Duration: 2026-01-17 to 2026-01-26

**Archive:**
- `.planning/milestones/v1.1-ROADMAP.md`
- `.planning/milestones/v1.1-REQUIREMENTS.md`
- `.planning/v1.1-MILESTONE-AUDIT.md`

## Performance Metrics (v1.1)

**Velocity:**
- Total plans completed: 37
- Average duration: ~3.5 min
- Total execution time: ~130 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1/1 | — | — |
| 2 | 5/5 | ~27 min | ~5 min |
| 3 | 5/5 | ~19 min | ~4 min |
| 4 | 3/3 | ~12 min | ~4 min |
| 5 | 5/5 | ~18 min | ~4 min |
| 6 | 18/18 | ~38 min | ~2 min |

## Quality at Ship (v1.1)

**Backend:**
- Tests: 550 passing
- Coverage: 97%
- Linting: ruff clean
- Type checking: mypy clean

**Frontend:**
- Tests: 544 passing
- Linting: eslint clean
- Type checking: svelte-check clean

## Accumulated Context

### Decisions

Recent decisions affecting current work:

- **v1.2 Pivot**: Build own MLX server instead of adapters/proxies — feasibility research confirmed, better control
- **Pydantic v2**: Use for all validation (Rust core, 5-50x faster than v1) — already used by FastAPI
- **Pydantic LogFire**: Replace Prometheus + OpenTelemetry — native FastAPI/HTTPX/LLM instrumentation
- **AuthLib**: Use existing auth infrastructure for API key encryption — consolidates JWT, OAuth2-ready
- **mlx-lm + mlx-vlm + mlx-embeddings**: Foundation libraries — mature, Apple-maintained, proven
- **Llama 3.x dual stop tokens**: Requires both eos_token_id and <|eot_id|> for proper completion — prevents infinite generation
- **New MLX memory API**: Use mx.get_* instead of deprecated mx.metal.get_* for future compatibility

See PROJECT.md Key Decisions table for full history.

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 9 readiness (Continuous Batching):**
- Paged KV cache implementation is complex — may need to study vLLM-MLX source carefully
- Concurrent request handling with async locks needs careful design to avoid deadlocks

**Phase 10 readiness (Cloud Fallback):**
- Cost tracking data source decision deferred: hardcoded pricing table vs API fetch

## Research Documents

**Feasibility Study:** `.planning/research/MLX-SERVER-FEASIBILITY.md`
- Confirms feasibility with high confidence
- Documents vLLM performance techniques (PagedAttention, continuous batching)
- Proposes architecture and technology stack
- Includes implementation roadmap aligned with phases

## Known Tech Debt (Carried Forward)

1. **Throughput metrics not available** — Will be solved by our own server with proper metrics
2. **mlx-openai-server v1.5.0 regression** — No longer relevant after v1.2 ships our own server
3. **Download completion UX** — Doesn't auto-refresh local models list after download completes

## Session Continuity

Last session: 2026-01-27T16:26:41Z
Stopped at: Completed 07-04-PLAN.md (Model adapters)
Resume file: None
Next: Execute 07-05-PLAN.md (Generation engine)
