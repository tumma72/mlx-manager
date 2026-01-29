# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-27)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** Phase 9 complete - ready for Phase 10

## Current Position

Phase: 9 of 12 (Continuous Batching & Paged KV Cache)
Plan: 8 of 8 complete (including gap closure)
Status: Phase complete
Last activity: 2026-01-29 — Completed 09-08-PLAN.md (Gap closure - inference engine wiring)

Progress: [██████████] 100% (Phase 9 complete - continuous batching ready with gap closed)

## Milestone v1.2 Summary

**Goal:** MLX Unified Server (pivoted from adapter/proxy approach)
**Status:** Phases 7-8 complete (14 plans total)
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
- **Stop token detection in loop**: mlx_lm.stream_generate() doesn't accept stop_tokens param — must check in loop
- **Completions vs Chat template**: Completions endpoint uses raw prompt, chat uses chat template
- **Type casting for union returns**: Use cast() for type safety with AsyncGenerator | dict returns
- **Queue-based threading for MLX**: MLX Metal requires thread affinity — use dedicated Thread + Queue instead of run_in_executor
- **make_sampler API**: mlx_lm.stream_generate() no longer accepts temp/top_p directly — use make_sampler()
- **Model size estimation heuristic**: Uses param count patterns (3B=2GB, 7B=4GB) for LRU eviction memory planning
- **Preload protection**: Preloaded models never evicted regardless of last_used time — use for startup models
- **Config-first model type detection**: Check vision_config, image_token_id, architectures before name patterns for reliable type detection
- **Processor as tokenizer**: Vision models store processor in tokenizer field to reuse LoadedModel structure
- **mlx-embeddings L2 normalized**: text_embeds output is already L2-normalized — no post-processing needed for cosine similarity
- **Simulated streaming for vision**: mlx-vlm generate() is non-streaming — yield complete response as single chunk then finish chunk
- **Processor extraction pattern**: Use getattr(tokenizer, 'tokenizer', tokenizer) to handle both Tokenizer and Processor objects for vision model compatibility
- **Mock spec pattern**: Use spec=[] parameter in Mock objects to prevent auto-creation of unwanted attributes
- **Priority IntEnum ordering**: HIGH=0, NORMAL=1, LOW=2 — lower numeric value = higher priority for natural heapq ordering
- **Aging rate 0.1**: LOW priority becomes NORMAL after 10s, HIGH after 20s — prevents starvation without being too aggressive
- **entry_count tie-breaker**: Guarantees FIFO ordering for same effective priority in priority queue
- **BLOCK_SIZE=32**: 32 tokens per block for paged KV cache — ~4% internal fragmentation vs 60-80% with pre-allocation
- **Stack-based free list**: O(1) allocation by popping from end of list
- **Eviction targets ref_count=0 not in free list**: Primarily prefix cached blocks that stayed allocated
- **Hash chaining for position context**: Same tokens at different positions produce different hashes via prev_hash parameter
- **Adaptive timing defaults**: idle_wait_ms=50.0, load_wait_ms=5.0 — longer wait when idle accumulates requests
- **Memory error retry delay**: Sleep idle_wait_ms after MemoryError to prevent busy loop
- **output_queue None signal**: None in output queue signals request completion
- **Sequential generation in BatchInferenceEngine**: mlx-lm doesn't support true batched generation (Issue #548) — generate sequentially in dedicated thread
- **Scheduler singleton pattern**: Module-level singleton with init/get/reset functions for testing
- **Endpoint-based priority**: /v1/batch/* gets LOW priority, others NORMAL — system-determined, not client-requested
- **Feature flag for batching**: enable_batching=False by default until stable
- **Graceful fallback**: Fall back to direct inference if scheduler unavailable
- **Callback-based benchmarking**: Benchmark accepts generate/submit callbacks — agnostic of implementation
- **Linear percentile interpolation**: calculate_percentile uses linear interpolation for smooth values

See PROJECT.md Key Decisions table for full history.

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 9 complete:**
- All 8 plans executed (types, priority queue, block manager, prefix cache, batch inference, scheduler manager, benchmark, gap closure)
- 174 batching tests passing
- Documentation complete (docs/BATCHING.md)
- Gap closed: configure_scheduler now wires BatchInferenceEngine to scheduler
- Ready for manual throughput testing with actual MLX inference

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

Last session: 2026-01-29T10:05:51Z
Stopped at: Completed 09-08-PLAN.md (Gap closure - inference engine wiring) - Phase 9 complete
Resume file: None
Next: Phase 10 (Cloud Fallback)
