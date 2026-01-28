# Phase 9: Continuous Batching & Paged KV Cache - Context

**Gathered:** 2026-01-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement continuous batching scheduler and paged KV cache for 2-4x throughput improvement. This is performance infrastructure — users experience faster multi-request throughput and better memory efficiency. The system processes multiple inference requests simultaneously during each token generation step.

</domain>

<decisions>
## Implementation Decisions

### Batching strategy
- New requests wait for current generation step to complete, then join next batch (no mid-generation interruption)
- Immediate request removal when generation completes — slot opens for waiting request (true continuous batching)
- Memory-based batch size limit by default, admin can override with fixed limit in configuration
- Adaptive timing: wait longer when idle to accumulate requests, process immediately under load

### Memory allocation
- 32-token block size for paged KV cache (balance between flexibility and efficiency)
- 85% memory pressure threshold triggers eviction consideration
- When memory full: evict prefix cache blocks first, then queue request if still not enough
- Block pool allocation strategy: Claude's discretion (pre-allocate vs dynamic based on Apple unified memory behavior)

### Request prioritization
- Three priority levels: high, normal, low
- Priority determination: API key tier (admin assigns) + endpoint-based override
  - Script/batch keys get 'low' tier
  - User-facing keys get 'normal' tier
  - Special keys get 'high' tier
  - Endpoints can enforce priority floor (e.g., /v1/batch/... forces low priority)
- Aging mechanism: waiting requests gradually increase effective priority to prevent starvation
- Queue overflow behavior: Claude's discretion

### Prefix caching
- Caching aggressiveness: Claude's discretion (threshold-based or cache all)
- No TTL — LRU eviction only under memory pressure
- Prefix matching strategy: Claude's discretion (efficient implementation)
- Per-model prefix cache (no cross-model sharing)

### Claude's Discretion
- Block pool allocation strategy (pre-allocate vs dynamic growth)
- Prefix caching aggressiveness (threshold vs cache all)
- Prefix matching implementation (exact, longest common, hash-based)
- Queue overflow behavior (reject, evict lowest priority, or backpressure)
- Adaptive timing parameters (idle wait window, load detection)

</decisions>

<specifics>
## Specific Ideas

- "We can't give everyone the option [for high priority], or everyone will use it" — priority must be system-determined, not client-requested
- Use case example: background RAG ingestion (low), live user queries (normal), controlled fast access (high via special API key)
- vLLM-MLX showed 3.4x throughput improvement (328→1112 tok/s) — this is the target

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-continuous-batching*
*Context gathered: 2026-01-28*
