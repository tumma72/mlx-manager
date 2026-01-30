# Phase 12: Production Hardening - Context

**Gathered:** 2026-01-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Observability, error handling, timeouts, audit logging, and performance validation for production deployment. Includes LogFire integration, unified error responses, configurable timeouts, request audit logs with admin panel, and CLI benchmarks with documented performance results.

</domain>

<decisions>
## Implementation Decisions

### Audit Log Visibility
- **Detail level:** Claude's discretion (likely essential + request context)
- **Content logging:** Never store prompt/response content — privacy-first, metadata only
- **Retention:** 30 days fixed
- **Error logging:** Same schema as success — status field indicates outcome, no separate error table

### Error Response Format
- **Format:** Claude's discretion, leaning toward RFC 7807 Problem Details (given dual OpenAI/Anthropic APIs) — research LogFire compatibility
- **Detail level:** Contextual detail — helpful info without exposing internals (e.g., "Model 'xyz' not found")
- **Trace ID:** Always include request_id in every error response for log correlation
- **Streaming errors:** Claude's discretion on SSE error event vs connection close

### Admin Log Panel UX
- **Data loading:** Infinite scroll
- **Filters:** Essential only — model, backend type, status (success/error), time range (LogFire web client provides advanced features)
- **Live updates:** WebSocket streaming for real-time log entries
- **Export:** Standard log format (easily parseable by common tools)

### Timeout Behavior
- **Messaging:** Claude's discretion on actionable guidance
- **Timeout tiers:** Per endpoint type — Chat: 15 min, Embeddings: 2 min, Completions: 10 min
- **Per-request control:** No — admin-configured defaults only
- **Partial response on timeout:** Discard — timeout = error, no partial content
- **Progress indication:** Claude's discretion
- **Configuration UI:** Expose per-endpoint timeouts in Settings UI
- **Cloud vs local:** Same timeouts per endpoint (no separate cloud timeouts)
- **Logging:** Timeout is an error status like any other failure

### Performance Testing
- **Scope:** Full routing matrix — local inference, OpenAI cloud, Anthropic cloud, failover scenarios
- **Coverage:** Throughput (tok/s) with batching on/off, single vs multi-model, network latency for cloud routing
- **Test models:** Size tiers — small (3B), medium (7B), large (14B+) to demonstrate scaling
- **Cloud benchmarks:** Include round-trip latency for OpenAI/Anthropic
- **Output:** CLI benchmarks documented in PERFORMANCE.md for v1.2 release

### Claude's Discretion
- Audit log detail level (within essential + context bounds)
- Error response format choice (RFC 7807 preferred, validate LogFire support)
- Streaming error handling approach
- Timeout messaging wording
- Progress indication during long requests

</decisions>

<specifics>
## Specific Ideas

- "LogFire web client provides more features which we do not need to reinvent" — keep admin panel simple, rely on LogFire for advanced filtering
- PERFORMANCE.md should demonstrate v1.2's high-performance server capabilities to attract users
- Benchmarks should show comparable performance to vLLM (CUDA) on Apple Silicon

</specifics>

<deferred>
## Deferred Ideas

- **UI-based performance testing:** "Test Performance" button on server tiles that runs benchmark series, stores results in database, and displays throughput on Profile/Server tiles — post-Phase 12 feature
- **Configurable retention:** Admin-adjustable log retention period — decided on fixed 30 days for now

</deferred>

---

*Phase: 12-production-hardening*
*Context gathered: 2026-01-30*
