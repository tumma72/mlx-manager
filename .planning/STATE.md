# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-27)

**Core value:** Enable developers to easily discover, download, configure, and run MLX models locally without command-line complexity — making local AI accessible and manageable.
**Current focus:** Milestone v1.2 - MLX Unified Server (COMPLETE)

## Current Position

Phase: 16 of 16 (MLX Manager Architecture Compliance)
Plan: 2 of 2 complete
Status: Phase complete
Last activity: 2026-02-07 - Completed 16-02-PLAN.md (Frontend SSE/WS Auth & Parser Cleanup)

Progress: [████████████████] 100% (2 of 2 plans in Phase 16)

**UAT Gaps Fixed:**
1. ~~Empty responses with thinking models~~ - FIXED (15-04: StreamingProcessor redesign)
2. ~~Thinking content not streamed~~ - FIXED (15-04: now yields reasoning chunks)
3. ~~All servers show same memory values~~ - FIXED (15-05: per-model memory calculation)
4. ~~Stop button does nothing~~ - FIXED (15-05: calls model.unload endpoint)
5. ~~Gemma vision model crashes~~ - FIXED (15-06: detection uses config-based approach)
6. ~~Model downloads hanging~~ - FIXED (15-07: immediate SSE yield + timeout)
7. ~~Obsolete profile fields clutter~~ - FIXED (15-08: removed 14 obsolete fields, added generation params)

## Milestone v1.2 Summary

**Goal:** MLX Unified Server (pivoted from adapter/proxy approach)
**Status:** Complete - all 9 phases verified
**Phases:** 9 phases (7-15)
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
- **BackendType string enum**: Enables direct JSON serialization for LOCAL, OPENAI, ANTHROPIC values
- **Priority-based pattern matching**: Higher priority patterns checked first for model routing
- **Anthropic max_tokens required**: Unlike OpenAI's optional, Anthropic requires max_tokens with Field(ge=1) and no default
- **System message separate field**: Anthropic stores system in dedicated field, not in messages array
- **Anthropic temperature 0.0-1.0**: Stricter bounds than OpenAI's 0.0-2.0
- **Custom AsyncCircuitBreaker instead of pybreaker**: pybreaker's async support requires Tornado; our implementation is async-native and simpler
- **Circuit breaker per-client instance**: Each CloudBackendClient has independent circuit state
- **Half-open state for gradual recovery**: Allows one test request after reset_timeout to check if backend recovered
- **typing.Any for content extraction**: Handles mixed Pydantic models and dict representations in protocol translator
- **InternalRequest dataclass**: Unified internal format for inference service consumption
- **cast() for httpx response.json()**: Type annotation fix for response.json() returning Any
- **Anthropic system message extraction**: System message extracted to separate 'system' field per Anthropic API spec
- **SSE event type from data.type**: Event type in Anthropic streaming comes from JSON data.type, not event: line
- **Routing before batching**: Cloud routing check happens before batching check — cloud routing has higher priority when both enabled
- **Automatic routing fallback**: Any routing exception falls through to local inference path
- **PBKDF2HMAC with 1.2M iterations**: Industry standard for key derivation from jwt_secret for Fernet encryption
- **AuthLib JWE for encryption**: A256KW+A256GCM replaces Fernet; SHA-256(jwt_secret) as key, no salt file needed
- **Static routes before dynamic**: /rules/priorities and /rules/test must come before /rules/{rule_id}
- **ServerConfig singleton**: Use id=1 for global pool settings, created on first access
- **Local API helpers in components**: When shared API client not available (parallel execution), define local fetch wrappers in component — refactor later
- **Memory mode conversion**: When toggling between % and GB, convert value to equivalent in new mode with proper clamping
- **svelte-sortable-list for drag-drop**: @rodrigodagostino/svelte-sortable-list v2 for Svelte 5 compatible drag-drop with accessibility
- **Optimistic reorder with rollback**: Update local state immediately on drag-drop, rollback from server on error
- **$derived.by() for type narrowing**: Use $derived.by() instead of $derived() for complex derivations requiring TypeScript type narrowing
- **Sliders icon for settings**: Use Sliders icon for settings link (Settings icon already used for Profiles)
- **settingsStore for provider tracking**: Centralized store tracks configured providers with helper methods
- **$derived.by() for reactive props**: Use function form for props that need reactivity when parent changes
- **ConfirmDialog pattern for deletes**: requestDelete opens dialog, confirmDelete executes, cancelDelete clears state
- **RFC 7807 Problem Details**: All API errors return type/title/status/detail/instance/request_id format
- **request_id format**: req_{12-char-hex} prefix makes IDs identifiable in logs
- **Type ignore for FastAPI handlers**: Starlette type stubs expect generic Exception but specific types work at runtime
- **LogFire configured BEFORE instrumented imports**: configure() must be called before importing instrumented libraries for proper tracing
- **send_to_logfire='if-token-present'**: Enables development without a LogFire token
- **Graceful LLM client instrumentation**: try/except for optional openai/anthropic packages
- **Per-endpoint timeouts**: Chat 15min, Completions 10min, Embeddings 2min via asyncio.wait_for
- **Streaming timeout error events**: SSE error event with type/message before close for graceful degradation
- **Privacy-first audit logging**: AuditLog model has no prompt/response content fields, only metadata
- **Background audit writes**: asyncio.create_task for non-blocking logging
- **track_request context manager**: Wraps request lifecycle, auto-logs on exit with status/error
- **Timeout settings as key-value pairs**: Individual keys (timeout_chat_seconds, etc.) in Setting table for flexibility
- **Range + Number inputs for timeouts**: Dual control pattern for quick slider adjustments and precise numeric entry
- **Backend detection from model name**: gpt/o1->openai, claude->anthropic, else local for benchmark classification
- **Typer-based CLI for benchmarks**: Consistency with existing mlx-manager CLI patterns
- **Async httpx client for benchmarks**: Matches async patterns in inference service
- **Mount at /v1 with no router prefix**: Prevents double prefix /v1/v1/* when routers already had /v1 prefix
- **Lazy app initialization via __getattr__**: Module-level app deferred to prevent LogFire config on import
- **Embedded database path**: When embedded_mode=True, MLX Server uses MLX Manager's database for shared audit logs
- **Parser options stubs**: With embedded server, parser options endpoints return empty lists for backward compatibility
- **Legacy endpoint messages**: start/stop/restart endpoints kept with informative messages about embedded mode
- **Direct async generator consumption**: Chat router calls generate_chat_completion() directly instead of httpx proxy
- **All profiles selectable for chat**: With embedded server, models load on-demand; no "running server" filter needed
- **Cast for Union return types**: Use cast(AsyncGenerator[dict, None], gen) for inference functions returning Union types
- **update_memory_limit sets MLX limit**: Calls mx.set_memory_limit() directly for immediate memory limit effect
- **apply_preload_list marks evictable**: Models not in preload list have preloaded=False for LRU eviction
- **refresh_rules clears backends**: Cached cloud backends closed and cleared on rule/credential changes
- **Adapters inherit from DefaultAdapter**: Makes existing adapters (Llama, Qwen, Mistral, Gemma) protocol-compliant without duplicating code
- **ToolChoiceOption as type alias**: `Literal["none", "auto", "required"] | dict[str, Any] | None` provides flexibility for tool_choice parameter
- **ChatMessage content nullable**: Allows tool-only assistant messages where content is None but tool_calls is populated
- **Optional protocol methods pattern**: Add to Protocol, implement defaults in DefaultAdapter, adapters inherit
- **Draft202012Validator for JSON Schema**: Use modern JSON Schema draft for structured output validation
- **Error path format**: dot.notation for objects, [N] for arrays in validation error paths
- **Type coercion in validation**: Schema-guided coercion for LLM outputs (string "5" -> int 5)
- **GLM4 deduplication via MD5 content hash**: GLM4 has known duplicate tag bug - hash name+args to deduplicate
- **Module-level parser instances**: Stateless parsers instantiated at module level to avoid repeated allocation
- **Parsers return empty list, adapters convert to None**: Consistent with protocol semantics
- **Tool injection into system message**: Tools are formatted by adapter and appended/prepended to system message
- **Post-generation tool call parsing**: Tool calls detected after full response generated, not during streaming
- **Streaming buffers for tool detection**: Accumulated text allows tool call detection in final chunk
- **Structured output validation at API layer**: Validation happens in chat.py with 400 error on schema failure
- **Pydantic models for ResponseProcessor**: Use BaseModel for ParseResult and ToolCall for type safety and model_dump() serialization
- **Single-pass extraction with span removal**: Collect all matches, merge overlapping spans, remove in reverse order to preserve indices
- **Callback-based pattern registration**: ResponseProcessor uses register_tool_pattern(pattern, callback) for extensibility
- **Partial marker buffering**: StreamingProcessor buffers incomplete markers until next token determines if pattern or content
- **Recursive after-pattern processing**: StreamingProcessor.feed() recursively processes content after pattern ends when in same token
- **Python tag pattern in streaming**: Include <|python_tag|>...<|eom_id|> in StreamingProcessor pattern filtering
- **ApiType enum for protocol selection**: Explicit api_type field (openai/anthropic) on CloudCredential instead of inferring from BackendType
- **Cache by credential ID**: Cloud backend cache uses credential ID instead of BackendType to support multiple providers of same API type
- **Backwards compatibility for api_type**: Credentials without api_type fall back to API_TYPE_FOR_BACKEND mapping
- **Don't pre-populate base_url in UI**: Let placeholder show default URL, send undefined to use server-side default
- **Database migration skips non-existent tables**: PRAGMA table_info returns empty for missing tables; skip migration to avoid ALTER TABLE errors
- **api_type default 'openai' in migration**: Backward compatibility for existing cloud_credentials rows
- **Catch multiple exceptions for enable_thinking**: TypeError, ValueError, KeyError, AttributeError all handled for tokenizer compatibility
- **DEBUG not WARNING for enable_thinking fallback**: Expected behavior for older tokenizers, not a warning condition
- **Golden file testing pattern**: Use fixtures/golden/{family}/*.txt for model output validation; parametrized tests auto-discover files
- **Immediate SSE yield pattern**: Yield initial status before blocking operations in async generators to prevent hang appearance
- **30s timeout for HF dry_run**: Wrap snapshot_download dry_run in asyncio.wait_for to prevent indefinite blocking
- **StreamEvent dataclass for OpenAI-compatible streaming**: Return StreamEvent with reasoning_content and content fields instead of tuple, following o1/o3 API spec
- **image_token_index detection**: Gemma 3 uses image_token_index instead of image_token_id for vision detection
- **Shared detect_multimodal()**: MLX server detection calls shared utility for badge/loading consistency
- **Model type mismatch error**: Clear message in vision.py guides user to unload/reload when detection was wrong
- **Loguru for structured logging**: Replace standard logging with Loguru for efficiency, auto-stacktraces via .exception(), simpler configuration
- **AuthLib jose for JWT**: Replace pyjwt with authlib.jose.jwt — unified auth library, accepts any valid HMAC signature
- **DecryptionError backward compat**: Aliased as InvalidToken for existing consumers in settings router
- **Separate log files by component**: mlx-server.log for inference, mlx-manager.log for app — easier debugging of distinct components
- **InterceptHandler for third-party compatibility**: Redirect standard logging to Loguru to capture third-party library logs

- **E2E tiered test infrastructure**: pytest markers (e2e, e2e_vision_quick, e2e_vision_full) with addopts excluding E2E from default run
- **Fallback model resolution for E2E**: Prefer qat variants over DWQ for Gemma models due to VisionConfig incompatibility in mlx-vlm 0.3.11
- **Manual pool init in ASGI tests**: httpx ASGITransport does not trigger FastAPI lifespan; initialize ModelPoolManager explicitly in fixture
- **Cleanup models after each E2E test**: Unload all loaded models between tests to prevent 7-16GB vision model memory accumulation

- **threading.Event for download cancellation**: Use threading.Event for cross-task cancellation signaling between async endpoints and executor-based downloads
- **Paused downloads not auto-resumed**: Paused downloads stay paused on server restart — user must explicitly click Resume
- **Inline cancel confirmation**: Cancel confirmation uses inline Confirm/Keep buttons rather than a modal dialog

- **Query-param JWT for SSE/WS**: Browser EventSource cannot send custom headers; pass token as ?token=<jwt> query parameter
- **WebSocket auth before accept**: Validate JWT before websocket.accept(), close(1008) on failure per RFC 6455
- **Public pool API (get_loaded_model)**: Routers use public method instead of accessing pool._models directly
- **Direct mock WebSocket tests**: Replace SyncTestClient-based WS tests with direct function calls to avoid lifespan DB issues

- **Conditional token in SSE/WS URLs**: Use `token ? \`?token=${token}\` : ""` for defensive handling when authStore.token is null
- **Unauthenticated WS test**: Explicit test verifies WebSocket creation without token works correctly

- **Inner tokenizer extraction for mlx-embeddings**: Use getattr(tokenizer, '_tokenizer', tokenizer) to access inner HF tokenizer since batch_encode_plus removed in transformers v5
- **Manual mlx.array conversion for batch encoding**: TokenizerWrapper is not callable; use inner tokenizer __call__ with return_tensors=None then convert to mx.array

- **Thinking model token budget for E2E**: System message tests use 512 max_tokens because Qwen3 thinking models consume tokens for reasoning before visible output
- **Cross-protocol E2E flexible tool assertion**: Tool call test accepts either tool_call or content response since small models may not always trigger tool calls

- **Audio models tokenizer=None**: Audio models loaded via mlx-audio don't use text tokenizers; tokenizer field set to None in LoadedModel
- **mlx-audio load_model auto-detection**: mlx_audio.utils.load_model auto-detects TTS vs STT from config and name patterns
- **TTS model.generate() returns GenerationResult iterable**: Each result has .audio (mx.array), .sample_rate, .audio_duration fields
- **STT temp file for audio input**: generate_transcription expects file path not bytes; write to tempfile then unlink
- **WAV default for TTS**: WAV requires no external dependencies (uses miniaudio); MP3 needs ffmpeg

See PROJECT.md Key Decisions table for full history.

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 14 gap closure (v1.2 blocking):**
- Plan 01-06: Complete
- Plan 07: Complete (Unified ResponseProcessor - tool call markers now removed)
- Plan 08: Complete (StreamingProcessor - patterns filtered during streaming)
- Plan 09: Complete (Generic OpenAI/Anthropic-compatible providers)

**Phase 13 MLX Server Integration:**
- Plans 01-05: Complete (awaiting final verification)

**Critical bugs fixed:**
1. ~~Tool call markers remain in response content~~ - FIXED (ResponseProcessor removes all spans)
2. ~~Streaming doesn't extract reasoning~~ - FIXED (StreamingProcessor filters and extracts)
3. ~~Multiple parsing passes inefficient~~ - FIXED (now single-pass)
4. ~~Database missing api_type/name columns~~ - FIXED (15-02: migration added)
5. ~~Vision processor attribute access broken~~ - FIXED (15-02: getattr pattern in all adapters)
6. ~~Qwen enable_thinking crashes~~ - FIXED (15-02: catch all exception types)
7. ~~Streaming logs every token at INFO~~ - FIXED (15-02: changed to DEBUG)
8. ~~Model downloads hang~~ - FIXED (15-07: immediate SSE yield + 30s timeout)

## Research Documents

**Feasibility Study:** `.planning/research/MLX-SERVER-FEASIBILITY.md`
- Confirms feasibility with high confidence
- Documents vLLM performance techniques (PagedAttention, continuous batching)
- Proposes architecture and technology stack
- Includes implementation roadmap aligned with phases

## Known Tech Debt (Carried Forward)

1. **Throughput metrics not available** — Will be solved by our own server with proper metrics
2. ~~**mlx-openai-server v1.5.0 regression**~~ — Resolved: mlx-openai-server removed
3. **Download completion UX** — Doesn't auto-refresh local models list after download completes
4. ~~**RunningInstance and servers router refactoring**~~ — Resolved: Plan 13-02
5. **Pre-existing mypy errors** — 4 type errors in chat.py, system.py, settings.py (unrelated to recent work)

## Session Continuity

Last session: 2026-02-07T12:44:53Z
Stopped at: Completed 16-02-PLAN.md (Frontend SSE/WS Auth & Parser Cleanup)
Resume file: None
Next: Phase 16 complete. All architecture compliance items resolved.

### Roadmap Evolution

- Phase 15 added: Code Cleanup & Integration Tests (dead code removal, bug fixes, integration tests for ResponseProcessor)
- Phase 16 added: MLX Manager Architecture Compliance (SSE/WS auth, JWT warning, deprecated endpoint removal, router decoupling)
