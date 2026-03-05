# MLX Manager v1.2 — Pre-Release Code Review

**Date:** 2026-03-05
**Scope:** Full codebase review across 5 dimensions
**Status:** Must-fix items RESOLVED (2026-03-05)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Adherence](#1-architecture-adherence)
3. [Design Patterns](#2-design-patterns)
4. [Clean Code](#3-clean-code)
5. [Security](#4-security)
6. [Performance](#5-performance)
7. [Triage Matrix](#triage-matrix)

---

## Executive Summary

| Dimension | Findings | Critical | High | Medium | Low |
|-----------|----------|----------|------|--------|-----|
| Architecture | 15 | 0 | 0 | 3 | 0 |
| Design Patterns | 12 | 0 | 1 | 5 | 4 |
| Clean Code | 18 | 0 | 0 | 6 | 12 |
| Security | 10 | 0 | 3 | 4 | 2 |
| Performance | 12 | 1 | 2 | 7 | 2 |

**Overall assessment:** The codebase is solid for a v1.2 release. The composable adapter architecture is a clear improvement over the original blueprint. The main concerns are: (1) one critical performance issue with Metal thread affinity, (2) three high-severity security issues around unauthenticated inference endpoints and SSRF, and (3) a god-class pattern in `ModelPoolManager` that will hinder future development.

---

## 1. Architecture Adherence

The implementation has **evolved significantly** from ARCHITECTURE.md. The core 3-layer pipeline concept is preserved in spirit, but the structure differs fundamentally: a **single unified adapter with data-driven config** instead of the planned **ABC class hierarchy**.

### Positive Evolutions (9 findings)

These are deliberate improvements over the blueprint that should be kept:

| # | What Changed | Why It's Better |
|---|-------------|-----------------|
| A-1 | `ModelAdapter` is a concrete class, not an ABC hierarchy | Data-driven `FamilyConfig` + strategy functions eliminate combinatorial explosion of 11+ subclass files |
| A-2 | No per-family adapter files (`adapters/families/`) | 3 files (`composable.py`, `configs.py`, `strategies.py`) replace 10+ planned files |
| A-4 | IR types use Pydantic `BaseModel`, not `@dataclass` | Consistent with project-wide Pydantic standardization |
| A-5 | Additional IR types (`InternalRequest`, `InferenceResult`, `RoutingOutcome`) | Properly extend IR for cloud routing and protocol translation |
| A-6 | `ProtocolFormatter` has 5 methods, not 2 | Richer interface handles full streaming lifecycle (start/events/end) needed for Anthropic protocol |
| A-12 | Per-endpoint routers (`api/v1/chat.py`, etc.) instead of per-protocol | More granular and RESTful |
| A-13 | `AdapterResult` is concrete Pydantic, not ABC | Polymorphic hierarchy works the same way via regular inheritance |
| A-14 | `EmbeddingResult` adds `total_tokens` field | Essential for usage reporting |
| A-15 | `TranscriptionResult` adds `language` field | Standard STT feature |

### Concerning Drifts (3 findings)

These deviate from architectural intent in ways that create technical debt:

#### A-8: ModelAdapter is a God Class (~950 lines)

**Architecture says:** Adapter handles input preparation and output processing. Inference service orchestrates generation.

**Code does:** `ModelAdapter` in `composable.py` owns the **entire** generation pipeline via `generate()`, `generate_step()`, `generate_embeddings()`, `generate_speech()`, `transcribe()` — five modalities with full orchestration.

```
composable.py:513   — generate()           ~80 lines
composable.py:591   — generate_step()      ~130 lines
composable.py:725   — generate_embeddings() ~70 lines
composable.py:794   — generate_speech()     ~80 lines
composable.py:874   — transcribe()          ~80 lines
```

**Impact:** Concentrates too many responsibilities. Harder to test pipeline stages in isolation. Makes batching integration difficult (see A-10).

#### A-9: Inference Service Has Dual Legacy/Modern Code Paths

**Architecture says:** Thin orchestrator delegating to adapter methods.

**Code does:** `services/inference.py` (~760 lines) maintains two parallel paths:
- **Modern path** (lines 238-279): delegates to `adapter.generate_step()`
- **Legacy path** (lines 281-375): direct mlx-lm/mlx-vlm calls with manual stream processing

The legacy path duplicates logic already in the adapter. `_prepare_generation()` always sets `ctx.messages`, so the legacy path is effectively dead code.

#### A-10: Batched Path Bypasses 3-Layer Pipeline

**Architecture says:** Pipeline flow: API Layer -> Adapter Pipeline -> Batching -> Pool -> MLX.

**Code does:** In `api/v1/chat.py` lines 487-536, the batched path:
1. Skips `adapter.prepare_input()` — uses `apply_chat_template()` directly
2. Skips `StreamProcessor` entirely — no tool/thinking extraction
3. Skips `ProtocolFormatter` — formats chunks manually
4. Only supports OpenAI protocol — no Anthropic batching

**Impact:** Batched requests cannot extract tool calls or thinking content. Protocol-neutral formatting is lost.

### Expected Gaps (1 finding)

#### A-11: Paged KV Cache Not Connected to GPU Memory

`PagedBlockManager` exists with proper block allocation/deallocation, but it manages abstract block IDs — not connected to MLX's actual KV cache tensors. This is a Phase 9 item and expected.

### Recommendation

Update ARCHITECTURE.md to reflect the composable adapter pattern (clearly superior). Address A-8 by extracting generation orchestration back to the inference service, which would also fix A-9 (eliminate legacy path) and enable A-10 (batching through the pipeline).

---

## 2. Design Patterns

### Good Patterns Found

| Pattern | Location | Notes |
|---------|----------|-------|
| Data-driven Strategy | `adapters/configs.py` + `strategies.py` | Textbook `FamilyConfig` -> strategy dispatch. No if/elif chains |
| Probe Strategy Protocol | `services/probe/strategy.py` | Clean registry with `register_strategy()` / `get_probe_strategy()` |
| Composable Factory | `adapters/composable.py:958` | `create_adapter()` — config lookup, fallback, clean delegation |
| Pub/Sub Observer | `services/loading_progress.py` | `subscribe()` / `unsubscribe()` / `emit()` with late-join catch-up |
| Formatter ABC | `services/formatters/base.py` | Clean protocol abstraction with proper polymorphism |

### Findings

#### DP-1 [HIGH]: `ModelPoolManager` is a God Class (1231 lines, 5+ responsibilities)

**File:** `mlx_server/models/pool.py`

Handles: model loading (4 types), LRU eviction, memory management, profile settings, LoRA loading, size estimation, capabilities attachment, adapter creation, preloading, cache key management.

**Suggested decomposition:**
1. `ModelLoader` — type-detection + loading dispatch
2. `AdapterConfigurator` — parser resolution + profile settings + adapter creation
3. `ModelPoolManager` — remains as cache/eviction orchestrator delegating to the above

This single refactor would also fix DP-4 (duplicated loading), DP-8 (deep nesting), and DP-9 (deferred imports).

#### DP-2 [MEDIUM]: Three Inconsistent Singleton Styles

| Style | Examples | Issue |
|-------|----------|-------|
| Bare module-level instance | `health_checker`, `hf_client`, `settings` | Can't reset for testing, constructor runs at import |
| `get_*()` / `reset_*()` (lazy) | `get_loading_progress()`, `get_metrics()` | Correct pattern |
| Direct module attribute assignment | `pool.model_pool = ModelPoolManager(...)` in `main.py:182` | Bypasses encapsulation |

**Fix:** Standardize on Style 2 (`get_*()` / `reset_*()`) for all singletons.

#### DP-3 [MEDIUM]: Module-Level Mutable State Without Thread Safety

**File:** `services/hf_client.py:57-84`

```python
_cancel_events: dict[str, threading.Event] = {}

def register_cancel_event(download_id: str) -> threading.Event:
    event = threading.Event()
    _cancel_events[download_id] = event  # No lock
    return event
```

Mutated from async main thread AND executor threads. Python's GIL provides accidental safety but breaks with free-threaded Python 3.13+.

**Fix:** Wrap with `threading.Lock`.

#### DP-4 [MEDIUM]: Duplicated Model Loading if/elif Chains

**File:** `pool.py` — same VISION/EMBEDDINGS/AUDIO/TEXT dispatch at lines 456-486 AND 999-1024.

**Fix:** Extract `_load_by_type(model_id, model_type)` helper.

#### DP-5 [MEDIUM]: `ProbingCoordinator` Accesses Pool Private Attributes

**File:** `services/probe/coordinator.py:51-54`

```python
was_preloaded = model_id in self._pool._models          # private!
original_settings = self._pool._profile_settings.get(model_id)  # private!
```

**Fix:** Add `pool.is_loaded(model_id)` and `pool.get_profile_settings(model_id)` public methods.

#### DP-6 [MEDIUM]: `model_type` Used as String Instead of Enum

`LoadedModel.model_type` is `str` (`pool.py:40`), `ModelAdapter._model_type` is `str` (`composable.py:65`), yet `ModelType` enum exists. Code does constant `model_type.value` / `ModelType(cached_type)` conversions and string comparisons like `self._model_type == "vision"`.

**Fix:** Use `ModelType` enum directly in `LoadedModel` and `ModelAdapter`.

#### DP-7 [LOW]: Template Strategy Code Duplication

`strategies.py` — `qwen_template`, `glm4_template`, `mistral_template`, `liquid_template` all share the same try/fallback pattern. Extract `_apply_with_fallback()` helper.

#### DP-8 [LOW]: Deep Nesting in `_load_model()` (326 lines, 5 levels)

**Fix:** Break into `_resolve_model_type()`, `_load_raw_model()`, `_create_and_configure_adapter()`.

#### DP-9 [LOW]: Deferred Imports Masking Circular Dependencies

`pool.py` -> `database.py`, `pool.py` -> `parsers`, `health_checker.py` -> `pool` — all use function-level imports to avoid circular imports. The `pool.py` -> `database.py` cycle is particularly concerning since pool is in `mlx_server` but reaches into the manager's database layer.

#### DP-10 [LOW]: Long Parameter Lists (8-11 params) in Inference Functions

`generate_chat_stream()` and `generate_chat_complete_response()` both take 8 identical params. `_stream_completion()` takes 11.

**Fix:** Group generation parameters into a `GenerationParams` value object.

#### DP-11 [LOW]: HealthChecker is a No-Op Debug Logger

`health_checker.py:46-61` — `_check_model_pool()` imports pool, calls `get_memory_usage()`, logs at debug level. Provides no operational value.

**Fix:** Either make it emit health events or remove the dead complexity.

---

## 3. Clean Code

### Hardcoded Values (8 findings)

| # | Location | Value | Suggestion |
|---|----------|-------|------------|
| CC-1a | `main.py:321-326` | CORS ports `5173`, `4173` | Move to `config.py` |
| CC-1b | `settings.py:715-717` | Timeouts `900`, `600`, `120` | Import from `TimeoutSettings` defaults |
| CC-1c | Multiple files | `timeout=10.0`, `timeout=5.0` | Constants `HTTP_TEST_TIMEOUT`, `CLI_STATUS_TIMEOUT` |
| CC-1d | `cli.py:295` | `max_models=2` | `CLI_MAX_MODELS = 2` |
| CC-1e | `models.py:244` | `sleep(60)` cleanup delay | `DOWNLOAD_TASK_CLEANUP_DELAY = 60` |
| CC-1f | `composable.py:560,643` | `timeout=600.0` vision gen | `VISION_GENERATION_TIMEOUT = 600.0` |
| CC-1g | `composable.py:754,761` | `max_length=512` embeddings | `EMBEDDINGS_MAX_LENGTH = 512` |
| CC-1h | `settings.py:217` + `cloud/anthropic.py` | `"2023-06-01"` API version | `ANTHROPIC_API_VERSION` shared constant |

### DRY Violations (8 findings)

| # | Location | Repetitions | Fix |
|---|----------|-------------|-----|
| CC-2a | `settings.py` | Cache invalidation pattern x6 | Extract `_invalidate_routing_cache()` |
| CC-2b | `settings.py:555-571, 646-662` | Preloaded profiles query x2 | Extract `_get_preloaded_profiles()` |
| CC-2c | `pool.py:456-486, 999-1024` | Model loading if/elif x2 | Extract `_load_model_by_type()` |
| CC-2d | `pool.py` | Adapter creation logic x2 | Extract `_create_adapter_for_model()` |
| CC-2e | `cli.py:591-596, 612-617` | Capabilities formatting x2 | Extract `_format_capabilities()` |
| CC-2f | `settings.py:513-524` + `cloud/router.py:116-129` | Pattern matching x2 (uses `re.match` vs `re.fullmatch` — **possible bug**) | Unify into shared utility |
| CC-2g | `settings.py` | Provider connection test pattern x2 | Extract `_test_api_connection()` |
| CC-2h | `servers.py` | Profile+model lookup x4 | Extract `_get_profile_with_model()` |

#### CC-2f Detail: Potential Bug in Pattern Matching

`settings.py:_matches_pattern()` uses `re.match()` (matches prefix only), while `cloud/router.py:_pattern_matches()` uses `re.fullmatch()` (matches entire string). This means routing rules may match differently depending on which code path evaluates them.

### Loops That Should Be Comprehensions (3 findings)

| # | Location | Current | Suggested |
|---|----------|---------|-----------|
| CC-3a | `cli.py:511-514` | `for step... extend` | `[d for step in steps if step.diagnostics for d in step.diagnostics]` |
| CC-3b | `cli.py:591-596` | `cap_strs` append loop | Comprehension with ternary |
| CC-3c | `composable.py:829-833` | `audio_segments` append loop | `[result.audio for result in results]` |

### Long if/elif Chains (3 findings)

| # | Location | Branches | Fix |
|---|----------|----------|-----|
| CC-4a | `hf_api.py:65-77` | 5 (quantization detection) | Dict-based lookup with pattern tuples |
| CC-4b | `tool_call.py:625-639` | 4 (value type parsing) | Try/except + dict for booleans |
| CC-4c | `pool.py:456-486` | 4 (model type loading) | Strategy dict `_LOADERS: dict[ModelType, Callable]` |

### Missing Python Builtins (1 finding)

| # | Location | Current | Suggested |
|---|----------|---------|-----------|
| CC-5a | `models.py:139-142` | Manual loop to find existing task | `next((tid for tid, task in ... if ...), None)` |

### Functions Too Long (9 findings)

| File | Method | Lines | Primary Fix |
|------|--------|-------|-------------|
| `pool.py` | `_load_model` | ~200 | Split into load/adapter/capabilities |
| `pool.py` | `load_model_as` | ~120 | Share logic with `_load_model` |
| `composable.py` | `generate_step` | ~130 | Extract vision/text streaming |
| `composable.py` | `generate_speech` | ~60 | Acceptable |
| `anthropic.py` (formatter) | `parse_request` | ~130 | Extract block processing helpers |
| `anthropic.py` (formatter) | `stream_end` | ~165 | Extract block emission helpers |
| `cli.py` | `_probe_all` | ~65 | Acceptable |
| `settings.py` | `update_pool_config` | ~88 | Extract shared queries |
| `settings.py` | `test_provider_connection` | ~85 | Extract shared HTTP test helper |

### Inconsistent Naming (2 findings)

| # | Issue | Files |
|---|-------|-------|
| CC-7a | `db` vs `session` for DB dependency | `models.py`, `servers.py` use `db`; `settings.py` uses `session` |
| CC-7b | Redundant local re-imports of `get_model_pool` | `servers.py:118,148,181,218,448` despite top-level import at line 22 |

### Frontend Issues (3 findings)

| # | Location | Issue | Fix |
|---|----------|-------|-----|
| CC-8a | `client.ts` | Same fetch pattern repeated ~40 times | Extract `apiGet<T>()`, `apiPost<T>()`, etc. |
| CC-8b | Various `.svelte.ts` | Hardcoded polling intervals (10000, 5000, 30000, 3000) | Centralize in `constants.ts` |
| CC-8c | `client.ts:704-711, 737-742` | Audit log filter param building x2 | Extract `buildAuditParams()` |

---

## 4. Security

### Positive Security Practices

The codebase gets many things right:

| Practice | Location | Notes |
|----------|----------|-------|
| Path traversal protection | `main.py:374-376` | `resolved.is_relative_to(STATIC_DIR.resolve())` |
| Safe subprocess usage | All `subprocess.run` calls | List args, no `shell=True`, no user input in commands |
| SQL injection prevention | All routers | SQLModel/SQLAlchemy ORM with parameterized queries |
| Safe expression evaluation | `mcp.py` | `ast.parse` + safe evaluator, no `eval()` |
| No dangerous deserialization | Entire codebase | No `pickle`, `yaml.load`, `exec` |
| Sound encryption | `services/encryption.py` | AuthLib JWE (A256KW + A256GCM), auto-generated secret with 0o600 perms |
| MLX Server error handling | `generic_exception_handler` | Hides internal details, returns "An unexpected error occurred" |
| Default localhost binding | `config.py` | Both servers bind to `127.0.0.1`, not `0.0.0.0` |
| CORS not wildcard | `main.py` | Limited to specific localhost ports |

### Findings

#### S-1 [HIGH]: MLX Server Inference Endpoints Have No Authentication

**Files:** `mlx_server/api/v1/__init__.py:14-22`

All inference endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/messages`, `/v1/audio/*`) have zero authentication. The admin endpoints have **optional** token auth that defaults to open:

```python
# mlx_server/config.py
admin_token: str | None = Field(
    default=None,
    description="When None, admin endpoints are open.",
)
```

**Attack vector:** Any local process can make inference requests, load/unload models, and reconfigure the server. If the host is changed from `127.0.0.1` to `0.0.0.0`, network-wide access is possible.

**Fix:** Add API key authentication to inference endpoints. Don't default admin endpoints to open.

#### S-2 [HIGH]: Server-Side Request Forgery (SSRF) via Image URLs

**File:** `mlx_server/services/image_processor.py:63-66, 93-136`

```python
elif image_input.startswith(("http://", "https://")):
    img = await _fetch_image_from_url(image_input, client)
```

Fetches arbitrary URLs with `follow_redirects=True`. No URL validation.

**Attack vector:** `http://169.254.169.254/` (cloud metadata), `http://localhost:10242/api/...` (internal APIs), `http://192.168.x.x/...` (LAN scanning).

**Fix:** Block RFC 1918 private addresses, link-local (169.254.x.x), and localhost. Consider DNS resolution check before connecting.

#### S-3 [HIGH]: Arbitrary Local File Read via Image Path

**File:** `mlx_server/services/image_processor.py:67-72`

```python
else:
    # Assume local file path
    img = Image.open(image_input)
```

No path validation. Error message leaks path existence info.

**Attack vector:** `/etc/passwd`, `~/.ssh/id_rsa`, any file PIL can open.

**Fix:** Remove local file path support from API-facing code, or validate paths against allowed directories.

#### S-4 [MEDIUM]: Missing Input Validation on User Registration

**File:** `models/dto/auth.py:20-23`

```python
class UserCreate(BaseModel):
    email: str    # No EmailStr, no format validation
    password: str # No min_length, no max_length
```

**Fix:** Use `EmailStr`, add `min_length=8` / `max_length=128` on password.

#### S-5 [MEDIUM]: ReDoS Risk in User-Supplied Regex Patterns

**Files:** `routers/settings.py:316-317, 513-524`

Users can create routing rules with regex patterns that are compiled and matched without protection against catastrophic backtracking.

**Fix:** Use `re2` library (linear-time guarantee) or validate regex complexity before compilation.

#### S-6 [MEDIUM]: Rate Limiting Disabled by Default

**File:** `mlx_server/config.py:168-173`

```python
rate_limit_rpm: int = Field(default=0, ...)  # 0 = disabled
```

Combined with S-1 (no auth), the server is completely open. The management API login endpoint also has no rate limiting (brute-force risk).

**Fix:** Enable reasonable default (e.g., 600 RPM). Add specific login rate limiting.

#### S-7 [MEDIUM]: JWT Token Exposed in Query Parameters

**File:** `dependencies.py:70-71`

SSE/WebSocket endpoints accept JWT as URL query parameter (inherent EventSource limitation). Tokens appear in server logs, browser history, proxy logs.

**Fix:** Document the risk. Issue short-lived tokens specifically for SSE connections.

#### S-8 [MEDIUM]: Exception Details Leaked to Clients in Management API

**Files:** `routers/models.py:78`, `routers/system.py:139`

```python
raise HTTPException(status_code=500, detail=str(e))
```

**Fix:** Return generic messages. Log full exceptions server-side.

#### S-9 [LOW]: WebSocket Audit Log Missing Auth on MLX Server Side

**File:** `mlx_server/api/v1/admin.py:481-517`

`/v1/admin/ws/audit-logs` accepts WebSocket without auth check (the parent proxy validates JWT, but the endpoint is directly accessible).

**Fix:** Add admin token validation before `websocket.accept()`.

#### S-10 [LOW]: Unbounded `download_tasks` Dictionary

**File:** `routers/models.py:61`

Repeated `POST /api/models/download` calls can grow the dict without bound.

**Fix:** Add max-size limit or periodic cleanup.

---

## 5. Performance

### P-1 [CRITICAL]: New Thread Per Inference — No Metal Thread Affinity

**File:** `mlx_server/utils/metal.py:53-54, 103-104`

```python
gen_thread = threading.Thread(target=_worker, daemon=True)
gen_thread.start()
```

Both `run_on_metal_thread` and `stream_from_metal_thread` create a **new thread for every request**. The module docstring states "Metal GPU operations have thread affinity" — but creating throwaway threads defeats this. Each new thread forces MLX/Metal to potentially re-establish GPU context.

**Why this is critical:** This is the inference hot path. Thread creation costs ~50-200us on macOS. More importantly, Metal context re-initialization adds latency. Under concurrent load, rapid thread creation/destruction causes kernel scheduling overhead.

**Fix:** Single persistent Metal worker thread with job queue:

```python
class MetalWorker:
    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            fn, result_queue = self._queue.get()
            try:
                result_queue.put(fn())
            except Exception as e:
                result_queue.put(e)
```

### P-2 [HIGH]: Sync `list_local_models()` Blocks Event Loop

**File:** `services/hf_client.py:517-583`, called from `routers/models.py:93`

```python
all_local = hf_client.list_local_models()  # SYNC: blocks event loop
```

Performs heavy filesystem I/O: iterates all HF cache dirs, `Path.rglob("*")` with `stat()` on every file, reads `config.json` per model. With 20+ models, blocks for seconds.

**Fix:** `await loop.run_in_executor(None, hf_client.list_local_models)`

### P-3 [HIGH]: O(n^2) String Concatenation in Generation Loop

**Files:** `services/inference.py:450`, `adapters/composable.py:584`

```python
response_text += token_text  # Called once per token, O(n^2) worst case
```

For 4096-token completions, worst case is O(n^2) memory copies on the Metal thread.

**Fix:** Use list accumulator + `"".join(parts)`.

### P-4 [MEDIUM]: Sequential Auto-Start Model Loading

**File:** `main.py:227-256`

```python
for profile in auto_start_profiles:
    await pool.model_pool.preload_model(model_id)  # Sequential, 5-30s each
```

With 3 profiles, startup takes 90+ seconds. Server doesn't accept requests during this time.

**Fix:** Use `asyncio.gather()` or accept requests immediately with 503 until ready.

### P-5 [MEDIUM]: httpx Client Created Per-Request

**Files:** `routers/system.py:199,218,253`, `services/hf_api.py:157,250`

```python
async with httpx.AsyncClient() as client:  # New connection pool every time
```

Each construction creates new TCP connection + TLS handshake.

**Fix:** Module-level `httpx.AsyncClient` with connection reuse.

### P-6 [MEDIUM]: HTTP Proxy to Self for Audit Logs

**File:** `routers/system.py:174-271`

Three endpoints make HTTP requests to `http://localhost:{port}/v1/admin/...` to reach code in the **same process**. Full TCP + HTTP overhead to call yourself.

**Fix:** Import and call MLX Server audit service functions directly.

### P-7 [MEDIUM]: Sync Subprocess in Async Launchd Handlers

**File:** `services/launchd.py:89-146`

All `subprocess.run()` calls block the event loop when called from async router handlers.

**Fix:** Use `asyncio.create_subprocess_exec()` or `run_in_executor`.

### P-8 [MEDIUM]: Sync Filesystem Scan in Download Polling

**File:** `services/hf_client.py:425`

```python
current_bytes = self._get_directory_size(download_dir)  # SYNC rglob every 1s
```

**Fix:** `await loop.run_in_executor(None, self._get_directory_size, download_dir)`

### P-9 [MEDIUM]: Double-Copy Audio Data

**File:** `adapters/composable.py:851`

```python
audio_np = np.array(audio.tolist())  # MLX -> Python list -> numpy
```

For 10s audio at 24kHz: 240,000 Python float objects created and discarded.

**Fix:** `np.array(audio, copy=False)` (modern MLX supports direct conversion).

### P-10 [MEDIUM]: Double Tokenization for Embeddings Token Count

**File:** `adapters/composable.py:758-762`

Tokenizes each text a second time purely for counting, after batch encoding already tokenized them.

**Fix:** Count from batch result: `total_tokens = int(encoded["attention_mask"].sum())`

### P-11 [MEDIUM]: N+1 Queries in Model Sync

**File:** `services/model_registry.py:27-48`

```python
for lm in local_models:
    result = await session.execute(select(Model).where(Model.repo_id == lm.model_id))
```

30 models = 30 individual SELECT queries.

**Fix:** Pre-fetch all existing repo_ids in a single query:
```python
existing = {r[0] for r in (await session.execute(select(Model.repo_id))).all()}
```

### P-12 [LOW]: Unbounded download_tasks Dict

**File:** `routers/models.py:61`

No max-size limit. Grows with each download request.

**Fix:** Add periodic cleanup or max-size eviction.

---

## Triage Matrix

Suggested prioritization for v1.2 release:

### Must Fix Before Release

| # | Finding | Effort | Risk if Skipped |
|---|---------|--------|-----------------|
| P-1 | Metal thread affinity (new thread per request) | Medium | Performance regression under any concurrency |
| P-3 | O(n^2) string concat in generation loop | Trivial | Latency scales quadratically with output length |
| P-2 | Sync `list_local_models()` blocks event loop | Trivial | Blocks all requests during model listing |
| S-2 | SSRF via image URLs | Small | Network scanning from server |
| S-3 | Arbitrary local file read via image path | Trivial | File read from any API caller |

### Should Fix Before Release

| # | Finding | Effort | Risk if Skipped |
|---|---------|--------|-----------------|
| S-1 | No auth on inference endpoints | Medium | Any local process can use GPU resources |
| S-4 | Missing input validation on auth | Trivial | Weak passwords, invalid emails |
| S-8 | Exception details leaked to clients | Trivial | Info disclosure |
| P-9 | Double-copy audio data | Trivial | Wasted memory on TTS |
| P-10 | Double tokenization for embeddings | Trivial | Unnecessary CPU on embedding requests |
| CC-2f | `re.match` vs `re.fullmatch` discrepancy | Trivial | Routing rules may behave inconsistently |

### Should Fix Post-Release (v1.2.1)

| # | Finding | Effort | Notes |
|---|---------|--------|-------|
| DP-1 | `ModelPoolManager` god class | Large | Refactor into Loader + Configurator + Pool |
| A-8/A-9 | Adapter owns generation + legacy paths | Large | Extract orchestration, delete legacy |
| A-10 | Batching bypasses pipeline | Large | Requires adapter refactor first |
| P-5 | httpx per-request | Small | Connection pooling |
| P-6 | HTTP proxy to self | Medium | Direct function calls |
| CC-2a-h | DRY violations | Small each | Extract helpers |
| S-5 | ReDoS in routing rules | Small | Use `re2` or validate |
| S-6 | Rate limiting defaults | Small | Enable reasonable defaults |

### Nice to Have (v1.3+)

| # | Finding | Effort | Notes |
|---|---------|--------|-------|
| DP-2 | Singleton style standardization | Medium | Consistency improvement |
| DP-6 | String -> enum for model_type | Medium | Type safety improvement |
| CC-8a | Frontend API client DRY | Medium | Generic helpers |
| P-4 | Parallel auto-start loading | Small | Faster startup |
| P-7 | Async subprocess for launchd | Small | Non-blocking launchd ops |
