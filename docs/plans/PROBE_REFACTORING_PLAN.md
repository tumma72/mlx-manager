# Probe System Refactoring Plan

**Version:** 1.2
**Date:** 2026-02-26
**Status:** ALL PHASES COMPLETE (6c deferred as future work)
**Scope:** `backend/mlx_manager/services/probe/` + related parser/family detection

---

## Executive Summary

The probe system works but suffers from three classes of problems:

1. **Code duplication** — The step-yielding boilerplate is repeated ~15 times; KV cache estimation is duplicated; config reading patterns are scattered
2. **Parser misidentification** — Models like Qwen3-Coder get mapped to wrong parsers (hermes_json / llama_xml instead of qwen3_coder_xml) due to marker collisions and alphabetical validation ordering
3. **Architectural sprawl** — `coordinator.py` at 823 lines handles orchestration, sweeps, AND persistence; the GenerativeProbe base class is underused since actual generation happens in the coordinator

The plan is organized into 6 phases, ordered by priority and impact. Each phase is independently shippable.

---

## Table of Contents

1. [Phase 1: Step Yielding Boilerplate (HIGH)](#phase-1-step-yielding-boilerplate)
2. [Phase 2: Family-Aware Parser Selection (HIGH/CRITICAL)](#phase-2-family-aware-parser-selection)
3. [Phase 3: Coordinator Decomposition (MEDIUM)](#phase-3-coordinator-decomposition)
4. [Phase 4: Shared Utilities Consolidation (MEDIUM)](#phase-4-shared-utilities-consolidation)
5. [Phase 5: GenerativeProbe Realignment (MEDIUM)](#phase-5-generativeprobe-realignment)
6. [Phase 6: Operational Improvements (LOW-MEDIUM)](#phase-6-operational-improvements)

---

## Phase 1: Step Yielding Boilerplate

**Priority:** HIGH | **Effort:** Small | **Risk:** Low
**Files touched:** All probe files

### Problem

Every probe check repeats the same 8-12 line pattern:

```python
yield ProbeStep(step="check_X", status="running")
try:
    result = do_something()
    yield ProbeStep(step="check_X", status="completed", capability="cap", value=result)
except Exception as e:
    yield ProbeStep(step="check_X", status="failed", error=str(e))
```

This appears ~15+ times across `text_gen.py`, `vision.py`, `embeddings.py`, `audio.py`, and `coordinator.py`. Every new check adds another copy. Inconsistencies in error handling are easy to introduce.

### Solution: Async Context Manager

Add a `probe_step` async context manager to `steps.py`:

```python
@asynccontextmanager
async def probe_step(
    step: str,
    capability: str | None = None,
) -> AsyncGenerator[ProbeStepContext, None]:
    """Wraps the try/yield/catch pattern for probe steps.

    Usage:
        async with probe_step("check_X", "cap") as ctx:
            ctx.value = do_something()
            ctx.details = {"extra": "info"}

    Yields ProbeStep(status="running") on enter.
    Yields ProbeStep(status="completed") on successful exit.
    Yields ProbeStep(status="failed") on exception.
    """
```

The context manager cannot directly `yield` SSE steps (you can't yield from a context manager into an async generator), so the approach needs to be a **collector pattern**: the context manager appends steps to a list, and the caller yields from the list.

Alternative: A **decorator-based** approach wrapping the entire probe method, or a simple **helper function** that takes a callable:

```python
async def run_step(
    step: str,
    fn: Callable[[], Awaitable[T] | T],
    capability: str | None = None,
) -> AsyncGenerator[ProbeStep, None]:
    yield ProbeStep(step=step, status="running")
    try:
        result = await fn() if asyncio.iscoroutinefunction(fn) else fn()
        yield ProbeStep(step=step, status="completed", capability=capability, value=result)
    except Exception as e:
        yield ProbeStep(step=step, status="failed", error=str(e))
```

Usage:
```python
async for s in run_step("check_context", lambda: estimate_max_tokens(model_id, loaded), "practical_max_tokens"):
    yield s
```

### Design Decision Needed

Choose between:
- **Option A: Helper async generator** (`run_step`) — Simple, composable, no magic. Callers still yield explicitly.
- **Option B: Collector context manager** — Cleaner call site but adds a mutable state object. Steps must be yielded after the `async with` block.
- **Option C: Keep as-is** — Accept the boilerplate to preserve explicitness.

**Recommendation:** Option A (helper generator). It's the most Pythonic for async generators and doesn't require a separate collection step.

### Affected Locations

| File | Approximate count | Lines |
|------|-------------------|-------|
| `text_gen.py` | 2 | 60-72 |
| `vision.py` | 4 | 80-139 |
| `embeddings.py` | 4 | 37-104 |
| `audio.py` | 3 | 49-120 |
| `coordinator.py` | 6+ | Throughout probe(), _sweep_generative_capabilities() |

### Validation

- All existing probe tests pass unchanged
- SSE output format remains identical (verify via a simple E2E test or manual probe)
- `to_sse()` serialization unaffected

---

## Phase 2: Family-Aware Parser Selection

**Priority:** HIGH / CRITICAL | **Effort:** Medium | **Risk:** Medium
**Files touched:** `coordinator.py`, `base.py`, `strategy.py`, `configs.py`, `registry.py`

### Problem

Models like Qwen3-Coder-Next get assigned wrong parsers. The root cause is a chain of three interacting issues:

#### Issue 2a: Marker Collision

The `<tool_call>` stream marker is shared by **4 parsers**:
- `hermes_json`
- `glm4_native`
- `glm4_xml`
- `qwen3_coder_xml`

When the probe detects `<tool_call>` in output, `_discover_and_map_tags()` returns all 4 as candidates.

#### Issue 2b: Alphabetical Validation Order

In `_sweep_tools()` (coordinator.py:666-691), matched parsers are validated in `sorted()` order:

```python
for pid in sorted(matched_parser_ids):  # glm4_native, glm4_xml, hermes_json, qwen3_coder_xml
    if TOOL_PARSERS[pid]().validates(last_output, "get_weather"):
        return ("detected", pid, ...)  # First one that validates wins
```

If `glm4_xml` or `hermes_json` happens to validate on Qwen3-Coder output (e.g., because the output is ambiguous or partially matches), it wins before `qwen3_coder_xml` is even tried.

#### Issue 2c: Family Detection Disconnect

Two separate family detection systems exist:
- `utils/model_detection.py` → returns variant IDs like `"qwen3_coder"` (correct for Qwen3-Coder)
- `adapters/registry.py` → returns family keys like `"qwen"` (too broad for Qwen3-Coder)

The probe uses `detect_model_family()` from the adapter registry, which returns `"qwen"` for all Qwen models. The `FAMILY_CONFIGS` dict has a `"nemotron"` entry (which correctly uses `Qwen3CoderXmlParser`) but no `"qwen3_coder"` alias.

### Solution: Family-Prioritized Validation

The fix requires changes at multiple levels:

#### Step 2.1: Unify Family Detection

Consolidate the two `detect_model_family()` functions. The adapter registry version (`adapters/registry.py`) should incorporate the variant-aware logic from `utils/model_detection.py`:

```python
# adapters/registry.py — updated FAMILY_PATTERNS
FAMILY_PATTERNS = {
    # More specific patterns FIRST (order matters for first-match)
    "nemotron": ["nemotron", "qwen3-coder", "qwen3_coder"],  # ← Add Qwen3-Coder here
    "qwen": ["qwen"],  # Base Qwen (after nemotron is checked)
    ...
}
```

Alternatively, add `"qwen3_coder"` as an alias in `FAMILY_CONFIGS`:
```python
FAMILY_CONFIGS["qwen3_coder"] = FAMILY_CONFIGS["nemotron"]
```

#### Step 2.2: Family-Aware Parser Prioritization

When the probe has detected both a model family AND candidate parsers, it should **try the family's declared parser first** before falling back to alphabetical order.

In `coordinator.py`, modify the validation phase of both `_sweep_thinking()` and `_sweep_tools()`:

```python
# Current (alphabetical):
for pid in sorted(matched_parser_ids):
    ...

# Proposed (family-first, then alphabetical):
family_parser_id = self._get_family_parser_id(result.model_family)
ordered_pids = _prioritize_parsers(matched_parser_ids, family_parser_id)
for pid in ordered_pids:
    ...
```

Where `_prioritize_parsers()` puts the family-declared parser first:
```python
def _prioritize_parsers(candidates: set[str], family_parser_id: str | None) -> list[str]:
    """Order candidates: family-declared parser first, then alphabetical."""
    if family_parser_id and family_parser_id in candidates:
        return [family_parser_id] + sorted(candidates - {family_parser_id})
    return sorted(candidates)
```

And `_get_family_parser_id()` reads from `FAMILY_CONFIGS`:
```python
def _get_family_parser_id(self, family: str | None) -> str | None:
    if not family:
        return None
    config = FAMILY_CONFIGS.get(family)
    if config and config.tool_parser_factory:
        return config.tool_parser_factory().parser_id
    return None
```

#### Step 2.3: Stricter `validates()` for Ambiguous Parsers

For parsers sharing the `<tool_call>` marker, their `validates()` methods should be strict enough to reject output from other formats. Audit each parser:

| Parser | Should REJECT | How |
|--------|---------------|-----|
| `hermes_json` | Output with `<function=name>` inside `<tool_call>` | Check for JSON object, not `<function=` |
| `glm4_native` | Output with `<function=name>` or JSON | Check for `<name>` / `<param>` XML structure |
| `glm4_xml` | Output with `<function=name>` | Check for `<name>` + `<arguments>` XML |
| `qwen3_coder_xml` | Output with JSON or `<name>` | Check for `<function=` pattern |

The `extract()` methods already distinguish these formats via their regex patterns, but `validates()` calls `extract()` — so if `extract()` returns empty for wrong format, `validates()` correctly rejects. **Verify this empirically** by running each parser's `validates()` against sample output from each format.

#### Step 2.4: Add Family-Parser Consistency Diagnostic

When probe detects a parser that differs from the family's declared parser, emit a diagnostic:

```python
if detected_parser_id != family_parser_id:
    diagnostics.append(ProbeDiagnostic(
        level=DiagnosticLevel.INFO,
        category=DiagnosticCategory.TOOL_DIALECT,
        message=f"Detected parser '{detected_parser_id}' differs from family default '{family_parser_id}'",
        details={"detected": detected_parser_id, "family_default": family_parser_id},
    ))
```

### Validation

- Probe Qwen3-Coder model → should detect `qwen3_coder_xml` (not hermes_json)
- Probe Nemotron model → should detect `qwen3_coder_xml`
- Probe base Qwen3 model → should detect `hermes_json` (unchanged)
- Probe GLM4 model → should detect `glm4_native` or `glm4_xml` (unchanged)
- All existing probe tests pass
- Add unit test: `test_family_prioritized_parser_selection` with mock outputs for each colliding format

---

## Phase 3: Coordinator Decomposition

**Priority:** MEDIUM | **Effort:** Medium | **Risk:** Low
**Files touched:** `coordinator.py` → split into `coordinator.py` + `sweeps.py`

### Problem

`coordinator.py` is 823 lines handling three distinct concerns:
1. **Lifecycle orchestration** (probe steps 1-8): ~245 lines
2. **Parser sweeps** (_sweep_thinking + _sweep_tools): ~375 lines
3. **Database persistence** (_save_capabilities): ~35 lines

The sweep methods are the bulk of the file and are logically independent from the orchestration. They're also the most complex part and would benefit from focused testing.

### Solution: Extract `sweeps.py`

Create `sweeps.py` containing:
- `sweep_thinking()` — extracted from `ProbingCoordinator._sweep_thinking()`
- `sweep_tools()` — extracted from `ProbingCoordinator._sweep_tools()`
- `sweep_generative_capabilities()` — extracted from `ProbingCoordinator._sweep_generative_capabilities()`

These become module-level async functions that take explicit parameters instead of accessing `self._pool`:

```python
# sweeps.py
async def sweep_thinking(
    model_id: str,
    loaded: LoadedModel,
    strategy: GenerativeProbe,
    template_params: dict[str, Any] | None,
    result: ProbeResult,
    family_config: FamilyConfig | None = None,  # ← For Phase 2 family-aware selection
) -> tuple[bool, str, list[ProbeDiagnostic], list[TagDiscovery]]:
    ...

async def sweep_tools(
    model_id: str,
    loaded: LoadedModel,
    strategy: GenerativeProbe,
    result: ProbeResult,
    family_config: FamilyConfig | None = None,
) -> tuple[str | None, str | None, list[ProbeDiagnostic], list[TagDiscovery]]:
    ...
```

The coordinator becomes a thin orchestrator (~250 lines) that calls into `sweeps.py`.

### Secondary Benefit

With sweeps extracted, they can be independently unit-tested with mock adapters/outputs, without needing to spin up the full coordinator + pool.

### Validation

- All existing tests pass
- No behavior change — pure structural refactor
- Coordinator imports from sweeps module

---

## Phase 4: Shared Utilities Consolidation

**Priority:** MEDIUM | **Effort:** Small | **Risk:** Low
**Files touched:** `base.py`, `text_gen.py`, `vision.py`, `embeddings.py`, `audio.py`

### Problem 4a: Duplicated KV Cache Estimation

Near-identical functions in `text_gen.py:75-82` and `vision.py:207-215`:

```python
def _estimate_practical_max_tokens(model_id, loaded):
    from mlx_manager.mlx_server.utils.kv_cache import estimate_practical_max_tokens
    return estimate_practical_max_tokens(model_id, loaded.size_gb)
```

### Solution 4a

Move to `base.py` as a shared utility:

```python
# base.py
def estimate_context_window(model_id: str, loaded: LoadedModel) -> int | None:
    """Estimate practical max tokens via KV cache calculation."""
    from mlx_manager.mlx_server.utils.kv_cache import estimate_practical_max_tokens
    return estimate_practical_max_tokens(model_id, loaded.size_gb)
```

Both `TextGenProbe` and `VisionProbe` call `estimate_context_window()`.

### Problem 4b: Scattered Config Reading

Multiple probes call `read_model_config(model_id)` and check for specific keys with different fallback chains:
- `vision.py:186-204`: Checks `image_token_id`, `video_token_id`
- `embeddings.py:120-131`: Checks `max_position_embeddings`, `max_seq_length`, `max_sequence_length`
- `audio.py:145-223`: Checks `architectures`, `model_type`, codec indicators

### Solution 4b

Add a thin helper to `base.py`:

```python
def get_model_config_value(model_id: str, *keys: str, default: Any = None) -> Any:
    """Read first matching key from model config.json with fallback."""
    config = read_model_config(model_id)
    if config is None:
        return default
    for key in keys:
        if key in config:
            return config[key]
    return default
```

This eliminates the repeated `config = read_model_config(model_id); if config is None: return None; return config.get("key")` pattern.

### Validation

- All existing tests pass
- No behavior change

---

## Phase 5: GenerativeProbe Realignment

**Priority:** MEDIUM | **Effort:** Medium | **Risk:** Medium
**Files touched:** `base.py`, `text_gen.py`, `vision.py`, `coordinator.py`/`sweeps.py`

### Problem

`GenerativeProbe` (`base.py:287-330`) provides `_generate()` as a shared method for adapter-based generation. `TextGenProbe` and `VisionProbe` inherit it. But the actual generative work (thinking/tool sweeps) happens in `coordinator.py`, not in the strategies. The strategies only do static checks.

This creates a confusing split: TextGenProbe inherits GenerativeProbe but doesn't use `_generate()` in its own `probe()` method — only the coordinator uses it externally via `strategy._generate()`.

### Solution: Two Options

#### Option A: Move Sweep Orchestration Into GenerativeProbe (Preferred)

Make `GenerativeProbe` own the generative sweep steps. The coordinator still orchestrates the lifecycle, but delegates generative sweeps to the strategy:

```python
# base.py
class GenerativeProbe(BaseProbe):
    async def sweep_capabilities(
        self, model_id, loaded, result, family_config
    ) -> AsyncGenerator[ProbeStep, None]:
        """Run thinking + tool sweeps. Default impl for text/vision."""
        async for step in sweep_thinking(...):
            yield step
        async for step in sweep_tools(...):
            yield step
```

The coordinator calls `strategy.sweep_capabilities()` instead of its own `_sweep_generative_capabilities()`. Vision can override to add image-specific generation behavior.

This makes the architecture consistent: strategies own ALL type-specific behavior (static AND generative).

#### Option B: Remove GenerativeProbe Inheritance

Make `TextGenProbe` and `VisionProbe` inherit directly from `BaseProbe`. Move `_generate()` to a standalone function in `base.py` or `sweeps.py`. The coordinator calls it directly.

This is simpler but loses the OOP structure that currently signals "these probes can generate."

**Recommendation:** Option A, which aligns the inheritance with the actual responsibility.

### Validation

- All existing tests pass
- Probe output identical
- Sweep behavior unchanged — just relocated

---

## Phase 6: Operational Improvements

**Priority:** LOW-MEDIUM | **Effort:** Varies | **Risk:** Low

### 6a: Generation Timeouts

**Problem:** Text/vision generation in sweeps has no timeout. A model that generates endlessly blocks the probe indefinitely. Audio probes already have timeouts (`audio.py:226-252`).

**Solution:** Add `asyncio.wait_for()` wrapper in `GenerativeProbe._generate()`:

```python
async def _generate(self, loaded, messages, *, max_tokens=800, timeout: float = 60.0):
    return await asyncio.wait_for(
        self._generate_inner(loaded, messages, max_tokens=max_tokens),
        timeout=timeout,
    )
```

Yield a `ProbeStep(status="failed", error="Generation timed out")` on `asyncio.TimeoutError`.

### 6b: Verbose Flag Utilization

**Problem:** `verbose` is passed through the chain but barely used. It could control:
- Step-level timing (how long each step took)
- Raw output inclusion in step details
- Parser validation details (which parsers were tried, which rejected, why)

**Solution:** Thread verbose through sweep functions. When `verbose=True`, add `details={"raw_output": ..., "parsers_tried": [...], "timing_ms": ...}` to ProbeStep.

### 6c: Incremental Probing

**Problem:** No way to re-probe only tools or only thinking without re-running the entire pipeline. The `--force` flag clears everything.

**Solution (future):** Add optional `scopes` parameter to `probe_model()`:

```python
async def probe_model(model_id, *, verbose=False, scopes: set[str] | None = None):
    # scopes: {"type", "thinking", "tools", "context", "all"}
    # None = all (backward compat)
```

This is lower priority but would significantly speed up iteration when debugging parser selection.

### 6d: Report Availability in API

**Problem:** `generate_support_report()` is CLI-only. The frontend ProbeModal has its own copy logic that's separate from the structured report.

**Solution:** Add an optional `include_report` query parameter to the API endpoint. When set, the final `probe_complete` step includes the markdown report in `details.report`.

### 6e: Family-Parser Audit Command

**Problem:** No easy way to verify parser selection across all cached models.

**Solution:** Add `mlx-manager probe --audit` that runs the parser selection logic (without actual generation) against all cached models and reports:
- Detected family
- FamilyConfig parser
- Stored DB parser (if probed before)
- Mismatch warnings

---

## Implementation Order

```
Phase 1 (Step Boilerplate)     ─── Can start immediately, no deps
    │
Phase 2 (Parser Selection)    ─── Can start immediately, no deps
    │                               (highest business value)
    │
Phase 3 (Coordinator Split)   ─── After Phase 1 (cleaner with helpers)
    │                               After Phase 2 (sweeps contain fix)
    │
Phase 4 (Shared Utils)        ─── After Phase 1, parallel with Phase 3
    │
Phase 5 (GenerativeProbe)     ─── After Phase 3 (depends on sweeps.py)
    │
Phase 6 (Operational)         ─── After all above, can be picked individually
```

Phases 1 and 2 are independent and can be done in parallel. Phase 2 is the most critical for correctness.

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | Very low — mechanical refactor | Verify SSE output format unchanged |
| 2 | Medium — changes parser selection logic | Add unit tests for each colliding format; test with real models (Qwen3-Coder, Nemotron, base Qwen3, GLM4) |
| 3 | Low — structural extraction | No behavior change; import paths updated |
| 4 | Very low — utility consolidation | Existing tests cover all paths |
| 5 | Medium — responsibility relocation | Ensure coordinator+strategy contract is clear |
| 6 | Low — additive features | Each sub-task is independent |

---

## Test Strategy

### Unit Tests (New)

| Test | Phase | Purpose |
|------|-------|---------|
| `test_run_step_helper` | 1 | Verify helper yields correct step sequence |
| `test_family_prioritized_parser_selection` | 2 | Mock outputs for each colliding format; verify correct parser |
| `test_marker_collision_resolution` | 2 | All 4 `<tool_call>` parsers with each format's output |
| `test_qwen3_coder_family_detection` | 2 | Verify "qwen3-coder" → nemotron family |
| `test_sweep_thinking_standalone` | 3 | Extracted sweep with mock adapter |
| `test_sweep_tools_standalone` | 3 | Extracted sweep with mock adapter |
| `test_estimate_context_window_shared` | 4 | Verify shared function works for both text/vision |

### E2E Tests (Existing, verify pass)

- `pytest -m e2e` — all existing probe E2E tests
- Manual probe of Qwen3-Coder-Next (verify qwen3_coder_xml parser)
- Manual probe of GLM-4.7-Flash (verify glm4_native parser unchanged)
- Manual probe of base Qwen3-0.6B (verify hermes_json parser unchanged)

---

## File Inventory (Current → Proposed)

| Current File | Lines | Phase | Change |
|---|---|---|---|
| `coordinator.py` | 823 | 2,3,5 | Extract sweeps, add family-aware logic, thin to ~250 lines |
| `base.py` | 429 | 1,4 | Add `run_step()` helper, `estimate_context_window()`, `get_model_config_value()` |
| `steps.py` | 122 | 1 | Add `ProbeStepContext` if using context manager approach |
| `strategy.py` | 96 | — | No changes |
| `text_gen.py` | 83 | 1,4 | Use `run_step()`, use shared `estimate_context_window()` |
| `vision.py` | 216 | 1,4 | Use `run_step()`, use shared `estimate_context_window()` |
| `embeddings.py` | 164 | 1,4 | Use `run_step()`, use shared `get_model_config_value()` |
| `audio.py` | 318 | 1 | Use `run_step()` |
| `report.py` | 203 | 6d | Optionally expose via API |
| `service.py` | 39 | 6c | Optionally add `scopes` parameter |
| **NEW: `sweeps.py`** | ~400 | 3 | Extracted from coordinator |
| `adapters/configs.py` | — | 2 | Add `"qwen3_coder"` alias |
| `adapters/registry.py` | — | 2 | Update `FAMILY_PATTERNS` ordering |
