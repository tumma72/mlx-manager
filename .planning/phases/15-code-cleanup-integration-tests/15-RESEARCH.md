# Phase 15: Code Cleanup & Integration Tests - Research

**Researched:** 2026-02-04 (Updated)
**Domain:** Code cleanup, database migrations, golden file testing, application logging audit
**Confidence:** HIGH

## Summary

Phase 15 focuses on three distinct cleanup domains: removing dead parser code now replaced by ResponseProcessor, fixing critical infrastructure bugs discovered during Phase 14 UAT, and creating comprehensive integration tests using golden files to validate the ResponseProcessor works correctly with all model families.

**Update 2026-02-04:** Investigation confirms significant progress already made:
1. **Dead code already removed** - `adapters/parsers/` folder does not exist, adapter methods `parse_tool_calls()` and `extract_reasoning()` already removed from protocol
2. **Database migration already in place** - `api_type` and `name` columns added to `cloud_credentials` via `migrate_schema()` in database.py
3. **Golden file infrastructure exists** - `test_response_processor_golden.py` with parametrized tests, golden files for all 6 families
4. **Exception handling already improved** - Qwen adapter catches `(TypeError, ValueError, KeyError, AttributeError)` for enable_thinking fallback

**Primary recommendation:** Focus on expanding streaming golden file coverage, adding thinking golden files for all supporting families, auditing logging levels, and diagnosing LogFire trace delivery issues.

## Standard Stack

The project already uses the core technologies needed for this phase:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >=8.0.0 | Test framework | Already project standard, supports parametrization |
| SQLite | 3.33.0+ | Database | Built-in to Python, existing migration pattern established |
| SQLAlchemy/SQLModel | 0.0.22+ | ORM | Project standard for database access |
| Python logging | stdlib | Logging | Standard library, already integrated |
| logfire | >=3.0.0 | Observability | Already installed, needs debugging |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-golden | 1.0.1 | Golden file testing | NOT NEEDED - manual approach works fine |
| pytest-regressions | 3.0+ | Snapshot testing | NOT NEEDED - manual approach works fine |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual golden files | pytest-golden plugin | Plugin adds dependency; manual approach has zero deps and full control |
| PRAGMA migrations | Alembic migrations | Alembic already installed but PRAGMA pattern is simpler for SQLite |

**Installation:**
```bash
# No new dependencies needed - all tools already installed
cd backend && source .venv/bin/activate
```

## Architecture Patterns

### Current Golden File Structure (Verified 2026-02-04)
```
backend/tests/fixtures/golden/
├── gemma/
│   └── tool_calls.txt           # EXISTS
├── glm4/
│   ├── tool_calls.txt           # EXISTS - GLM4 XML format
│   └── duplicate_tools.txt      # EXISTS - Deduplication test
├── hermes/
│   └── tool_calls.txt           # EXISTS
├── llama/
│   ├── tool_calls.txt           # EXISTS - Llama XML format
│   └── python_tag.txt           # EXISTS - Python tag format
├── minimax/
│   └── tool_calls.txt           # EXISTS
└── qwen/
    ├── tool_calls.txt           # EXISTS
    ├── thinking.txt             # EXISTS
    └── stream/                  # ONLY family with stream/
        ├── thinking_chunks.txt
        └── tool_call_chunks.txt
```

### Missing Golden Files (To Add)
```
backend/tests/fixtures/golden/
├── gemma/
│   └── stream/                  # MISSING - needs tool_call_chunks.txt
├── glm4/
│   ├── thinking.txt             # MISSING - GLM4 supports reasoning
│   └── stream/                  # MISSING
├── hermes/
│   └── stream/                  # MISSING
├── llama/
│   ├── thinking.txt             # MISSING - Llama-thinking variants
│   └── stream/                  # MISSING
└── minimax/
    └── stream/                  # MISSING
```

### Pattern 1: SQLite Column Addition with PRAGMA Check (ALREADY IMPLEMENTED)
**What:** Check if column exists before adding it to avoid errors on existing databases
**When to use:** Adding columns to SQLite tables with existing data
**Example:**
```python
# Source: backend/mlx_manager/database.py (existing, verified 2026-02-04)
migrations: list[tuple[str, str, str, str | None]] = [
    ("server_profiles", "tool_call_parser", "TEXT", None),
    ("server_profiles", "reasoning_parser", "TEXT", None),
    ("server_profiles", "message_converter", "TEXT", None),
    ("server_profiles", "system_prompt", "TEXT", None),
    # CloudCredential columns for provider configuration (Phase 14 bug fix)
    ("cloud_credentials", "api_type", "TEXT", "'openai'"),
    ("cloud_credentials", "name", "TEXT", "''"),
]
```

### Pattern 2: Golden File Test Parametrization (ALREADY IMPLEMENTED)
**What:** Auto-discover and parametrize tests from golden file directory structure
**Example:**
```python
# Source: backend/tests/mlx_server/test_response_processor_golden.py (existing)
def collect_tool_call_files() -> list[tuple[str, Path]]:
    """Collect all tool call golden files for parametrization."""
    test_cases: list[tuple[str, Path]] = []
    for family_dir in sorted(GOLDEN_DIR.iterdir()):
        if not family_dir.is_dir():
            continue
        tool_file = family_dir / "tool_calls.txt"
        if tool_file.exists():
            test_cases.append((family_dir.name, tool_file))
    return test_cases

@pytest.mark.parametrize("family,golden_file", collect_tool_call_files())
def test_tool_calls_extracted(self, family: str, golden_file: Path):
    """Verify tool calls are extracted from model family format."""
```

### Pattern 3: Exception-Specific Fallback for Qwen Thinking (ALREADY IMPLEMENTED)
**What:** Catch specific exceptions when enable_thinking fails
**Example:**
```python
# Source: backend/mlx_manager/mlx_server/models/adapters/qwen.py (existing)
try:
    result: str = cast(
        str,
        actual_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            enable_thinking=True,  # Qwen3 thinking mode
        ),
    )
    return result
except (TypeError, ValueError, KeyError, AttributeError) as e:  # Catches all needed
    logger.debug(f"Tokenizer doesn't support enable_thinking, falling back: {e}")
```

### Pattern 4: Processor Attribute Access for Vision Models
**What:** Handle both Tokenizer and Processor objects consistently
**When to use:** Any code that accesses tokenizer methods
**Example:**
```python
# Source: backend/mlx_manager/mlx_server/models/adapters/base.py (existing)
def get_stop_tokens(self, tokenizer: Any) -> list[int]:
    """Handles both Tokenizer and Processor objects (vision models use Processor)."""
    # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    return [actual_tokenizer.eos_token_id]
```

### Anti-Patterns to Avoid
- **Skipping PRAGMA check:** Always check column existence before ALTER TABLE to support existing databases
- **INFO logging for tokens:** Token-level and per-chunk data should be DEBUG, not INFO
- **Generating golden files in tests:** Golden files should be pre-created and committed, never generated at test time
- **Testing implementation details:** Test behavior (tool calls extracted, markers removed) not internal methods
- **Catching bare Exception:** Use specific exception types for fallback handling

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Golden file discovery | Manual test list | Directory iteration + parametrize | Automatically picks up new files |
| Database schema version tracking | Custom version table | Project's existing PRAGMA pattern | Already works, no migrations history needed |
| Logger hierarchy | Manual logger naming | `logging.getLogger(__name__)` | Automatic module-based naming |
| Streaming buffer accumulation | Custom buffer class | List append + "".join() | Standard pattern, no overhead |

**Key insight:** This phase doesn't need new libraries. Python stdlib (logging, pathlib) + pytest + existing patterns cover all needs.

## Common Pitfalls

### Pitfall 1: SQLite NOT NULL Constraint Without Default
**What goes wrong:** `ALTER TABLE ADD COLUMN col TEXT NOT NULL` fails on non-empty tables
**Why it happens:** SQLite can't add NOT NULL columns without defaults to tables with existing rows
**How to avoid:** Always provide DEFAULT value when adding columns: `ADD COLUMN col TEXT DEFAULT ''`
**Warning signs:** "Cannot add NOT NULL column without default value" error

### Pitfall 2: Deleting Code Without Checking Imports
**What goes wrong:** Delete parsers directory but forget imports in adapters/tests - runtime import errors
**Why it happens:** Search-and-replace misses, testing only happy paths
**How to avoid:**
1. Find all imports first: `grep -r "from.*adapters.parsers" backend/`
2. Remove imports before deleting files
3. Run full test suite after deletion
**Warning signs:** `ModuleNotFoundError: No module named 'mlx_manager.mlx_server.models.adapters.parsers'`

### Pitfall 3: INFO Logging in Hot Paths
**What goes wrong:** `logger.info()` in per-token streaming loop floods logs, degrades performance
**Why it happens:** Debug code left at INFO level, not audited before commit
**How to avoid:**
- Audit all `logger.info()` calls in streaming/inference code
- Move token-level logs to DEBUG
- Reserve INFO for significant events (model loaded, inference complete)
**Warning signs:** Log file grows MB/second during streaming, console unreadable

### Pitfall 4: LogFire Traces Not Reaching Server
**What goes wrong:** Traces don't appear in LogFire dashboard despite configuration
**Why it happens:** Multiple potential causes - token issues, network, timing, flush
**How to avoid:**
1. Verify LOGFIRE_TOKEN environment variable is set correctly
2. Check network connectivity to logfire-eu.pydantic.dev
3. Use `logfire.force_flush()` to ensure pending spans are sent
4. Verify `logfire.configure()` is called before instrumented imports
5. Enable console output to verify local trace creation
**Warning signs:** No errors in logs but no traces in dashboard

### Pitfall 5: Vision Model Processor Attribute Access
**What goes wrong:** AttributeError when accessing tokenizer methods on vision models
**Why it happens:** Vision models use Processor that wraps tokenizer with different attribute names
**How to avoid:** Use `getattr(processor, "tokenizer", processor)` pattern consistently
**Warning signs:** "processor does not have 'chat_template' or 'tokenizer' attribute"

## Code Examples

Verified patterns from existing codebase:

### Golden File Content Examples

**Qwen Tool Call (golden/qwen/tool_calls.txt):**
```
I'll help you search for that information.
<tool_call>{"name": "search", "arguments": {"query": "weather in London"}}</tool_call>
Let me check that for you.
```

**Llama Tool Call (golden/llama/tool_calls.txt):**
```
I can help you with that calculation.
<function=calculate>{"expression": "2 + 2"}</function>
The answer is ready.
```

**GLM4 Tool Call (golden/glm4/tool_calls.txt):**
```
I'll look that up.
<tool_call><name>lookup</name><arguments>{"key": "value"}</arguments></tool_call>
Here's what I found.
```

**Streaming Chunks (golden/qwen/stream/tool_call_chunks.txt):**
```
I'll search for
 that. <tool_call>{"name":
 "search", "arguments": {"query":
 "test"}}</tool_call> Done
 searching.
```

### Logging Level Audit Pattern
```python
# Source: Python logging best practices
import logging

logger = logging.getLogger(__name__)

# Streaming/inference hot path - use DEBUG
def generate_tokens():
    for token in tokens:
        logger.debug(f"Generated token: {repr(token)}")  # DEBUG: per-token detail
        yield token

# Application events - use INFO
async def load_model(model_id: str):
    logger.info(f"Loading model: {model_id}")  # INFO: significant event
    model = await loader.load(model_id)
    logger.info(f"Model loaded: {model_id}")  # INFO: completion event
    return model

# Errors and warnings
def parse_tool_call(text: str):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid tool call JSON: {e}")  # WARNING: handled error
        return None
```

### LogFire Configuration (Existing)
```python
# Source: backend/mlx_manager/observability/logfire_config.py
logfire.configure(
    service_name=service_name,
    service_version=service_version,
    send_to_logfire="if-token-present",  # Offline mode without token
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Parser classes per family | ResponseProcessor single-pass | Phase 14 (2026-02) | Dead parser code already removed |
| Separate adapter methods | Unified ResponseProcessor | Phase 14 (2026-02) | parse_tool_calls/extract_reasoning removed |
| Manual test fixtures | Golden file organization | Phase 15 (2026-02-02) | Infrastructure in place |
| Missing db columns | migrate_schema() | Phase 14+ | api_type/name columns added |

**Deprecated/outdated (Already Removed):**
- `adapters/parsers/` directory: Does not exist (confirmed 2026-02-04)
- `parse_tool_calls()` adapter methods: Not in protocol or any adapter
- `extract_reasoning()` adapter methods: Not in protocol or any adapter

## Open Questions

Things that couldn't be fully resolved:

1. **LogFire trace collection issue**
   - What we know: User reports traces not reaching logfire-eu.pydantic.dev despite hours of operation
   - What's unclear: Is this a network issue, configuration issue, or LogFire service issue?
   - Recommendation:
     - Verify LOGFIRE_TOKEN environment variable is set and valid
     - Add `logfire.force_flush()` calls at strategic points
     - Enable console logging to verify traces are being created locally
     - Test with `send_to_logfire='always'` temporarily to force sending
     - Check network connectivity to LogFire EU endpoint

2. **Vision model processor attribute access**
   - What we know: UAT reports "processor does not have 'chat_template' or 'tokenizer' attribute" for Gemma vision
   - What's unclear: Exact code path causing the error (vision.py line 71 uses `loaded.tokenizer` which is actually a Processor)
   - Recommendation: Audit all processor attribute access in vision.py, ensure using `getattr(processor, 'tokenizer', processor)` pattern consistently

3. **Streaming golden file coverage**
   - What we know: Only Qwen has stream/ subdirectory with golden files
   - What's unclear: Best way to generate streaming chunk sequences for other families
   - Recommendation: Manually create representative chunk sequences for each family, ensure chunk boundaries split patterns

## Sources

### Primary (HIGH confidence)
- Existing codebase (verified 2026-02-04):
  - `backend/mlx_manager/database.py` - migrate_schema() pattern
  - `backend/mlx_manager/mlx_server/models/adapters/` - no parsers/, no parse_tool_calls
  - `backend/tests/mlx_server/test_response_processor_golden.py` - parametrized tests
  - `backend/tests/fixtures/golden/` - existing golden files
- [Logfire API Reference](https://logfire.pydantic.dev/docs/reference/api/logfire/)
- [Logfire Troubleshooting](https://logfire.pydantic.dev/docs/reference/self-hosted/troubleshooting/)

### Secondary (MEDIUM confidence)
- [pytest-golden PyPI](https://pypi.org/project/pytest-golden/)
- [SQLite ALTER TABLE Documentation](https://www.sqlite.org/lang_altertable.html)
- [Python Logging Best Practices](https://betterstack.com/community/guides/logging/python/python-logging-best-practices/)

### Tertiary (LOW confidence)
- Web searches for LogFire troubleshooting patterns

## Metadata

**Confidence breakdown:**
- Dead code removal: HIGH - Verified code already removed
- Database migration: HIGH - Verified columns already added
- Golden file testing: HIGH - Infrastructure exists, just needs expansion
- Exception handling: HIGH - Verified implementation correct
- LogFire debugging: MEDIUM - Configuration correct, root cause unclear
- Vision fix: MEDIUM - Issue identified, solution clear but needs verification

**Research date:** 2026-02-04 (Updated from 2026-02-02)
**Valid until:** 2026-04-04 (60 days - stable domain, tools mature)

## Implementation Notes for Planning

### What's Already Done (Verified 2026-02-04)
1. `adapters/parsers/` folder - Does not exist (already deleted)
2. `parse_tool_calls()` and `extract_reasoning()` - Not in adapter protocol or implementations
3. `api_type` and `name` columns - Already in migrate_schema() (database.py lines 43-45)
4. Qwen exception handling - Already catches all needed types including AttributeError
5. Golden file infrastructure - test_response_processor_golden.py exists with parametrized tests
6. Golden files for all 6 families - tool_calls.txt exists for all

### Remaining Work
1. **Add streaming golden files** for families other than Qwen (stream/ subdirectory)
2. **Add thinking golden files** for Llama and GLM4 (they support reasoning mode)
3. **Audit logging levels** throughout the application
4. **Diagnose LogFire** trace delivery issue
5. **Fix vision processor** attribute access in vision.py
6. **Verify no dead code references** remain in tests or imports
