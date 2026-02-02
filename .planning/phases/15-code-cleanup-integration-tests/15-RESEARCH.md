# Phase 15: Code Cleanup & Integration Tests - Research

**Researched:** 2026-02-02
**Domain:** Code cleanup, database migrations, golden file testing, application logging audit
**Confidence:** HIGH

## Summary

Phase 15 focuses on three distinct cleanup domains: removing dead parser code now replaced by ResponseProcessor, fixing critical infrastructure bugs discovered during Phase 14 UAT, and creating comprehensive integration tests using golden files to validate the ResponseProcessor works correctly with all model families.

The research confirms that:
1. **Dead code removal is straightforward** - parsers directory exists and is imported by adapters, tests reference parser classes
2. **SQLite migrations follow existing pattern** - project already uses PRAGMA-based migration approach in database.py
3. **Golden file testing is well-established** - pytest ecosystem has mature plugins (pytest-golden, pytest-regressions) with parametrization support
4. **Python logging audit has clear best practices** - module-level loggers with __name__, appropriate levels (DEBUG for token-level, INFO for events)

**Primary recommendation:** Use project's existing migration pattern (PRAGMA table_info + ALTER TABLE), organize golden files by model family, and audit logging levels throughout the application for consistency with Python best practices.

## Standard Stack

The project already uses the core technologies needed for this phase:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >=8.0.0 | Test framework | Already project standard, supports parametrization |
| SQLite | 3.33.0+ | Database | Built-in to Python, existing migration pattern established |
| SQLAlchemy/SQLModel | 0.0.22+ | ORM | Project standard for database access |
| Python logging | stdlib | Logging | Standard library, already integrated |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-golden | 1.0.1 | Golden file testing | Optional - manual approach works fine |
| pytest-regressions | 3.0+ | Snapshot testing | Alternative to pytest-golden |
| Pydantic LogFire | 3.0.0+ | Observability | Already installed, needs debugging |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual golden files | pytest-golden plugin | Plugin adds dependency; manual approach has zero deps and full control |
| PRAGMA migrations | Alembic migrations | Alembic already installed but not configured; PRAGMA pattern works and is established |

**Installation:**
```bash
# No new dependencies needed - all tools already installed
cd backend && source .venv/bin/activate
```

## Architecture Patterns

### Recommended Project Structure
```
backend/
├── tests/
│   ├── fixtures/
│   │   └── golden/
│   │       ├── qwen/
│   │       │   ├── tool_calls.txt
│   │       │   ├── thinking.txt
│   │       │   └── stream/
│   │       │       ├── thinking_chunks.txt
│   │       │       └── tool_call_chunks.txt
│   │       ├── llama/
│   │       ├── glm4/
│   │       ├── hermes/
│   │       ├── minimax/
│   │       └── gemma/
│   └── mlx_server/
│       └── test_response_processor_integration.py
└── mlx_manager/
    ├── database.py              # Has migrate_schema() function
    └── mlx_server/
        └── models/adapters/
            └── parsers/         # DELETE THIS DIRECTORY
```

### Pattern 1: SQLite Column Addition with PRAGMA Check
**What:** Check if column exists before adding it to avoid errors on existing databases
**When to use:** Adding columns to SQLite tables with existing data
**Example:**
```python
# Source: SQLite official docs + project's existing database.py pattern
async def migrate_schema() -> None:
    """Add missing columns to existing tables."""
    migrations: list[tuple[str, str, str, str | None]] = [
        ("cloud_credentials", "api_type", "TEXT", "'openai'"),  # Default for backcompat
        ("cloud_credentials", "name", "TEXT", "''"),
    ]

    async with engine.begin() as conn:
        for table, column, col_type, default in migrations:
            # Check if column exists using PRAGMA
            result = await conn.execute(text(f"PRAGMA table_info({table})"))
            columns = [row[1] for row in result.fetchall()]

            if column not in columns:
                # Add the column with default value
                default_clause = f" DEFAULT {default}" if default is not None else ""
                sql = f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}"
                logger.info(f"Migrating database: {sql}")
                await conn.execute(text(sql))
```

### Pattern 2: Golden File Testing with Parametrization
**What:** Organize golden files by model family, use pytest.mark.parametrize to test all families
**When to use:** Integration testing with known-good model outputs
**Example:**
```python
# Source: pytest documentation + pytest-golden patterns
import pytest
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"

@pytest.mark.parametrize("family,format_type", [
    ("qwen", "tool_calls"),
    ("qwen", "thinking"),
    ("llama", "tool_calls"),
    ("glm4", "tool_calls"),
    ("hermes", "tool_calls"),
    ("minimax", "tool_calls"),
    ("gemma", "tool_calls"),
])
def test_response_processor_golden(family: str, format_type: str):
    """Test ResponseProcessor with golden files."""
    golden_file = GOLDEN_DIR / family / f"{format_type}.txt"
    golden_text = golden_file.read_text()

    processor = get_response_processor()
    result = processor.process(golden_text)

    # Assertions based on format_type
    if format_type == "tool_calls":
        assert len(result.tool_calls) > 0
        assert "<tool_call>" not in result.content
    elif format_type == "thinking":
        assert result.reasoning is not None
        assert "<think>" not in result.content
```

### Pattern 3: Streaming Golden Files with Chunk Sequences
**What:** Store streaming output as sequences of chunks to test StreamingProcessor pattern filtering
**When to use:** Validating streaming pattern detection and buffering
**Example:**
```python
# Source: Project pattern from StreamingProcessor
# Golden file format: One chunk per line
# backend/tests/fixtures/golden/qwen/stream/thinking_chunks.txt:
# Let me <think>analyze
# this problem</think> and
# provide an answer

def test_streaming_processor_golden():
    """Test StreamingProcessor filters patterns across chunks."""
    chunks_file = GOLDEN_DIR / "qwen" / "stream" / "thinking_chunks.txt"
    chunks = chunks_file.read_text().splitlines()

    processor = StreamingProcessor()
    accumulated = []

    for chunk in chunks:
        filtered = processor.feed(chunk)
        if filtered:
            accumulated.append(filtered)

    final = processor.finalize()
    if final:
        accumulated.append(final)

    full_output = "".join(accumulated)
    assert "<think>" not in full_output
    assert "</think>" not in full_output
    assert "analyze this problem" in processor.get_reasoning()
```

### Pattern 4: Exception-Specific Fallback for Qwen Thinking
**What:** Catch specific exceptions (TypeError, ValueError, KeyError) when enable_thinking fails
**When to use:** Gradual degradation when tokenizer doesn't support optional parameters
**Example:**
```python
# Source: Python best practices + project context
try:
    result = actual_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        enable_thinking=True,  # Qwen3 thinking mode
    )
except (TypeError, ValueError, KeyError) as e:
    # Older tokenizers don't support enable_thinking parameter
    logger.debug(f"Tokenizer doesn't support enable_thinking, falling back: {e}")
    result = actual_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
```

### Pattern 5: Module-Level Logger with Appropriate Levels
**What:** Use `logger = logging.getLogger(__name__)` at module level, DEBUG for verbose/token data, INFO for events
**When to use:** All Python modules in the application
**Example:**
```python
# Source: Python logging best practices
import logging

logger = logging.getLogger(__name__)

# Good: DEBUG for token-level details
logger.debug(f"Token: {repr(token)}")
logger.debug(f"First content starts with: {repr(preview)}")

# Good: INFO for significant events
logger.info("Applied Qwen3 chat template with enable_thinking=True")
logger.info(f"Migrating database: {sql}")

# Bad: INFO for per-token logs (floods output)
# logger.info(f"First content starts with: {repr(preview)}")  # This is currently in chat.py
```

### Anti-Patterns to Avoid
- **Skipping PRAGMA check:** Always check column existence before ALTER TABLE to support existing databases
- **INFO logging for tokens:** Token-level and per-chunk data should be DEBUG, not INFO
- **Generating golden files in tests:** Golden files should be pre-created and committed, never generated at test time
- **Testing implementation details:** Test behavior (tool calls extracted, markers removed) not internal methods

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Golden file management | Custom file loading/comparison logic | pytest parametrize + Path.read_text() | Parametrize creates one test per file, built-in stdlib tools sufficient |
| Database schema version tracking | Custom version table | Project's existing PRAGMA pattern | Already works, no migrations history needed for SQLite |
| Logger hierarchy | Manual logger naming | `logging.getLogger(__name__)` | Automatic module-based naming, standard practice |
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

### Pitfall 4: Golden Files Without Model Context
**What goes wrong:** File named `tool_call.txt` but unclear which model family or format it represents
**Why it happens:** Flat file structure without organization
**How to avoid:** Organize by family first: `golden/qwen/tool_calls.txt`, `golden/llama/tool_calls.txt`
**Warning signs:** Test failures don't indicate which model family failed

### Pitfall 5: Testing Live Model Generation Instead of Golden Files
**What goes wrong:** Integration tests call actual model inference, slow and flaky
**Why it happens:** Misunderstanding "integration test" as "test with real model"
**How to avoid:** Integration tests validate ResponseProcessor integrates with all model family formats using pre-recorded golden outputs, not live generation
**Warning signs:** Tests require model downloads, take minutes to run, fail on different hardware

### Pitfall 6: LogFire Configuration Timing
**What goes wrong:** LogFire instrumentation doesn't capture traces
**Why it happens:** `logfire.configure()` called AFTER importing instrumented libraries (FastAPI, httpx)
**How to avoid:** Call `logfire.configure()` BEFORE any instrumented imports (project already does this in main.py)
**Warning signs:** Application runs but no traces appear in LogFire dashboard

## Code Examples

Verified patterns from official sources and project code:

### Database Migration Pattern (Project Standard)
```python
# Source: backend/mlx_manager/database.py (existing pattern)
async def migrate_schema() -> None:
    """Add missing columns to existing tables.

    SQLite doesn't support adding columns in CREATE TABLE IF NOT EXISTS,
    so we need to manually add new columns to existing databases.
    """
    migrations: list[tuple[str, str, str, str | None]] = [
        ("cloud_credentials", "api_type", "TEXT", "'openai'"),
        ("cloud_credentials", "name", "TEXT", "''"),
    ]

    async with engine.begin() as conn:
        for table, column, col_type, default in migrations:
            # Check if column exists
            result = await conn.execute(text(f"PRAGMA table_info({table})"))
            columns = [row[1] for row in result.fetchall()]

            if column not in columns:
                # Add the column
                default_clause = f" DEFAULT {default}" if default is not None else ""
                sql = f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}"
                logger.info(f"Migrating database: {sql}")
                await conn.execute(text(sql))
```

### Golden File Organization
```
# Source: pytest-golden patterns + decision from CONTEXT.md
backend/tests/fixtures/golden/
├── qwen/
│   ├── tool_calls.txt          # Complete response with Hermes-style tool calls
│   ├── thinking.txt            # Complete response with <think> tags
│   └── stream/
│       ├── thinking_chunks.txt # Chunk sequence: think tags split across chunks
│       └── tool_call_chunks.txt
├── llama/
│   ├── tool_calls.txt          # Llama XML style: <function=name>{}</function>
│   └── stream/
│       └── tool_call_chunks.txt
├── glm4/
│   ├── tool_calls.txt          # GLM4 XML style: <tool_call><name>...</name></tool_call>
│   ├── duplicate_tools.txt     # Tests GLM4 deduplication
│   └── stream/
├── hermes/
│   └── tool_calls.txt          # Hermes style: <tool_call>{}</tool_call>
├── minimax/
│   └── tool_calls.txt
└── gemma/
    └── tool_calls.txt
```

### Parametrized Integration Test
```python
# Source: pytest documentation + project test patterns
import pytest
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"

# Get all golden files dynamically
def collect_golden_files():
    """Collect all golden files for parametrization."""
    test_cases = []
    for family_dir in GOLDEN_DIR.iterdir():
        if not family_dir.is_dir():
            continue

        # Complete response tests
        for golden_file in family_dir.glob("*.txt"):
            test_cases.append((family_dir.name, golden_file.stem, golden_file))

        # Streaming tests
        stream_dir = family_dir / "stream"
        if stream_dir.exists():
            for golden_file in stream_dir.glob("*.txt"):
                test_cases.append((family_dir.name, f"stream/{golden_file.stem}", golden_file))

    return test_cases

@pytest.mark.parametrize("family,format_type,golden_file", collect_golden_files())
def test_response_processor_golden(family: str, format_type: str, golden_file: Path):
    """Test ResponseProcessor with all golden files."""
    golden_text = golden_file.read_text()

    if format_type.startswith("stream/"):
        # Test streaming processor
        processor = StreamingProcessor()
        chunks = golden_text.splitlines()
        accumulated = []

        for chunk in chunks:
            filtered = processor.feed(chunk)
            if filtered:
                accumulated.append(filtered)

        final = processor.finalize()
        if final:
            accumulated.append(final)

        full_output = "".join(accumulated)

        # Verify patterns filtered
        assert "<think>" not in full_output
        assert "<tool_call>" not in full_output
        assert "<function=" not in full_output
    else:
        # Test complete response processor
        processor = get_response_processor()
        result = processor.process(golden_text)

        # Verify markers removed from content
        assert "<tool_call>" not in result.content
        assert "</tool_call>" not in result.content
        assert "<function=" not in result.content
        assert "<think>" not in result.content
        assert "</think>" not in result.content
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

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Parser classes per family | ResponseProcessor single-pass | Phase 14 (2026-02) | Dead parser code needs removal |
| Manual test fixtures | Golden file organization | Phase 15 research | Standard pytest pattern established |
| Alembic migrations | PRAGMA-based migrations | Project inception | Lightweight for SQLite |
| pytest-golden plugin | Manual parametrization | Phase 15 decision | No extra dependencies needed |

**Deprecated/outdated:**
- `adapters/parsers/` directory: Replaced by ResponseProcessor, delete entire directory
- `parse_tool_calls()` adapter methods: ResponseProcessor handles all parsing, remove from adapters
- `extract_reasoning()` adapter methods: ResponseProcessor handles extraction, remove from adapters
- Alembic auto-generation: Not configured/used in project, PRAGMA pattern is standard

## Open Questions

Things that couldn't be fully resolved:

1. **LogFire trace collection issue**
   - What we know: User reports traces not reaching logfire-eu.pydantic.dev despite hours of operation
   - What's unclear: Is this a network issue, configuration issue, or LogFire service issue?
   - Recommendation:
     - Verify `logfire.configure()` called before instrumented imports (already done in main.py)
     - Check LogFire token validity
     - Test with `send_to_logfire='always'` temporarily to force sending
     - Review LogFire troubleshooting docs for self-hosted/connectivity issues
     - Consider creating minimal reproduction case

2. **Vision model processor attribute access**
   - What we know: UAT reports "processor does not have 'chat_template' or 'tokenizer' attribute" for Gemma vision
   - What's unclear: Exact code path causing the error (vision.py or adapters?)
   - Recommendation: Audit all processor attribute access in vision.py, ensure using `getattr(processor, 'tokenizer', processor)` pattern consistently

3. **Golden file generation process**
   - What we know: Need golden files for 6+ model families across multiple formats
   - What's unclear: Best way to generate initial golden files (manual creation vs temporary script)
   - Recommendation: Create temporary script to generate golden files from actual model outputs once, commit files, delete script

## Sources

### Primary (HIGH confidence)
- [SQLite ALTER TABLE Official Documentation](https://www.sqlite.org/lang_altertable.html)
- [Python Logging Best Practices Complete Guide 2026](https://www.carmatec.com/blog/python-logging-best-practices-complete-guide/)
- [10 Best Practices for Logging in Python](https://betterstack.com/community/guides/logging/python/python-logging-best-practices/)
- [pytest How to parametrize fixtures and test functions](https://docs.pytest.org/en/stable/how-to/parametrize.html)
- Project's existing `database.py` migrate_schema() pattern
- Project's existing test structure in `backend/tests/`

### Secondary (MEDIUM confidence)
- [pytest-golden PyPI](https://pypi.org/project/pytest-golden/)
- [GitHub - oprypin/pytest-golden](https://github.com/oprypin/pytest-golden)
- [Pytest Regressions Data: Golden File Updates 2025](https://johal.in/pytest-regressions-data-golden-file-updates-2025/)
- [SQLite: How to Check If a Column Exists and Add It If Missing](https://www.tutorialpedia.org/blog/check-if-a-column-exists-in-sqlite/)
- [Pydantic AI Debugging & Monitoring with Pydantic Logfire](https://ai.pydantic.dev/logfire/)
- [Logfire Troubleshooting Documentation](https://logfire.pydantic.dev/docs/reference/self-hosted/troubleshooting/)

### Tertiary (LOW confidence)
- [10 Ways to Test ML Code: Fixtures, Seeds, Golden Files](https://medium.com/@connect.hashblock/10-ways-to-test-ml-code-fixtures-seeds-golden-files-811310517cae)
- [Introduction to golden testing](https://ro-che.info/articles/2017-12-04-golden-tests)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already installed and in use
- Architecture: HIGH - Patterns verified in existing codebase and official docs
- Pitfalls: HIGH - Based on SQLite constraints, Python logging standards, pytest best practices
- Golden file testing: HIGH - Well-established pattern with multiple proven approaches
- Database migrations: HIGH - Existing pattern in database.py, SQLite official docs
- LogFire debugging: MEDIUM - Official docs exist but specific issue unclear

**Research date:** 2026-02-02
**Valid until:** 2026-04-02 (60 days - stable domain, tools mature)

**Additional notes:**
- No parsers directory currently exists despite imports referencing it - code may already be partially deleted
- Golden files directory doesn't exist yet - will be created during planning
- Project already has strong test coverage (67%) and quality gates in place
- This phase is primarily cleanup/verification rather than new feature development
