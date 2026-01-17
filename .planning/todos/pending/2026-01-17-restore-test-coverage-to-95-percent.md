---
created: 2026-01-17T15:15
title: Restore test coverage to 95%+ and audit skipped tests
area: testing
files:
  - backend/tests/test_fuzzy_matcher.py
  - backend/tests/test_hf_api.py
---

## Problem

Test coverage has dropped from target 95%+ to 88% on the backend. This is unacceptable as the README.md displays an auto-generated coverage badge with a 95%+ goal.

Additionally, there are 33 skipped tests that need to be audited:
- 32 in `TestFuzzyMatcherDifflib` — marked skip because Rapidfuzz was chosen over Difflib
- 1 in `test_hf_api.py` — `test_search_returns_results` skipped due to network requirement

The concern is that skipped tests may be hiding broken functionality. Tests should either:
1. Run and pass
2. Be updated to pass
3. Be removed if no longer relevant

Skipping should not be used to hide broken tests.

## Solution

### Completed ✓

1. **Skipped tests audit** — DONE
   - Removed `TestFuzzyMatcherDifflib` (32 tests) - tested non-production code
   - Converted `test_search_returns_results` to 7 mocked tests covering all error paths
   - Result: **0 skipped tests** (was 33)

2. **hf_api.py coverage** — DONE
   - Added tests for: success, fallback, timeout, HTTP error, request error, missing fields
   - Result: **99% coverage** (was 61%)

3. **Existing download test** — DONE
   - Added test for duplicate download returning existing task_id
   - Covers lines 69-79 in routers/models.py

### Remaining (89% → 95%)

To reach 95%, need to cover ~95 more lines. Main gaps:

| File | Coverage | Uncovered | Notes |
|------|----------|-----------|-------|
| main.py | 65% | 40 lines | Lifespan, static file serving |
| routers/models.py | 59% | 52 lines | SSE streaming, download progress |
| server_manager.py | 88% | 24 lines | Process management |
| parser_options.py | 71% | 10 lines | Exception handlers |
| database.py | 85% | 11 lines | Migration code |
| system.py | 85% | 12 lines | Launchd operations |

**Recommended approach:**
1. Focus on models.py SSE tests (biggest impact)
2. Add parser_options exception tests (quick win)
3. Consider 92-93% as realistic near-term target

### Deferred

- Combined coverage report in make test output
- Coverage threshold enforcement in CI
