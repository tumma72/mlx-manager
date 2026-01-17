---
created: 2026-01-17T15:15
title: Restore test coverage to 95%+ and audit skipped tests
area: testing
files:
  - backend/tests/test_fuzzy_matcher.py
  - backend/tests/test_hf_api.py
---

## Problem — RESOLVED ✓

Test coverage had dropped from target 95%+ to 88% on the backend. Additionally, there were 33 skipped tests that needed auditing.

## Solution — COMPLETED ✓

### 1. Skipped Tests Audit
- Removed `TestFuzzyMatcherDifflib` (32 tests) - tested non-production code
- Converted `test_search_returns_results` to 7 mocked tests covering all error paths
- Result: **0 skipped tests** (was 33)

### 2. Coverage Improvements

| File | Before | After | Notes |
|------|--------|-------|-------|
| parser_options.py | 71% | 100% | Added ImportError/Exception tests |
| main.py | 65% | 91% | Added download task management tests |
| routers/models.py | 59% | 100% | Added SSE, active downloads, parsers tests |
| database.py | 85% | 94% | Added migration and recovery tests |

### 3. Coverage Thresholds Enforced
- Backend: `fail_under = 95` in pyproject.toml
- Frontend: 95% thresholds for statements, branches, functions, lines in vitest.config.ts
- Combined coverage report added to `make test` output

### Final Results
- **Backend: 95.24%** (415 tests passing)
- **Frontend: 100% lines, 95.39% branches** (132 tests passing)
- **Average: 97.50%**
