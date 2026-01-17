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

1. **Audit skipped tests:**
   - `TestFuzzyMatcherDifflib` (32 tests): These test an alternative implementation kept for reference. Options:
     - Remove entirely (Difflib not used in production)
     - Convert to integration tests that run in CI only
     - Keep as-is if skip is justified documentation

   - `test_search_returns_results`: Network-dependent test. Options:
     - Mock the network call
     - Move to integration test suite
     - Keep skip with clear justification

2. **Restore coverage to 95%+:**
   - Review uncovered lines in coverage report (88% current)
   - Add tests for:
     - `mlx_manager/main.py` (65% → target areas: lines 63-70, 79-95, 100-126)
     - `mlx_manager/routers/models.py` (59% → target areas: lines 69-98, 135-143)
     - `mlx_manager/services/hf_api.py` (64% → target areas: lines 111-155)

3. **Update Makefile:**
   - Add combined coverage report for backend + frontend
   - Fail CI if coverage drops below threshold
