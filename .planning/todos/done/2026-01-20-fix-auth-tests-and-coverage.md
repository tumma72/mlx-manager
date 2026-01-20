---
created: 2026-01-20T11:35
title: Fix auth tests and restore coverage
area: testing
files:
  - backend/tests/*
  - frontend/src/lib/api/client.test.ts
  - backend/mlx_manager/routers/auth.py
  - frontend/src/routes/(public)/login/+page.svelte
---

## Problem

Phase 3 (User-Based Authentication) implementation introduced breaking changes that weren't reflected in tests:

1. All backend API endpoints now require authentication (added `get_current_user` dependency)
2. Frontend API client now includes auth headers on all requests
3. Login page flow was fixed (token stored before auth.me())
4. Backend auth router has new last-admin-disable protection

Existing tests are failing because:
- Backend tests don't mock/provide auth tokens
- Frontend API client tests may not expect auth headers
- Coverage has dropped below threshold

Run `make test` from root folder to see current failures.

## Solution

1. **Backend tests**: Add auth fixtures/mocks to provide valid JWT tokens for protected endpoints
2. **Frontend tests**: Update API client tests to expect Authorization headers
3. **New tests needed**:
   - Auth router endpoints (register, login, user management)
   - Last admin protection logic
   - Login page component tests
4. Restore coverage to 95%+ threshold
