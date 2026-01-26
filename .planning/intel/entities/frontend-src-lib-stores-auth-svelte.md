---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/stores/auth.svelte.ts
type: hook
updated: 2026-01-21
status: active
---

# auth.svelte.ts

## Purpose

Svelte 5 runes-based store for authentication state management. Manages JWT token and user state with localStorage persistence. Used by the API client for auth header injection and throughout the UI for conditional rendering based on auth status.

## Exports

- `AuthStore` - Class with token, user, loading state and methods
- `authStore` - Singleton store instance

## Dependencies

- [[frontend-src-lib-api-types]] - User type definition

## Used By

TBD

## Notes

Auto-initializes on client-side via module-level code. Provides isAuthenticated and isAdmin derived getters. clearAuth() is called by API client on 401 responses.
