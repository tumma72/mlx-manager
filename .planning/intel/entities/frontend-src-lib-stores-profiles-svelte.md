---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/stores/profiles.svelte.ts
type: hook
updated: 2026-01-21
status: active
---

# profiles.svelte.ts

## Purpose

Svelte 5 runes-based store for server profile state management. Uses in-place array reconciliation for efficient updates and polling coordinator for centralized refresh. Profiles change less frequently than servers so uses a longer 10-second polling interval.

## Exports

- `ProfileStore` - Class with profile CRUD methods
- `profileStore` - Singleton store instance

## Dependencies

- [[frontend-src-lib-api-client]] - API client for profile operations
- [[frontend-src-lib-api-types]] - ServerProfile types
- [[frontend-src-lib-utils-reconcile]] - Array reconciliation
- [[frontend-src-lib-services-polling-coordinator-svelte]] - Polling coordination

## Used By

TBD

## Notes

Uses custom equality function comparing all profile fields. Only shows loading state on initial load, not background polls, to avoid UI flicker.
