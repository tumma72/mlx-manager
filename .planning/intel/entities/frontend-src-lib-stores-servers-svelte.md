---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/stores/servers.svelte.ts
type: hook
updated: 2026-01-21
status: active
---

# servers.svelte.ts

## Purpose

Svelte 5 runes-based store for running server state management. Uses in-place array reconciliation for efficient updates, SvelteMap/SvelteSet for proper reactivity, polling coordinator for centralized refresh, and HMR state preservation for development experience. Tracks running, starting, restarting, and failed server states.

## Exports

- `FailedServer` - Interface for failure details
- `ServerStore` - Class with server management methods
- `serverStore` - Singleton store instance

## Dependencies

- [[frontend-src-lib-api-client]] - API client for server operations
- [[frontend-src-lib-api-types]] - RunningServer type
- [[frontend-src-lib-utils-reconcile]] - Array reconciliation
- [[frontend-src-lib-services-polling-coordinator-svelte]] - Polling coordination
- svelte/reactivity - SvelteMap, SvelteSet

## Used By

TBD

## Notes

Uses custom equality function allowing small drift in uptime/memory to reduce update frequency. HMR state preservation stores state on window object to survive hot reloads. Initial load sets loading=true, but background polls do not to avoid flicker.
