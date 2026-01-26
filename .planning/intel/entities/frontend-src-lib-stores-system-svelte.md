---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/stores/system.svelte.ts
type: hook
updated: 2026-01-21
status: active
---

# system.svelte.ts

## Purpose

Svelte 5 runes-based store for system information state. Manages memory stats (polled every 30 seconds) and system info (fetched on demand). Uses shallow comparison before updates to prevent unnecessary re-renders when values haven't changed.

## Exports

- `SystemStore` - Class with memory and info management
- `systemStore` - Singleton store instance

## Dependencies

- [[frontend-src-lib-api-client]] - API client for system endpoints
- [[frontend-src-lib-api-types]] - SystemMemory, SystemInfo types
- [[frontend-src-lib-services-polling-coordinator-svelte]] - Polling coordination

## Used By

TBD

## Notes

Memory is polled automatically; system info is fetched on demand since it rarely changes. Uses dedicated equality functions to avoid updates when values are identical.
