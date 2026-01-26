---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/services/polling-coordinator.svelte.ts
type: service
updated: 2026-01-21
status: active
---

# polling-coordinator.svelte.ts

## Purpose

Centralized polling management singleton for all frontend stores. Provides request deduplication (returns existing promise if refresh is in-flight), throttling (prevents rapid successive calls), tab visibility handling (pauses when hidden, resumes with immediate refresh when visible), and registration-based polling for servers, profiles, and system memory.

## Exports

- `PollingKey` - Type for polling identifiers ("servers" | "profiles" | "system-memory")
- `PollingConfig` - Interface for polling configuration
- `PollingCoordinator` - Main coordinator class
- `pollingCoordinator` - Singleton instance

## Dependencies

None

## Used By

TBD

## Notes

Key methods: register() to configure polling, start()/stop() to control intervals, refresh() for manual refresh with deduplication, pause()/resume() for key-specific control, setGlobalPause() for app-wide pause. Automatically handles document.visibilitychange events.
