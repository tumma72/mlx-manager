---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/stores/index.ts
type: module
updated: 2026-01-21
status: active
---

# index.ts (stores)

## Purpose

Barrel export for all Svelte 5 runes-based stores. Provides a single import point for auth, server, profile, system, downloads, and model configuration state management, plus the polling coordinator for centralized refresh management.

## Exports

- `authStore` - Authentication state (token, user)
- `serverStore` - Running servers state and operations
- `profileStore` - Server profiles state and CRUD operations
- `systemStore` - System memory and info state
- `downloadsStore` - Download progress tracking
- `modelConfigStore` - Model characteristics cache
- `pollingCoordinator` - Centralized polling management

## Dependencies

- [[frontend-src-lib-stores-auth-svelte]] - Auth store
- [[frontend-src-lib-stores-servers-svelte]] - Servers store
- [[frontend-src-lib-stores-profiles-svelte]] - Profiles store
- [[frontend-src-lib-stores-system-svelte]] - System store
- [[frontend-src-lib-stores-downloads-svelte]] - Downloads store
- [[frontend-src-lib-stores-models-svelte]] - Model config store
- [[frontend-src-lib-services-index]] - Polling coordinator

## Used By

TBD
