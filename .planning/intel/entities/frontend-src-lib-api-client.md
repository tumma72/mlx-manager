---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/api/client.ts
type: api
updated: 2026-01-21
status: active
---

# client.ts

## Purpose

Type-safe API client for communicating with the FastAPI backend. Provides organized API namespaces (auth, profiles, models, servers, system) with methods that handle authentication headers, response parsing, error handling, and automatic 401 redirect to login. All methods return typed promises matching the backend response schemas.

## Exports

- `ApiError` - Custom error class with HTTP status code
- `auth` - Authentication API (register, login, me, listUsers, updateUser, deleteUser, resetPassword, getPendingCount)
- `profiles` - Profile management API (list, get, create, update, delete, duplicate, getNextPort)
- `models` - Model operations API (search, listLocal, startDownload, getActiveDownloads, delete, detectOptions, getAvailableParsers, getConfig)
- `servers` - Server control API (list, start, stop, restart, health, status)
- `system` - System info API (memory, info, parserOptions, launchd install/uninstall/status)
- `ActiveDownload` - Interface for active download info

## Dependencies

- [[frontend-src-lib-api-types]] - All TypeScript interfaces
- [[frontend-src-lib-stores-auth-svelte]] - Auth store for token injection

## Used By

TBD

## Notes

Handles 401 responses by clearing auth state and redirecting to login. Parses FastAPI validation error arrays into readable messages.
