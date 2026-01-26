---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/routers/__init__.py
type: module
updated: 2026-01-21
status: active
---

# __init__.py (routers)

## Purpose

Barrel export for all FastAPI routers. Provides a single import point for including all API route handlers in the main application.

## Exports

- `auth_router` - Authentication and user management endpoints
- `models_router` - Model search, download, and management endpoints
- `profiles_router` - Server profile CRUD endpoints
- `servers_router` - Server lifecycle control endpoints
- `system_router` - System info and launchd management endpoints

## Dependencies

- [[backend-mlx_manager-routers-auth]] - Auth router
- [[backend-mlx_manager-routers-models]] - Models router
- [[backend-mlx_manager-routers-profiles]] - Profiles router
- [[backend-mlx_manager-routers-servers]] - Servers router
- [[backend-mlx_manager-routers-system]] - System router

## Used By

TBD
