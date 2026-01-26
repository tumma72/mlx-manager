---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/routers/profiles.py
type: api
updated: 2026-01-21
status: active
---

# profiles.py (router)

## Purpose

Provides REST API endpoints for server profile CRUD operations. Profiles define mlx-openai-server configurations including model path, port, parser options, and other settings. Supports listing, creating, updating, deleting, and duplicating profiles, plus determining the next available port.

## Exports

- `router` - FastAPI APIRouter with /api/profiles prefix

## Dependencies

- [[backend-mlx_manager-config]] - Settings for default port
- [[backend-mlx_manager-database]] - Database session management
- [[backend-mlx_manager-dependencies]] - Authentication and profile lookup
- [[backend-mlx_manager-models]] - ServerProfile models and schemas
- fastapi - Web framework
- sqlmodel - Database queries

## Used By

TBD

## Notes

Enforces unique names and ports across profiles. Duplicate operation copies all settings except auto_start flag and assigns the next available port.
