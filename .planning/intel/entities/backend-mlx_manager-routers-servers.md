---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/routers/servers.py
type: api
updated: 2026-01-21
status: active
---

# servers.py (router)

## Purpose

Provides REST API endpoints for managing mlx-openai-server process lifecycle. Supports listing running servers, starting/stopping/restarting server instances, health checks, status queries, and live log streaming via Server-Sent Events. Maintains database records of running instances for persistence across manager restarts.

## Exports

- `router` - FastAPI APIRouter with /api/servers prefix

## Dependencies

- [[backend-mlx_manager-database]] - Database session management
- [[backend-mlx_manager-dependencies]] - Authentication and profile lookup
- [[backend-mlx_manager-models]] - RunningInstance, ServerProfile, and response models
- [[backend-mlx_manager-services-server_manager]] - Server process management
- fastapi - Web framework and streaming responses
- sqlmodel - Database queries

## Used By

TBD

## Notes

Cleans up stale running_instance records when the actual process is no longer running. Log streaming uses SSE for real-time updates.
