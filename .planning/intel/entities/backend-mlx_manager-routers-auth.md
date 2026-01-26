---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/routers/auth.py
type: api
updated: 2026-01-21
status: active
---

# auth.py (router)

## Purpose

Provides REST API endpoints for user authentication and user management. Handles user registration (first user becomes admin), JWT login, current user retrieval, and admin-only user management (list, update status, delete, reset password). Implements the approval workflow where non-admin users start in pending status.

## Exports

- `router` - FastAPI APIRouter with /api/auth prefix

## Dependencies

- [[backend-mlx_manager-database]] - Database session management
- [[backend-mlx_manager-dependencies]] - Authentication dependencies
- [[backend-mlx_manager-models]] - User models and schemas
- [[backend-mlx_manager-services-auth_service]] - Password hashing and JWT tokens
- fastapi - Web framework and OAuth2 support
- sqlalchemy - Database queries

## Used By

TBD

## Notes

First registered user automatically becomes admin and is approved. Prevents removing/disabling the last admin. Uses OAuth2PasswordRequestForm for login compatibility with standard OAuth2 clients.
