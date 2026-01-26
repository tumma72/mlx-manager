---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/dependencies.py
type: module
updated: 2026-01-21
status: active
---

# dependencies.py

## Purpose

FastAPI dependencies for common operations across routes. Provides reusable dependency functions for authentication (current user from JWT), authorization (admin check), and resource lookup (profile by ID with 404 handling). Uses OAuth2 bearer token scheme for JWT extraction.

## Exports

- `oauth2_scheme` - OAuth2PasswordBearer for token extraction from Authorization header
- `get_profile_or_404(profile_id, session) -> ServerProfile` - Lookup profile or raise 404
- `get_current_user(token, session) -> User` - Validate JWT and return authenticated user
- `get_admin_user(current_user) -> User` - Verify user is admin or raise 403

## Dependencies

- [[backend-mlx_manager-database]] - Database session injection
- [[backend-mlx_manager-models]] - ServerProfile, User, UserStatus
- [[backend-mlx_manager-services-auth_service]] - JWT token decoding
- fastapi - Depends, HTTPException, OAuth2PasswordBearer
- sqlmodel - Database queries

## Used By

TBD
