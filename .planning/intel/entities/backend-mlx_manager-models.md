---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/models.py
type: model
updated: 2026-01-21
status: active
---

# models.py

## Purpose

Defines all SQLModel database entities and Pydantic response schemas for the backend. This is the single source of truth for data models including users, server profiles, running instances, downloads, and settings. Provides both database table definitions (with `table=True`) and API response/request schemas.

## Exports

- `UserStatus` - Enum for user account status (pending, approved, disabled)
- `UserBase` - Base model with email field
- `User` - User database table with auth fields
- `UserCreate` - Registration request schema
- `UserPublic` - Public user response (excludes password)
- `UserLogin` - Login request schema
- `UserUpdate` - Admin user update schema
- `Token` - JWT token response
- `PasswordReset` - Password reset request
- `ServerProfileBase` - Base model for profile settings
- `ServerProfile` - Server profile database table
- `ServerProfileCreate` - Profile creation schema
- `ServerProfileUpdate` - Profile update schema
- `RunningInstance` - Running server instance tracking table
- `DownloadedModel` - Downloaded model cache table
- `Setting` - Application settings table
- `Download` - Active download tracking table
- `ServerProfileResponse` - Profile API response
- `RunningServerResponse` - Running server API response
- `ModelSearchResult` - HuggingFace search result schema
- `LocalModel` - Local model info schema
- `SystemMemory` - Memory stats schema
- `SystemInfo` - System info schema
- `HealthStatus` - Health check response
- `LaunchdStatus` - Launchd status response
- `ServerStatus` - Server process status

## Dependencies

- sqlmodel - ORM and schema validation framework

## Used By

TBD
