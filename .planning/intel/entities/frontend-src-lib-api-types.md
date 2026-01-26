---
path: /Users/atomasini/Development/mlx-manager/frontend/src/lib/api/types.ts
type: model
updated: 2026-01-21
status: active
---

# types.ts

## Purpose

Defines all TypeScript interfaces for the frontend API layer, providing type-safe contracts for data exchanged between the SvelteKit frontend and FastAPI backend. These types mirror the backend Pydantic/SQLModel schemas and ensure compile-time type checking for API responses and requests.

## Exports

- `ServerProfile` - Complete server profile configuration with all mlx-openai-server options
- `ServerProfileCreate` - Schema for creating new server profiles
- `ServerProfileUpdate` - Schema for partial profile updates
- `RunningServer` - Running server instance with metrics (memory, CPU, uptime)
- `ModelSearchResult` - HuggingFace model search result with metadata
- `ModelCharacteristics` - Model architecture details from config.json
- `LocalModel` - Locally downloaded model information
- `SystemMemory` - System memory stats including MLX recommendations
- `SystemInfo` - System information (OS, chip, versions)
- `DownloadProgress` - Model download progress tracking
- `HealthStatus` - Server health check result
- `LaunchdStatus` - macOS launchd service status
- `ServerStatus` - Detailed server process status
- `ModelDetectionInfo` - Model family detection and parser recommendations
- `ParserOptions` - Available parser options from mlx-openai-server
- `UserStatus` - User account status enum type
- `User` - User account information
- `Token` - JWT authentication token
- `UserCreate` - Registration request schema
- `PasswordReset` - Password reset request schema
- `UserUpdate` - User update request schema

## Dependencies

None

## Used By

TBD
