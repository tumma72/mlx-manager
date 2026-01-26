---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/routers/system.py
type: api
updated: 2026-01-21
status: active
---

# system.py (router)

## Purpose

Provides REST API endpoints for system information and macOS launchd service management. Returns memory stats (with accurate physical RAM on macOS via sysctl), system info (OS, chip, versions), available parser options, and launchd service install/uninstall/status for auto-starting MLX servers at login.

## Exports

- `router` - FastAPI APIRouter with /api/system prefix
- `get_physical_memory_bytes()` - Get accurate physical RAM (sysctl on macOS)

## Dependencies

- [[backend-mlx_manager-config]] - Settings for memory recommendations
- [[backend-mlx_manager-database]] - Database session management
- [[backend-mlx_manager-dependencies]] - Authentication and profile lookup
- [[backend-mlx_manager-models]] - Response models
- [[backend-mlx_manager-services-launchd]] - Launchd service management
- [[backend-mlx_manager-services-parser_options]] - Parser options discovery
- psutil - Cross-platform system metrics
- platform - OS information

## Used By

TBD

## Notes

Uses sysctl on macOS for accurate physical memory (psutil can report inflated values including compressed memory). Memory is reported in GiB (binary, 1024^3).
