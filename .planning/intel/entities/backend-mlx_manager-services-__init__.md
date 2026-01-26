---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/services/__init__.py
type: module
updated: 2026-01-21
status: active
---

# __init__.py (services)

## Purpose

Barrel export for backend service singletons. Provides a single import point for the core services that manage HuggingFace integration, server processes, health monitoring, and launchd services.

## Exports

- `hf_client` - HuggingFace client singleton for model operations
- `server_manager` - Server process manager singleton
- `health_checker` - Background health checker singleton
- `launchd_manager` - Launchd service manager singleton

## Dependencies

- [[backend-mlx_manager-services-health_checker]] - Health checker service
- [[backend-mlx_manager-services-hf_client]] - HuggingFace client service
- [[backend-mlx_manager-services-launchd]] - Launchd manager service
- [[backend-mlx_manager-services-server_manager]] - Server manager service

## Used By

TBD
