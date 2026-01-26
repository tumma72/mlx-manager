---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/services/health_checker.py
type: service
updated: 2026-01-21
status: active
---

# health_checker.py

## Purpose

Background service that periodically monitors the health of all running MLX servers. Runs on a configurable interval (default 30 seconds), checks each running instance, updates health status in the database, and detects stopped servers. Started/stopped via the application lifespan handler.

## Exports

- `HealthChecker` - Class with start(), stop(), and health check logic
- `health_checker` - Singleton instance

## Dependencies

- [[backend-mlx_manager-database]] - Database session for updating instances
- [[backend-mlx_manager-models]] - RunningInstance, ServerProfile models
- [[backend-mlx_manager-services-server_manager]] - For health check and running state

## Used By

TBD

## Notes

Marks instances as "stopped" if the process is no longer running. Uses asyncio task that can be cancelled on shutdown.
