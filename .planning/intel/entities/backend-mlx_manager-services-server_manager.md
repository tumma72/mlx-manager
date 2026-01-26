---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/services/server_manager.py
type: service
updated: 2026-01-21
status: active
---

# server_manager.py

## Purpose

Manages mlx-openai-server subprocess lifecycle. Handles starting servers with proper command-line arguments, stopping with graceful SIGTERM or force SIGKILL, health checking via /v1/models endpoint, collecting process stats (memory, CPU), reading log output, and detecting failures. Central coordination point for all server process operations.

## Exports

- `ServerManager` - Class for server process management
- `server_manager` - Singleton instance

## Dependencies

- [[backend-mlx_manager-models]] - ServerProfile configuration
- [[backend-mlx_manager-types]] - HealthCheckResult, ServerStats, RunningServerInfo
- [[backend-mlx_manager-utils-command_builder]] - Build server command and log path
- [[backend-mlx_manager-utils-model_detection]] - Check mlx-lm version support
- httpx - HTTP client for health checks
- psutil - Process stats collection
- subprocess - Process management

## Used By

TBD

## Notes

Pre-launch version check prevents starting unsupported model families. Logs are captured to files and streamed via SSE. Process status detection includes checking log content for error patterns since mlx-openai-server may exit with code 0 on some errors.
