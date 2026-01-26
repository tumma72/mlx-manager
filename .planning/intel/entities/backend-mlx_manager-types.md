---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/types.py
type: model
updated: 2026-01-21
status: active
---

# types.py

## Purpose

Defines TypedDict types for structured data used across the backend services. These types provide type hints for dictionaries returned by services like health checks, server stats, model search results, and download status. They complement SQLModel schemas by typing internal service data structures.

## Exports

- `HealthCheckResult` - Result from server health check with status, response time, model loaded state
- `ServerStats` - Statistics for running server process (memory, CPU, status)
- `RunningServerInfo` - Combined running server information
- `ModelSearchResult` - Search result from HuggingFace Hub
- `ModelCharacteristics` - Model characteristics from config.json (architecture, quantization, multimodal)
- `LocalModelInfo` - Information about locally downloaded model
- `DownloadStatus` - Status update for model download progress
- `LaunchdStatus` - Status of a launchd service

## Dependencies

None (uses only Python typing module)

## Used By

TBD
