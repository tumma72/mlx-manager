---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/routers/models.py
type: api
updated: 2026-01-21
status: active
---

# models.py (router)

## Purpose

Provides REST API endpoints for MLX model operations including HuggingFace model search, local model listing, model downloads with SSE progress streaming, model deletion, and model configuration/detection. Handles the complete model lifecycle from discovery to local management.

## Exports

- `router` - FastAPI APIRouter with /api/models prefix
- `DownloadRequest` - Request body for starting downloads
- `download_tasks` - In-memory dictionary tracking active downloads

## Dependencies

- [[backend-mlx_manager-database]] - Database session management
- [[backend-mlx_manager-dependencies]] - Authentication dependencies
- [[backend-mlx_manager-models]] - Data models and schemas
- [[backend-mlx_manager-services-hf_client]] - HuggingFace client for search/download
- [[backend-mlx_manager-services-parser_options]] - Parser options discovery
- [[backend-mlx_manager-utils-model_detection]] - Model family detection
- fastapi - Web framework
- sqlmodel - Database queries

## Used By

TBD

## Notes

Uses Server-Sent Events (SSE) for real-time download progress. Downloads are tracked both in-memory (for active connections) and in database (for persistence across restarts).
