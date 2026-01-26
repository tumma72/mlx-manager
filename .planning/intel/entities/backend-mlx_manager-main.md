---
path: /Users/atomasini/Development/mlx-manager/backend/mlx_manager/main.py
type: module
updated: 2026-01-21
status: active
---

# main.py

## Purpose

FastAPI application entry point that configures the web server, registers routers, sets up CORS middleware, manages application lifecycle (startup/shutdown), and handles static file serving for the embedded SvelteKit frontend. The lifespan handler initializes the database, cleans up stale instances, resumes incomplete downloads, and starts the background health checker.

## Exports

- `app` - FastAPI application instance
- `cleanup_stale_instances()` - Remove orphaned running instance records
- `cancel_download_tasks()` - Cancel active downloads on shutdown
- `resume_pending_downloads(pending)` - Resume interrupted downloads
- `lifespan(app)` - Application lifespan context manager
- `health()` - Health check endpoint
- `serve_spa(request, full_path)` - SPA routing with fallback to index.html

## Dependencies

- [[backend-mlx_manager-database]] - Database initialization and sessions
- [[backend-mlx_manager-models]] - RunningInstance model
- [[backend-mlx_manager-routers-__init__]] - All API routers
- [[backend-mlx_manager-services-health_checker]] - Background health monitoring
- [[backend-mlx_manager-services-hf_client]] - HuggingFace client for downloads
- [[backend-mlx_manager-services-server_manager]] - Server process management
- fastapi - Web framework
- uvicorn - ASGI server (when run directly)

## Used By

TBD

## Notes

Static file serving is conditional - only enabled when the frontend build exists in mlx_manager/static/. In development, the frontend runs separately on port 5173.
