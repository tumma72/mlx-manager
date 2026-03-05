"""MLX Model Manager - FastAPI Application."""

# Suppress deprecation warnings from mlx-lm (uses deprecated mx.metal.device_info)
import warnings

warnings.filterwarnings(
    "ignore",
    message="mx.metal.device_info is deprecated",
    category=DeprecationWarning,
)

# Configure Loguru FIRST (before any other imports)
from mlx_manager.logging_config import intercept_standard_logging, setup_logging

setup_logging()
intercept_standard_logging()

from loguru import logger

# Configure LogFire (after logging, before instrumented imports)
from mlx_manager import __version__
from mlx_manager.observability.logfire_config import (
    configure_logfire,
    instrument_fastapi,
    instrument_httpx,
    instrument_sqlalchemy,
)

configure_logfire(service_version=__version__)
instrument_httpx()

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mlx_manager.config import settings as manager_settings
from mlx_manager.database import (
    async_session,
    detect_orphaned_downloads,
    engine,
    init_db,
    recover_incomplete_downloads,
)
from mlx_manager.mlx_server.config import mlx_server_settings

# MLX Server imports for embedded mode
from mlx_manager.mlx_server.main import create_app as create_mlx_server_app
from mlx_manager.mlx_server.models import pool
from mlx_manager.mlx_server.models.pool import ModelPoolManager
from mlx_manager.mlx_server.utils.memory import set_memory_limit
from mlx_manager.routers import (
    auth_router,
    mcp_router,
    models_router,
    profiles_router,
    servers_router,
    settings_router,
    system_router,
)
from mlx_manager.routers.models import download_tasks
from mlx_manager.services.health_checker import health_checker
from mlx_manager.services.hf_client import hf_client

# Instrument SQLAlchemy after engine is imported
instrument_sqlalchemy(engine)

# Static files directory (embedded frontend build)
STATIC_DIR = Path(__file__).parent / "static"


# Track running download tasks for cleanup on shutdown
_download_tasks: list[asyncio.Task[None]] = []


async def cancel_download_tasks() -> None:
    """Cancel all running download tasks on shutdown."""
    if _download_tasks:
        logger.info(f"Cancelling {len(_download_tasks)} running download tasks...")
        for task in _download_tasks:
            if not task.done():
                task.cancel()
        # Wait for all tasks to be cancelled
        await asyncio.gather(*_download_tasks, return_exceptions=True)
        _download_tasks.clear()
        logger.info("Download tasks cancelled")


async def resume_pending_downloads(pending: list[tuple[int, str]]) -> None:
    """Resume downloads that were interrupted by server restart.

    Creates background tasks for each pending download and registers them
    in download_tasks so the frontend can connect via SSE.
    """
    import uuid

    for download_id, model_id in pending:
        task_id = str(uuid.uuid4())

        # Register in download_tasks so frontend can connect
        download_tasks[task_id] = {
            "model_id": model_id,
            "download_id": download_id,
            "status": "pending",
            "progress": 0,
        }

        # Create background task for the download and track it
        task = asyncio.create_task(_run_download_task(task_id, download_id, model_id))
        _download_tasks.append(task)
        logger.info(f"Resuming download for {model_id} (task_id={task_id})")


async def _run_download_task(task_id: str, download_id: int, model_id: str) -> None:
    """Background task to run a download and update progress."""
    from mlx_manager.routers.models import _update_download_record
    from mlx_manager.services.hf_client import (
        cleanup_cancel_event,
        register_cancel_event,
    )

    # Register cancel event so pause/cancel endpoints can signal this download
    cancel_event = register_cancel_event(str(download_id))

    try:
        async for progress in hf_client.download_model(model_id, cancel_event=cancel_event):
            download_tasks[task_id].update(progress.model_dump(exclude_none=True))

            status = progress.status
            is_final = status in ("completed", "failed", "cancelled")

            # Update DB on status changes and completion
            if is_final:
                await _update_download_record(
                    download_id,
                    status=status or "downloading",
                    downloaded_bytes=progress.downloaded_bytes or 0,
                    total_bytes=progress.total_bytes,
                    error=progress.error,
                    completed=status == "completed",
                )
                break
    except asyncio.CancelledError:
        # Task was cancelled during shutdown - this is expected
        logger.info(f"Download task cancelled for {model_id}")
        raise  # Re-raise to properly mark task as cancelled
    except Exception as e:
        logger.exception(f"Download failed for {model_id}: {e}")
        await _update_download_record(download_id, status="failed", error=str(e))
    finally:
        cleanup_cancel_event(str(download_id))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("MLX Manager starting up...")

    await init_db()

    # Initialize MLX Server model pool
    # Auto-detect memory limit from device if not explicitly configured
    max_memory_gb = mlx_server_settings.max_memory_gb
    if max_memory_gb <= 0:
        from mlx_manager.mlx_server.utils.memory import auto_detect_memory_limit

        max_memory_gb = auto_detect_memory_limit()
    set_memory_limit(max_memory_gb)
    pool.model_pool = ModelPoolManager(
        max_memory_gb=max_memory_gb,
        max_models=mlx_server_settings.max_models,
    )
    logger.info("MLX Server model pool initialized")

    # Initialize scheduler manager if batching enabled
    scheduler_mgr = None
    if mlx_server_settings.enable_batching:
        from mlx_manager.mlx_server.services.batching import init_scheduler_manager

        scheduler_mgr = init_scheduler_manager(
            block_pool_size=mlx_server_settings.batch_block_pool_size,
            max_batch_size=mlx_server_settings.batch_max_batch_size,
        )
        logger.info("MLX Server batching scheduler initialized")

    # Resume any incomplete downloads from previous sessions
    pending_downloads = await recover_incomplete_downloads()

    # Detect orphaned downloads (started outside MLX Manager) and adopt them
    orphaned_downloads = await detect_orphaned_downloads()
    pending_downloads.extend(orphaned_downloads)

    if pending_downloads:
        await resume_pending_downloads(pending_downloads)

    # Auto-load profiles marked with auto_start
    from mlx_manager.database import get_session
    from mlx_manager.mlx_server.models.pool import ProfileSettings

    async with get_session() as session:
        from sqlalchemy.orm import selectinload
        from sqlmodel import select

        from mlx_manager.models import ExecutionProfile

        result = await session.execute(
            select(ExecutionProfile)
            .where(ExecutionProfile.auto_start == True)  # noqa: E712
            .options(selectinload(ExecutionProfile.model))  # type: ignore[arg-type]
        )
        auto_start_profiles = result.scalars().all()

        if auto_start_profiles and pool.model_pool:
            for profile in auto_start_profiles:
                if not profile.model:
                    continue
                model_id = profile.model.repo_id
                try:
                    # Parse model_options to template_options
                    template_options = None
                    if profile.model_options:
                        import json

                        try:
                            template_options = json.loads(profile.model_options)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Register profile settings
                    pool.model_pool.register_profile_settings(
                        model_id,
                        ProfileSettings(
                            system_prompt=profile.default_system_prompt,
                            enable_tool_injection=profile.default_enable_tool_injection,
                            template_options=template_options,
                        ),
                    )

                    # Preload model (protected from eviction)
                    await pool.model_pool.preload_model(model_id)
                    logger.info(f"Auto-loaded profile '{profile.name}' (model: {model_id})")
                except Exception as e:
                    logger.exception(f"Failed to auto-load profile '{profile.name}': {e}")

            if auto_start_profiles:
                loaded = sum(
                    1
                    for p in auto_start_profiles
                    if p.model and pool.model_pool.is_loaded(p.model.repo_id)
                )
                logger.info(f"Auto-start: {loaded}/{len(auto_start_profiles)} profiles loaded")

    await health_checker.start()
    logger.info("MLX Manager ready")

    yield

    # Shutdown
    await cancel_download_tasks()
    await health_checker.stop()

    # Shutdown scheduler manager if initialized
    if scheduler_mgr is not None:
        await scheduler_mgr.shutdown()
        logger.info("MLX Server batching scheduler shutdown complete")

    # Cleanup model pool
    if pool.model_pool:
        await pool.model_pool.cleanup()
        logger.info("MLX Server model pool cleaned up")

    # Shut down the persistent Metal GPU worker thread and its Queue (which
    # uses a semaphore internally). Without this, the resource_tracker reports
    # a leaked semaphore on reloader shutdown.
    from mlx_manager.mlx_server.utils.metal import reset_metal_worker

    reset_metal_worker()

    # Dispose BOTH database engines to close all aiosqlite connections cleanly.
    # Without this, background threads and their synchronization primitives
    # leak on reloader shutdown (manifests as "leaked semaphore objects" warning).
    await engine.dispose()

    # Also dispose the MLX Server's separate engine (used for audit logs, etc.)
    from mlx_manager.mlx_server.database import _get_engine as get_mlx_engine

    try:
        mlx_engine = get_mlx_engine()
        await mlx_engine.dispose()
    except Exception:
        pass  # Engine may not have been initialized

    # Complete any pending log writes before shutdown
    await logger.complete()


app = FastAPI(
    title="MLX Model Manager",
    description="Web-based application for managing MLX-optimized language models",
    version=__version__,
    lifespan=lifespan,
)

# Instrument FastAPI with LogFire (after app creation)
instrument_fastapi(app)

# Share the manager's engine with the embedded MLX Server so both components
# use the same connection pool to the same SQLite file.  Must happen before
# create_mlx_server_app() so init_db() inside the server lifespan picks it up.
from mlx_manager.mlx_server.database import set_shared_engine

set_shared_engine(engine, async_session)

# Mount MLX Server at /v1 prefix (after instrumentation for proper tracing)
# Routes: /v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings
# Admin routes: /v1/admin/*
mlx_server_app = create_mlx_server_app(embedded=True)
app.mount("/v1", mlx_server_app, name="mlx_server")

# CORS configuration - more permissive since we serve frontend from same origin
app.add_middleware(
    CORSMiddleware,  # type: ignore[invalid-argument-type]
    allow_origins=[
        "http://localhost:5173",  # SvelteKit dev server
        "http://127.0.0.1:5173",
        "http://localhost:4173",  # SvelteKit preview
        "http://127.0.0.1:4173",
        f"http://localhost:{manager_settings.port}",  # Current port
        f"http://127.0.0.1:{manager_settings.port}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth_router)
app.include_router(mcp_router)
app.include_router(profiles_router)
app.include_router(models_router)
app.include_router(servers_router)
app.include_router(settings_router)
app.include_router(system_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


# Static file serving for production (embedded frontend)
if STATIC_DIR.exists():
    # Mount static assets directory
    assets_dir = STATIC_DIR / "_app"
    if assets_dir.exists():
        app.mount("/_app", StaticFiles(directory=assets_dir), name="app_assets")

    @app.get("/favicon.png")
    async def favicon() -> Response:
        """Serve favicon."""
        favicon_path = STATIC_DIR / "favicon.png"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        return JSONResponse({"error": "Not found"}, status_code=404)

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str) -> Response:
        """Serve SPA with fallback to index.html."""
        # Skip API routes (they should be handled by routers)
        # Note: MLX Server routes at /v1/* are handled by mounted sub-app
        if full_path.startswith("api/"):
            return JSONResponse({"error": "Not found"}, status_code=404)

        # Try to serve the exact file
        file_path = STATIC_DIR / full_path
        resolved = file_path.resolve()
        if not resolved.is_relative_to(STATIC_DIR.resolve()):
            return JSONResponse({"error": "Not found"}, status_code=404)
        if resolved.exists() and resolved.is_file():
            return FileResponse(resolved)

        # Fallback to index.html for SPA routing
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

        return JSONResponse({"error": "Not found"}, status_code=404)

else:
    # Development mode - no static files, frontend runs separately

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint (dev mode)."""
        return {
            "name": "MLX Model Manager",
            "version": __version__,
            "docs": "/docs",
            "note": "Frontend not embedded. Run frontend dev server separately.",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=manager_settings.port)
