"""MLX Model Manager - FastAPI Application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mlx_manager import __version__
from mlx_manager.database import init_db
from mlx_manager.routers import models_router, profiles_router, servers_router, system_router
from mlx_manager.services.health_checker import health_checker
from mlx_manager.services.server_manager import server_manager

# Static files directory (embedded frontend build)
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    await health_checker.start()

    yield

    # Shutdown
    await health_checker.stop()
    await server_manager.cleanup()


app = FastAPI(
    title="MLX Model Manager",
    description="Web-based application for managing MLX-optimized language models",
    version=__version__,
    lifespan=lifespan,
)

# CORS configuration - more permissive since we serve frontend from same origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # SvelteKit dev server
        "http://127.0.0.1:5173",
        "http://localhost:4173",  # SvelteKit preview
        "http://127.0.0.1:4173",
        "http://localhost:8080",  # Production
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(profiles_router)
app.include_router(models_router)
app.include_router(servers_router)
app.include_router(system_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Static file serving for production (embedded frontend)
if STATIC_DIR.exists():
    # Mount static assets directory
    assets_dir = STATIC_DIR / "_app"
    if assets_dir.exists():
        app.mount("/_app", StaticFiles(directory=assets_dir), name="app_assets")

    @app.get("/favicon.png")
    async def favicon():
        """Serve favicon."""
        favicon_path = STATIC_DIR / "favicon.png"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        return JSONResponse({"error": "Not found"}, status_code=404)

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve SPA with fallback to index.html."""
        # Skip API routes (they should be handled by routers)
        if full_path.startswith("api/"):
            return JSONResponse({"error": "Not found"}, status_code=404)

        # Try to serve the exact file
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Fallback to index.html for SPA routing
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

        return JSONResponse({"error": "Not found"}, status_code=404)

else:
    # Development mode - no static files, frontend runs separately

    @app.get("/")
    async def root():
        """Root endpoint (dev mode)."""
        return {
            "name": "MLX Model Manager",
            "version": __version__,
            "docs": "/docs",
            "note": "Frontend not embedded. Run frontend dev server separately.",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
