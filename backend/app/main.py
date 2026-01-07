"""MLX Model Manager - FastAPI Application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import models_router, profiles_router, servers_router, system_router
from app.services.health_checker import health_checker
from app.services.server_manager import server_manager


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
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # SvelteKit dev server
        "http://127.0.0.1:5173",
        "http://localhost:4173",  # SvelteKit preview
        "http://127.0.0.1:4173",
        "http://localhost:8080",  # Same origin
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(profiles_router)
app.include_router(models_router)
app.include_router(servers_router)
app.include_router(system_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "MLX Model Manager",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
