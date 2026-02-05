"""E2E test configuration and fixtures.

E2E tests require:
- Models pre-downloaded to HuggingFace cache
- The app_client fixture (uses ASGI transport, no external server needed)

Run with: pytest -m e2e_vision_quick
Run all:  pytest -m e2e
"""

import pytest
import httpx
from pathlib import Path

from mlx_manager.mlx_server.models import pool as pool_module

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden"
IMAGES_DIR = FIXTURES_DIR / "images"

# Reference models for E2E testing
VISION_MODEL_QUICK = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
VISION_MODEL_FULL = "mlx-community/gemma-3-27b-it-4bit-DWQ"


def is_model_available(model_id: str) -> bool:
    """Check if a model is downloaded in HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_id:
                return True
    except Exception:
        pass
    return False


@pytest.fixture(scope="session")
def vision_model_quick():
    """Return quick vision model ID, skip if not downloaded."""
    if not is_model_available(VISION_MODEL_QUICK):
        pytest.skip(f"Model {VISION_MODEL_QUICK} not downloaded")
    return VISION_MODEL_QUICK


@pytest.fixture(scope="session")
def vision_model_full():
    """Return full vision model ID, skip if not downloaded."""
    if not is_model_available(VISION_MODEL_FULL):
        pytest.skip(f"Model {VISION_MODEL_FULL} not downloaded")
    return VISION_MODEL_FULL


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def app_client():
    """Async HTTP client for the FastAPI app.

    Uses ASGI transport to call the FastAPI app directly.
    The app lifespan initializes the model pool automatically.
    """
    from mlx_manager.main import app
    from httpx import ASGITransport

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=600.0,  # Vision inference can be slow
    ) as client:
        yield client


@pytest.fixture(autouse=True)
async def cleanup_pool():
    """Reset model pool after each test to free memory."""
    yield
    if pool_module.model_pool is not None:
        loaded = list(pool_module.model_pool._models.keys())
        for model_id in loaded:
            await pool_module.model_pool.unload_model(model_id)
