"""E2E test configuration and fixtures.

E2E tests require:
- Models pre-downloaded to HuggingFace cache
- The app_client fixture (uses ASGI transport, no external server needed)

Run with: pytest -m e2e_vision_quick
Run all:  pytest -m e2e
"""

import os
from pathlib import Path

import httpx
import pytest

# Set test database to in-memory before importing app modules
os.environ.setdefault("MLX_MANAGER_DATABASE_PATH", ":memory:")

from mlx_manager.mlx_server.models import pool as pool_module
from mlx_manager.mlx_server.models.pool import ModelPoolManager

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden"
IMAGES_DIR = FIXTURES_DIR / "images"

# Reference models for E2E testing.
# Order matters: first available model is selected.
# Prefer qat variants over DWQ for Gemma due to VisionConfig compatibility.
VISION_MODELS_QUICK = [
    "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "mlx-community/gemma-3-12b-it-qat-4bit",
]
VISION_MODELS_FULL = [
    "mlx-community/gemma-3-27b-it-qat-4bit",
    "mlx-community/gemma-3-27b-it-4bit-DWQ",
]

# Reference model for cross-protocol text testing
TEXT_MODEL_QUICK = "mlx-community/Qwen3-0.6B-4bit-DWQ"

PROMPTS_DIR = GOLDEN_DIR / "prompts"

# Shared tool definition for tool call tests (OpenAI function calling format)
WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo'",
                }
            },
            "required": ["location"],
        },
    },
}


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


def _find_available_model(candidates: list[str]) -> str | None:
    """Find the first available model from a list of candidates."""
    for model_id in candidates:
        if is_model_available(model_id):
            return model_id
    return None


@pytest.fixture(scope="session")
def vision_model_quick():
    """Return quick vision model ID, skip if none available."""
    model = _find_available_model(VISION_MODELS_QUICK)
    if model is None:
        pytest.skip(
            f"No quick vision model available. Need one of: {', '.join(VISION_MODELS_QUICK)}"
        )
    return model


@pytest.fixture(scope="session")
def vision_model_full():
    """Return full vision model ID, skip if none available."""
    model = _find_available_model(VISION_MODELS_FULL)
    if model is None:
        pytest.skip(f"No full vision model available. Need one of: {', '.join(VISION_MODELS_FULL)}")
    return model


@pytest.fixture(scope="session")
def text_model_quick():
    """Return quick text model ID, skip if not downloaded."""
    if not is_model_available(TEXT_MODEL_QUICK):
        pytest.skip(f"Model {TEXT_MODEL_QUICK} not downloaded")
    return TEXT_MODEL_QUICK


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def app_client():
    """Async HTTP client for the FastAPI app.

    Initializes the model pool and database before creating the client,
    since httpx ASGITransport does not trigger FastAPI lifespan handlers.
    """
    from httpx import ASGITransport

    from mlx_manager.database import init_db
    from mlx_manager.main import app

    # Initialize database (creates tables in memory)
    await init_db()

    # Initialize model pool if not already set.
    # Use generous memory limit to support large models like Gemma-3-27b.
    if pool_module.model_pool is None:
        pool_module.model_pool = ModelPoolManager(
            max_memory_gb=48.0,
            max_models=4,
        )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=600.0,  # Vision inference can be slow
    ) as client:
        yield client


# Reference model for embeddings testing
EMBEDDINGS_MODEL = "mlx-community/all-MiniLM-L6-v2-4bit"


@pytest.fixture(scope="session")
def embeddings_model():
    """Return embeddings model ID, skip if not downloaded."""
    if not is_model_available(EMBEDDINGS_MODEL):
        pytest.skip(f"Model {EMBEDDINGS_MODEL} not downloaded")
    return EMBEDDINGS_MODEL


# Reference model for audio TTS testing
AUDIO_TTS_MODEL = "mlx-community/Kokoro-82M-4bit"


@pytest.fixture(scope="session")
def audio_tts_model():
    """Return audio TTS model ID, skip if not downloaded."""
    if not is_model_available(AUDIO_TTS_MODEL):
        pytest.skip(f"Model {AUDIO_TTS_MODEL} not downloaded")
    return AUDIO_TTS_MODEL


@pytest.fixture(autouse=True)
async def cleanup_pool():
    """Reset model pool after each test to free memory."""
    yield
    if pool_module.model_pool is not None:
        loaded = list(pool_module.model_pool._models.keys())
        for model_id in loaded:
            await pool_module.model_pool.unload_model(model_id)
