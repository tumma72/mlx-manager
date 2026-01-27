"""Model Pool Manager for loading and managing MLX models."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Container for a loaded model and its metadata."""

    model_id: str
    model: Any  # mlx_lm model
    tokenizer: Any  # HuggingFace tokenizer
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    size_gb: float = 0.0  # Estimated size

    def touch(self) -> None:
        """Update last_used timestamp."""
        self.last_used = time.time()


class ModelPoolManager:
    """Manages loading and caching of MLX models.

    Phase 7 implementation: Single model support.
    Phase 8 will add: Multi-model with LRU eviction.
    """

    def __init__(self, max_memory_gb: float = 48.0, max_models: int = 1):
        """Initialize the model pool.

        Args:
            max_memory_gb: Maximum memory for model pool
            max_models: Maximum number of hot models (1 for Phase 7)
        """
        self._models: dict[str, LoadedModel] = {}
        self._loading: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self.max_memory_gb = max_memory_gb
        self.max_models = max_models
        logger.info(
            f"ModelPoolManager initialized (max_memory={max_memory_gb}GB, max_models={max_models})"
        )

    async def get_model(self, model_id: str) -> LoadedModel:
        """Get a model, loading it if necessary.

        Args:
            model_id: HuggingFace model ID (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")

        Returns:
            LoadedModel with model and tokenizer

        Raises:
            RuntimeError: If model loading fails
        """
        async with self._lock:
            # Return cached model
            if model_id in self._models:
                loaded = self._models[model_id]
                loaded.touch()
                logger.debug(f"Model cache hit: {model_id}")
                return loaded

            # Wait if already loading
            if model_id in self._loading:
                logger.debug(f"Waiting for model load: {model_id}")

        # Wait outside lock if loading
        if model_id in self._loading:
            await self._loading[model_id].wait()
            return self._models[model_id]

        # Start loading
        return await self._load_model(model_id)

    async def _load_model(self, model_id: str) -> LoadedModel:
        """Load a model from HuggingFace.

        Args:
            model_id: Model identifier

        Returns:
            LoadedModel instance
        """
        async with self._lock:
            # Double-check after acquiring lock
            if model_id in self._models:
                return self._models[model_id]

            # Mark as loading
            self._loading[model_id] = asyncio.Event()

        logger.info(f"Loading model: {model_id}")
        start_time = time.time()

        try:
            # Import mlx_lm lazily
            from mlx_lm import load

            # Load in thread pool (blocking operation)
            # Note: load() returns (model, tokenizer) when return_config=False (default)
            result = await asyncio.to_thread(load, model_id)
            model, tokenizer = result[0], result[1]

            # Get memory after loading
            from mlx_manager.mlx_server.utils.memory import get_memory_usage

            memory = get_memory_usage()

            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=tokenizer,
                size_gb=memory["active_gb"],
            )

            async with self._lock:
                self._models[model_id] = loaded
                self._loading[model_id].set()
                del self._loading[model_id]

            elapsed = time.time() - start_time
            logger.info(f"Model loaded: {model_id} ({elapsed:.1f}s, {loaded.size_gb:.1f}GB)")
            return loaded

        except Exception as e:
            async with self._lock:
                if model_id in self._loading:
                    self._loading[model_id].set()
                    del self._loading[model_id]
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from the pool.

        Args:
            model_id: Model to unload

        Returns:
            True if model was unloaded, False if not found
        """
        async with self._lock:
            if model_id not in self._models:
                return False

            del self._models[model_id]

        # Clear cache after unloading
        from mlx_manager.mlx_server.utils.memory import clear_cache

        clear_cache()

        logger.info(f"Model unloaded: {model_id}")
        return True

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return list(self._models.keys())

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        return model_id in self._models

    async def cleanup(self) -> None:
        """Unload all models and clear cache."""
        async with self._lock:
            model_ids = list(self._models.keys())

        for model_id in model_ids:
            await self.unload_model(model_id)

        logger.info("Model pool cleaned up")


# Singleton instance (initialized in main.py lifespan)
model_pool: ModelPoolManager | None = None


def get_model_pool() -> ModelPoolManager:
    """Get the model pool singleton.

    Raises:
        RuntimeError: If pool not initialized
    """
    if model_pool is None:
        raise RuntimeError("Model pool not initialized")
    return model_pool
