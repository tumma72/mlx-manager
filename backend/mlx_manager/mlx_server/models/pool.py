"""Model Pool Manager for loading and managing MLX models."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
from fastapi import HTTPException

from mlx_manager.mlx_server.models.detection import detect_model_type
from mlx_manager.mlx_server.models.types import AdapterInfo, ModelType

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
    model_type: str = "text-gen"  # Type of model (from ModelType enum values)
    preloaded: bool = False  # Whether protected from eviction
    adapter_path: str | None = None  # Path to LoRA adapter if loaded with one
    adapter_info: AdapterInfo | None = None  # Adapter metadata if applicable

    def touch(self) -> None:
        """Update last_used timestamp."""
        self.last_used = time.time()


class ModelPoolManager:
    """Manages loading and caching of MLX models.

    Supports multi-model hot-swapping with LRU eviction for memory management.
    Preloaded models are protected from eviction.
    """

    def __init__(
        self,
        max_memory_gb: float = 48.0,
        max_models: int = 4,
        memory_limit_pct: float | None = None,
    ):
        """Initialize the model pool.

        Args:
            max_memory_gb: Maximum memory for model pool in GB (default: 48GB)
            max_models: Maximum number of hot models (default: 4)
            memory_limit_pct: Alternative to max_memory_gb as percentage of system memory
                             (e.g., 0.75 = 75%). Takes precedence over max_memory_gb.
        """
        self._models: dict[str, LoadedModel] = {}
        self._loading: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self.max_memory_gb = max_memory_gb
        self.max_models = max_models
        self._memory_limit_pct = memory_limit_pct
        logger.info(
            f"ModelPoolManager initialized (max_memory={max_memory_gb}GB, max_models={max_models})"
        )

    def _get_effective_memory_limit(self) -> float:
        """Get the effective memory limit in GB.

        If memory_limit_pct is set, calculates from total system memory.
        Otherwise returns max_memory_gb.

        Returns:
            Memory limit in GB
        """
        if self._memory_limit_pct is not None:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            return total_memory_gb * self._memory_limit_pct
        return self.max_memory_gb

    def _current_memory_gb(self) -> float:
        """Get the total memory used by loaded models.

        Returns:
            Sum of all loaded model sizes in GB
        """
        return sum(m.size_gb for m in self._models.values())

    def _evictable_models(self) -> list[LoadedModel]:
        """Get list of models that can be evicted (not preloaded).

        Returns:
            List of LoadedModel instances where preloaded is False
        """
        return [m for m in self._models.values() if not m.preloaded]

    async def _evict_lru(self) -> bool:
        """Evict the least recently used non-preloaded model.

        Returns:
            True if a model was evicted, False if no evictable models exist
        """
        evictable = self._evictable_models()
        if not evictable:
            return False

        # Find LRU model
        lru_model = min(evictable, key=lambda m: m.last_used)

        # Remove from pool
        del self._models[lru_model.model_id]

        # Clear cache
        from mlx_manager.mlx_server.utils.memory import clear_cache

        clear_cache()

        logger.info(
            f"Evicted LRU model: {lru_model.model_id} "
            f"(size={lru_model.size_gb:.1f}GB, last_used={lru_model.last_used:.0f})"
        )
        return True

    def _estimate_model_size(self, model_id: str) -> float:
        """Estimate model size based on name patterns.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Estimated size in GB
        """
        # Look for parameter count patterns like "3B", "7B", "13B", "70B"
        match = re.search(r"(\d+)[bB]", model_id)
        if match:
            param_count = int(match.group(1))
            # Rough mapping: 3B -> 2GB, 7B -> 4GB, 8B -> 5GB, 13B -> 8GB, 70B -> 40GB
            size_mapping = {
                3: 2.0,
                7: 4.0,
                8: 5.0,
                13: 8.0,
                70: 40.0,
            }
            # Return exact match or interpolate
            if param_count in size_mapping:
                return size_mapping[param_count]
            # For other sizes, estimate ~0.6GB per billion parameters (quantized)
            return param_count * 0.6
        # Default estimate
        return 4.0

    async def _ensure_memory_for_load(self, model_id: str) -> None:
        """Ensure there is enough memory to load a model.

        Evicts LRU models as needed until there is enough memory.

        Args:
            model_id: Model to load

        Raises:
            HTTPException: 503 if insufficient memory even after eviction
        """
        estimated_size = self._estimate_model_size(model_id)
        memory_limit = self._get_effective_memory_limit()

        while True:
            current = self._current_memory_gb()
            available = memory_limit - current

            if estimated_size <= available:
                # Enough memory available
                return

            # Try to evict LRU model
            evictable = self._evictable_models()
            if not evictable:
                # No models to evict
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Insufficient memory: need {estimated_size:.1f}GB, "
                        f"only {available:.1f}GB available after eviction"
                    ),
                )

            # Evict LRU
            await self._evict_lru()

    async def get_model(self, model_id: str) -> LoadedModel:
        """Get a model, loading it if necessary.

        Args:
            model_id: HuggingFace model ID (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")

        Returns:
            LoadedModel with model and tokenizer

        Raises:
            RuntimeError: If model loading fails
            HTTPException: 503 if insufficient memory
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

        # Ensure memory before loading
        async with self._lock:
            await self._ensure_memory_for_load(model_id)

        # Start loading
        return await self._load_model(model_id)

    async def _load_model(self, model_id: str) -> LoadedModel:
        """Load a model from HuggingFace.

        Detects model type and uses appropriate loader:
        - TEXT_GEN: mlx_lm.load()
        - VISION: mlx_vlm.load()
        - EMBEDDINGS: mlx_embeddings.utils.load()

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
            # Detect model type
            model_type = detect_model_type(model_id)
            logger.info(f"Detected model type: {model_type.value} for {model_id}")

            # Load based on type
            if model_type == ModelType.VISION:
                # Vision models use mlx-vlm (returns model, processor)
                from mlx_vlm import load as load_vlm  # type: ignore[import-untyped]

                result = await asyncio.to_thread(load_vlm, model_id)
                model, tokenizer = result[0], result[1]  # processor stored as tokenizer
            elif model_type == ModelType.EMBEDDINGS:
                # Embedding models use mlx-embeddings
                from mlx_embeddings.utils import (
                    load as load_embeddings,  # type: ignore[import-untyped]
                )

                result = await asyncio.to_thread(load_embeddings, model_id)
                model, tokenizer = result[0], result[1]
            else:
                # Text-gen models use mlx-lm
                from mlx_lm import load

                result = await asyncio.to_thread(load, model_id)
                model, tokenizer = result[0], result[1]

            # Get memory after loading
            from mlx_manager.mlx_server.utils.memory import get_memory_usage

            memory = get_memory_usage()

            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=tokenizer,
                model_type=model_type.value,
                size_gb=memory["active_gb"],
            )

            async with self._lock:
                self._models[model_id] = loaded
                self._loading[model_id].set()
                del self._loading[model_id]

            elapsed = time.time() - start_time
            logger.info(
                f"Model loaded: {model_id} (type={model_type.value}, "
                f"{elapsed:.1f}s, {loaded.size_gb:.1f}GB)"
            )
            return loaded

        except Exception as e:
            async with self._lock:
                if model_id in self._loading:
                    self._loading[model_id].set()
                    del self._loading[model_id]
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e

    async def preload_model(self, model_id: str) -> LoadedModel:
        """Preload a model and mark it as protected from eviction.

        Args:
            model_id: HuggingFace model ID

        Returns:
            LoadedModel with preloaded=True
        """
        loaded = await self.get_model(model_id)
        loaded.preloaded = True
        logger.info(f"Model preloaded (protected from eviction): {model_id}")
        return loaded

    # =========================================================================
    # LoRA Adapter Loading Methods
    # =========================================================================

    def _validate_adapter_path(self, adapter_path: str) -> AdapterInfo:
        """Validate an adapter path and parse its configuration.

        Args:
            adapter_path: Path to the LoRA adapter directory

        Returns:
            AdapterInfo with parsed adapter metadata

        Raises:
            ValueError: If adapter path is invalid or adapter_config.json is missing
        """
        path = Path(adapter_path)

        # Check if directory exists
        if not path.exists():
            raise ValueError(f"Adapter path does not exist: {adapter_path}")

        if not path.is_dir():
            raise ValueError(f"Adapter path is not a directory: {adapter_path}")

        # Check for adapter_config.json
        config_path = path / "adapter_config.json"
        if not config_path.exists():
            raise ValueError(
                f"adapter_config.json not found in adapter directory: {adapter_path}"
            )

        # Parse adapter_config.json
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid adapter_config.json: {e}") from e

        # Extract base_model if available
        base_model = config.get("base_model_name_or_path")

        return AdapterInfo(
            adapter_path=adapter_path,
            base_model=base_model,
            description=config.get("description"),
        )

    def _get_adapter_cache_key(self, model_id: str, adapter_path: str) -> str:
        """Generate a composite cache key for model+adapter combination.

        Args:
            model_id: Base model ID
            adapter_path: Path to LoRA adapter

        Returns:
            Composite cache key
        """
        return f"{model_id}::{adapter_path}"

    async def get_model_with_adapter(
        self, model_id: str, adapter_path: str
    ) -> LoadedModel:
        """Get a model with a LoRA adapter loaded.

        The model+adapter combination is cached separately from the base model,
        allowing the same base model to be loaded with multiple different adapters.

        Args:
            model_id: HuggingFace model ID for the base model
            adapter_path: Path to the LoRA adapter directory

        Returns:
            LoadedModel with adapter loaded and adapter_info populated

        Raises:
            ValueError: If adapter path is invalid
            RuntimeError: If model loading fails
            HTTPException: 503 if insufficient memory
        """
        # Validate adapter path first
        adapter_info = self._validate_adapter_path(adapter_path)

        # Create composite cache key
        cache_key = self._get_adapter_cache_key(model_id, adapter_path)

        async with self._lock:
            # Return cached model+adapter
            if cache_key in self._models:
                loaded = self._models[cache_key]
                loaded.touch()
                logger.debug(f"Model+adapter cache hit: {cache_key}")
                return loaded

            # Wait if already loading
            if cache_key in self._loading:
                logger.debug(f"Waiting for model+adapter load: {cache_key}")

        # Wait outside lock if loading
        if cache_key in self._loading:
            await self._loading[cache_key].wait()
            return self._models[cache_key]

        # Ensure memory before loading
        async with self._lock:
            await self._ensure_memory_for_load(model_id)

        # Load model with adapter
        return await self._load_model_with_adapter(
            model_id, adapter_path, adapter_info, cache_key
        )

    async def _load_model_with_adapter(
        self,
        model_id: str,
        adapter_path: str,
        adapter_info: AdapterInfo,
        cache_key: str,
    ) -> LoadedModel:
        """Load a model with a LoRA adapter.

        Only TEXT_GEN models support LoRA adapters currently.
        Vision and embedding models don't have LoRA support in mlx-vlm/mlx-embeddings.

        Args:
            model_id: Base model ID
            adapter_path: Path to LoRA adapter
            adapter_info: Parsed adapter metadata
            cache_key: Composite cache key for the model+adapter

        Returns:
            LoadedModel with adapter applied
        """
        async with self._lock:
            # Double-check after acquiring lock
            if cache_key in self._models:
                return self._models[cache_key]

            # Mark as loading
            self._loading[cache_key] = asyncio.Event()

        logger.info(f"Loading model with adapter: {model_id} + {adapter_path}")
        start_time = time.time()

        try:
            # Detect model type - adapters only supported for TEXT_GEN
            model_type = detect_model_type(model_id)

            if model_type != ModelType.TEXT_GEN:
                raise ValueError(
                    f"LoRA adapters only supported for text-gen models, "
                    f"got {model_type.value} for {model_id}"
                )

            # Load model with adapter using mlx-lm
            from mlx_lm import load

            result = await asyncio.to_thread(load, model_id, adapter_path=adapter_path)
            model, tokenizer = result[0], result[1]

            # Get memory after loading
            from mlx_manager.mlx_server.utils.memory import get_memory_usage

            memory = get_memory_usage()

            loaded = LoadedModel(
                model_id=cache_key,  # Use composite key as model_id for cache lookup
                model=model,
                tokenizer=tokenizer,
                model_type=model_type.value,
                size_gb=memory["active_gb"],
                adapter_path=adapter_path,
                adapter_info=adapter_info,
            )

            async with self._lock:
                self._models[cache_key] = loaded
                self._loading[cache_key].set()
                del self._loading[cache_key]

            elapsed = time.time() - start_time
            logger.info(
                f"Model+adapter loaded: {model_id} + {adapter_path} "
                f"({elapsed:.1f}s, {loaded.size_gb:.1f}GB)"
            )
            return loaded

        except Exception as e:
            async with self._lock:
                if cache_key in self._loading:
                    self._loading[cache_key].set()
                    del self._loading[cache_key]
            logger.error(f"Failed to load model with adapter {cache_key}: {e}")
            raise RuntimeError(f"Failed to load model with adapter: {e}") from e

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

    # =========================================================================
    # Dynamic Configuration Methods
    # =========================================================================

    def update_memory_limit(
        self,
        memory_gb: float | None = None,
        memory_pct: float | None = None,
    ) -> None:
        """Update memory limit at runtime.

        Args:
            memory_gb: Absolute memory limit in GB
            memory_pct: Memory limit as percentage of system memory (0.0-1.0)
        """
        if memory_pct is not None:
            self._memory_limit_pct = memory_pct
            self.max_memory_gb = self._get_effective_memory_limit()
        elif memory_gb is not None:
            self._memory_limit_pct = None
            self.max_memory_gb = memory_gb

        # Update MLX memory limit
        from mlx_manager.mlx_server.utils.memory import set_memory_limit

        set_memory_limit(self.max_memory_gb)

        logger.info(f"Memory limit updated to {self.max_memory_gb:.1f}GB")

    def update_max_models(self, max_models: int) -> None:
        """Update maximum hot models at runtime.

        Args:
            max_models: Maximum number of models to keep loaded simultaneously
        """
        self.max_models = max_models
        logger.info(f"Max models updated to {max_models}")

    async def apply_preload_list(self, model_ids: list[str]) -> dict[str, str]:
        """Preload specified models, unload others.

        Args:
            model_ids: List of model IDs to preload and protect from eviction

        Returns:
            dict of model_id -> status ('loaded', 'failed', 'already_loaded')
        """
        results: dict[str, str] = {}

        # Preload new models
        for model_id in model_ids:
            if self.is_loaded(model_id):
                self._models[model_id].preloaded = True
                results[model_id] = "already_loaded"
            else:
                try:
                    await self.preload_model(model_id)
                    results[model_id] = "loaded"
                except Exception as e:
                    logger.error(f"Failed to preload {model_id}: {e}")
                    results[model_id] = f"failed: {e}"

        # Mark models not in preload list as evictable
        for model_id, loaded in self._models.items():
            if model_id not in model_ids:
                loaded.preloaded = False

        return results

    def get_status(self) -> dict:
        """Get current pool status.

        Returns:
            dict with pool configuration and loaded models info
        """
        loaded_models = []
        for m in self._models.values():
            model_info: dict[str, Any] = {
                "model_id": m.model_id,
                "size_gb": m.size_gb,
                "preloaded": m.preloaded,
                "last_used": m.last_used,
            }
            if m.adapter_path:
                model_info["adapter_path"] = m.adapter_path
            if m.adapter_info:
                model_info["adapter_info"] = {
                    "adapter_path": m.adapter_info.adapter_path,
                    "base_model": m.adapter_info.base_model,
                    "description": m.adapter_info.description,
                }
            loaded_models.append(model_info)

        return {
            "max_memory_gb": self.max_memory_gb,
            "current_memory_gb": self._current_memory_gb(),
            "max_models": self.max_models,
            "loaded_models": loaded_models,
        }

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
