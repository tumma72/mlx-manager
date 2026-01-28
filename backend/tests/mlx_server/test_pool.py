"""Unit tests for ModelPoolManager with LRU eviction and multi-model support."""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mlx_manager.mlx_server.models.pool import LoadedModel, ModelPoolManager


class TestLoadedModel:
    """Tests for LoadedModel dataclass."""

    def test_loaded_model_fields_defaults(self) -> None:
        """Verify LoadedModel has model_type and preloaded fields with correct defaults."""
        loaded = LoadedModel(
            model_id="test-model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        assert loaded.model_id == "test-model"
        assert loaded.model_type == "text-gen"  # Default
        assert loaded.preloaded is False  # Default
        assert loaded.size_gb == 0.0  # Default
        assert loaded.loaded_at > 0
        assert loaded.last_used > 0

    def test_loaded_model_custom_fields(self) -> None:
        """Verify LoadedModel accepts custom model_type and preloaded values."""
        loaded = LoadedModel(
            model_id="test-vision",
            model=MagicMock(),
            tokenizer=MagicMock(),
            model_type="vision",
            preloaded=True,
            size_gb=8.5,
        )

        assert loaded.model_type == "vision"
        assert loaded.preloaded is True
        assert loaded.size_gb == 8.5

    def test_loaded_model_touch(self) -> None:
        """Verify touch() updates last_used timestamp."""
        loaded = LoadedModel(
            model_id="test-model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        original_last_used = loaded.last_used

        time.sleep(0.01)  # Small delay
        loaded.touch()

        assert loaded.last_used > original_last_used


class TestModelPoolManagerMemory:
    """Tests for ModelPoolManager memory limit functionality."""

    def test_pool_memory_limit_absolute(self) -> None:
        """Create pool with max_memory_gb, verify _get_effective_memory_limit() returns it."""
        pool = ModelPoolManager(max_memory_gb=24.0, max_models=4)

        assert pool._get_effective_memory_limit() == 24.0

    def test_pool_memory_limit_percentage(self) -> None:
        """Create pool with memory_limit_pct, verify calculation uses psutil."""
        # Mock psutil.virtual_memory to return fixed value (64GB)
        mock_memory = MagicMock()
        mock_memory.total = 64 * (1024**3)  # 64 GB in bytes

        with patch("psutil.virtual_memory", return_value=mock_memory):
            pool = ModelPoolManager(
                max_memory_gb=48.0,  # This should be ignored
                max_models=4,
                memory_limit_pct=0.5,  # 50% of 64GB = 32GB
            )

            limit = pool._get_effective_memory_limit()
            assert limit == 32.0

    def test_pool_current_memory_empty(self) -> None:
        """Verify _current_memory_gb() returns 0 for empty pool."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        assert pool._current_memory_gb() == 0.0

    def test_pool_current_memory_with_models(self) -> None:
        """Verify _current_memory_gb() sums loaded model sizes."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # Add models directly to simulate loaded state
        pool._models["model1"] = LoadedModel(
            model_id="model1",
            model=MagicMock(),
            tokenizer=MagicMock(),
            size_gb=4.0,
        )
        pool._models["model2"] = LoadedModel(
            model_id="model2",
            model=MagicMock(),
            tokenizer=MagicMock(),
            size_gb=8.0,
        )

        assert pool._current_memory_gb() == 12.0


class TestModelSizeEstimation:
    """Tests for model size estimation based on name patterns."""

    def test_estimate_model_size_3b(self) -> None:
        """Test estimation for 3B models."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        assert pool._estimate_model_size("mlx-community/Llama-3.2-3B-Instruct-4bit") == 2.0
        assert pool._estimate_model_size("org/model-3b-quantized") == 2.0

    def test_estimate_model_size_7b(self) -> None:
        """Test estimation for 7B models."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        assert pool._estimate_model_size("mlx-community/Mistral-7B-v0.1-4bit") == 4.0
        assert pool._estimate_model_size("model-7b") == 4.0

    def test_estimate_model_size_8b(self) -> None:
        """Test estimation for 8B models."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        assert pool._estimate_model_size("mlx-community/Llama-3.1-8B-Instruct") == 5.0

    def test_estimate_model_size_13b(self) -> None:
        """Test estimation for 13B models."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        assert pool._estimate_model_size("CodeLlama-13B") == 8.0
        assert pool._estimate_model_size("model-13b-chat") == 8.0

    def test_estimate_model_size_70b(self) -> None:
        """Test estimation for 70B models."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        assert pool._estimate_model_size("Llama-2-70B") == 40.0

    def test_estimate_model_size_interpolated(self) -> None:
        """Test estimation for non-standard sizes (interpolated)."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # 32B should interpolate: 32 * 0.6 = 19.2 GB
        assert pool._estimate_model_size("model-32B") == pytest.approx(19.2)

    def test_estimate_model_size_default(self) -> None:
        """Test default estimation when no size pattern found."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # No B/b pattern - should return default 4.0
        assert pool._estimate_model_size("mlx-community/some-model") == 4.0
        assert pool._estimate_model_size("gpt-like-model") == 4.0


class TestEvictableModels:
    """Tests for evictable model filtering."""

    def test_evictable_models_empty(self) -> None:
        """Verify _evictable_models() returns empty list for empty pool."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        assert pool._evictable_models() == []

    def test_evictable_models_mixed(self) -> None:
        """Verify _evictable_models() only returns non-preloaded models."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # Add preloaded model
        pool._models["preloaded"] = LoadedModel(
            model_id="preloaded",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=True,
        )
        # Add non-preloaded models
        pool._models["regular1"] = LoadedModel(
            model_id="regular1",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=False,
        )
        pool._models["regular2"] = LoadedModel(
            model_id="regular2",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=False,
        )

        evictable = pool._evictable_models()

        assert len(evictable) == 2
        model_ids = [m.model_id for m in evictable]
        assert "regular1" in model_ids
        assert "regular2" in model_ids
        assert "preloaded" not in model_ids

    def test_evictable_models_all_preloaded(self) -> None:
        """Verify _evictable_models() returns empty when all are preloaded."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        pool._models["preloaded1"] = LoadedModel(
            model_id="preloaded1",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=True,
        )
        pool._models["preloaded2"] = LoadedModel(
            model_id="preloaded2",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=True,
        )

        assert pool._evictable_models() == []


class TestLRUEviction:
    """Tests for LRU eviction logic."""

    @pytest.mark.asyncio
    async def test_evict_lru_removes_oldest(self) -> None:
        """Verify _evict_lru() removes the oldest non-preloaded model."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # Add models with different last_used times
        pool._models["newest"] = LoadedModel(
            model_id="newest",
            model=MagicMock(),
            tokenizer=MagicMock(),
            last_used=time.time(),
        )
        pool._models["oldest"] = LoadedModel(
            model_id="oldest",
            model=MagicMock(),
            tokenizer=MagicMock(),
            last_used=time.time() - 100,  # 100 seconds ago
        )
        pool._models["middle"] = LoadedModel(
            model_id="middle",
            model=MagicMock(),
            tokenizer=MagicMock(),
            last_used=time.time() - 50,  # 50 seconds ago
        )

        with patch("mlx_manager.mlx_server.utils.memory.clear_cache"):
            evicted = await pool._evict_lru()

        assert evicted is True
        assert "oldest" not in pool._models
        assert "newest" in pool._models
        assert "middle" in pool._models

    @pytest.mark.asyncio
    async def test_evict_lru_no_evictable(self) -> None:
        """Verify _evict_lru() returns False when no evictable models."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # Only preloaded models
        pool._models["preloaded"] = LoadedModel(
            model_id="preloaded",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=True,
        )

        evicted = await pool._evict_lru()

        assert evicted is False
        assert "preloaded" in pool._models

    @pytest.mark.asyncio
    async def test_preload_protection_survives_eviction(self) -> None:
        """Verify preloaded model survives eviction when other models are evictable."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # Add preloaded model (oldest)
        pool._models["protected"] = LoadedModel(
            model_id="protected",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=True,
            last_used=time.time() - 200,  # Oldest
        )
        # Add non-preloaded model (newer)
        pool._models["expendable"] = LoadedModel(
            model_id="expendable",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=False,
            last_used=time.time() - 50,  # Newer than protected
        )

        with patch("mlx_manager.mlx_server.utils.memory.clear_cache"):
            evicted = await pool._evict_lru()

        assert evicted is True
        # Protected model survives even though it's oldest
        assert "protected" in pool._models
        # Expendable model was evicted
        assert "expendable" not in pool._models


class TestInsufficientMemory:
    """Tests for insufficient memory error handling."""

    @pytest.mark.asyncio
    async def test_insufficient_memory_error(self) -> None:
        """Set up pool with small limit, verify 503 HTTPException raised."""
        # Create pool with very small memory limit
        pool = ModelPoolManager(max_memory_gb=4.0, max_models=4)

        # Add a preloaded model that takes all the memory
        pool._models["preloaded"] = LoadedModel(
            model_id="preloaded",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=True,
            size_gb=3.5,  # Uses most of 4GB limit
        )

        # Try to ensure memory for a model that needs more than available
        # Even after evicting all evictable models (none), not enough space
        with pytest.raises(HTTPException) as exc_info:
            await pool._ensure_memory_for_load("some-70B-model")  # Needs ~40GB

        assert exc_info.value.status_code == 503
        assert "Insufficient memory" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_eviction_frees_enough_memory(self) -> None:
        """Verify eviction loop frees enough memory for new model."""
        pool = ModelPoolManager(max_memory_gb=12.0, max_models=4)

        # Add evictable models totaling 8GB
        pool._models["model1"] = LoadedModel(
            model_id="model1",
            model=MagicMock(),
            tokenizer=MagicMock(),
            size_gb=4.0,
            last_used=time.time() - 100,  # Oldest
        )
        pool._models["model2"] = LoadedModel(
            model_id="model2",
            model=MagicMock(),
            tokenizer=MagicMock(),
            size_gb=4.0,
            last_used=time.time() - 50,
        )

        with patch("mlx_manager.mlx_server.utils.memory.clear_cache"):
            # Try to ensure memory for 8GB model (needs eviction)
            # Current: 8GB, Limit: 12GB, Needed: 8GB, Available: 4GB
            # Should evict oldest (model1) -> Available: 8GB
            await pool._ensure_memory_for_load("some-8B-model")

        # model1 should be evicted
        assert "model1" not in pool._models
        # model2 should remain
        assert "model2" in pool._models


class TestPoolIntegration:
    """Integration tests for pool functionality."""

    @pytest.mark.asyncio
    async def test_preload_model_marks_protected(self) -> None:
        """Verify preload_model() loads and marks as protected."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0, "peak_gb": 4.0, "cache_gb": 0.0},
            ),
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
        ):
            loaded = await pool.preload_model("test-3B-model")

        assert loaded.preloaded is True
        assert loaded.model_id == "test-3B-model"
        assert pool.is_loaded("test-3B-model")

    @pytest.mark.asyncio
    async def test_get_model_triggers_memory_check(self) -> None:
        """Verify get_model() checks memory before loading."""
        pool = ModelPoolManager(max_memory_gb=8.0, max_models=4)

        # Add model that uses most memory
        pool._models["existing"] = LoadedModel(
            model_id="existing",
            model=MagicMock(),
            tokenizer=MagicMock(),
            size_gb=6.0,
            last_used=time.time() - 100,
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)),
            patch(
                "mlx_manager.mlx_server.utils.memory.get_memory_usage",
                return_value={"active_gb": 4.0, "peak_gb": 4.0, "cache_gb": 0.0},
            ),
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
        ):
            # Request new 7B model (needs ~4GB)
            # Current: 6GB, Limit: 8GB, Available: 2GB
            # Should evict existing -> Available: 8GB
            loaded = await pool.get_model("new-7B-model")

        # Existing should be evicted
        assert "existing" not in pool._models
        # New model loaded
        assert loaded.model_id == "new-7B-model"
        assert pool.is_loaded("new-7B-model")
