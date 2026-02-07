"""Unit tests for ModelPoolManager with LRU eviction and multi-model support."""

import asyncio
import json
import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mlx_manager.mlx_server.models.pool import (
    LoadedModel,
    ModelPoolManager,
    get_model_pool,
)
from mlx_manager.mlx_server.models.types import AdapterInfo, ModelType

# ============================================================================
# Common mock fixtures
# ============================================================================

MOCK_MEMORY = {"active_gb": 4.0, "peak_gb": 4.0, "cache_gb": 0.0}


@pytest.fixture
def pool() -> ModelPoolManager:
    """Create a fresh ModelPoolManager for each test."""
    return ModelPoolManager(max_memory_gb=48.0, max_models=4)


@pytest.fixture
def mock_detect():
    """Mock detect_model_type to avoid HuggingFace cache access."""
    with patch("mlx_manager.mlx_server.models.pool.detect_model_type") as m:
        m.return_value = ModelType.TEXT_GEN
        yield m


@pytest.fixture
def mock_memory():
    """Mock get_memory_usage to return stable values."""
    with patch(
        "mlx_manager.mlx_server.utils.memory.get_memory_usage",
        return_value=MOCK_MEMORY,
    ) as m:
        yield m


@pytest.fixture
def mock_clear_cache():
    """Mock clear_cache to avoid MLX Metal calls."""
    with patch("mlx_manager.mlx_server.utils.memory.clear_cache") as m:
        yield m


@pytest.fixture
def mock_set_memory_limit():
    """Mock set_memory_limit to avoid MLX Metal calls."""
    with patch("mlx_manager.mlx_server.utils.memory.set_memory_limit") as m:
        yield m


@pytest.fixture
def mock_mlx_lm_load():
    """Mock mlx_lm.load for text-gen model loading."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
        with patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)) as m:
            yield m


# ============================================================================
# TestLoadedModel (existing)
# ============================================================================


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

    def test_loaded_model_adapter_fields(self) -> None:
        """Verify LoadedModel accepts adapter_path and adapter_info."""
        info = AdapterInfo(adapter_path="/tmp/adapter", base_model="base/model")
        loaded = LoadedModel(
            model_id="test-model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            adapter_path="/tmp/adapter",
            adapter_info=info,
        )

        assert loaded.adapter_path == "/tmp/adapter"
        assert loaded.adapter_info is not None
        assert loaded.adapter_info.base_model == "base/model"


# ============================================================================
# TestModelPoolManagerMemory (existing)
# ============================================================================


class TestModelPoolManagerMemory:
    """Tests for ModelPoolManager memory limit functionality."""

    def test_pool_memory_limit_absolute(self) -> None:
        """Create pool with max_memory_gb, verify _get_effective_memory_limit() returns it."""
        pool = ModelPoolManager(max_memory_gb=24.0, max_models=4)

        assert pool._get_effective_memory_limit() == 24.0

    def test_pool_memory_limit_percentage(self) -> None:
        """Create pool with memory_limit_pct, verify calculation uses device memory."""
        with patch(
            "mlx_manager.mlx_server.utils.memory.get_device_memory_gb",
            return_value=64.0,
        ):
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


# ============================================================================
# TestModelSizeEstimation (existing)
# ============================================================================


class TestModelSizeEstimation:
    """Tests for model size estimation based on name patterns."""

    @pytest.fixture(autouse=True)
    def _mock_hf_cache(self) -> Generator:
        """Mock HF cache to nonexistent path so all tests use name-pattern fallback."""
        with patch("mlx_manager.config.settings") as mock_settings:
            mock_settings.hf_cache_path = Path("/nonexistent/cache")
            yield

    def test_estimate_model_size_3b(self) -> None:
        """Test estimation for 3B models (name-pattern fallback)."""
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


# ============================================================================
# TestEvictableModels (existing)
# ============================================================================


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


# ============================================================================
# TestLRUEviction (existing)
# ============================================================================


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


# ============================================================================
# TestInsufficientMemory (existing)
# ============================================================================


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


# ============================================================================
# TestPoolIntegration (existing, fixed with detect_model_type mock)
# ============================================================================


class TestPoolIntegration:
    """Integration tests for pool functionality."""

    @pytest.mark.asyncio
    async def test_preload_model_marks_protected(
        self, mock_detect, mock_memory, mock_mlx_lm_load
    ) -> None:
        """Verify preload_model() loads and marks as protected."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        loaded = await pool.preload_model("test-3B-model")

        assert loaded.preloaded is True
        assert loaded.model_id == "test-3B-model"
        assert pool.is_loaded("test-3B-model")

    @pytest.mark.asyncio
    async def test_get_model_triggers_memory_check(
        self, mock_detect, mock_memory, mock_mlx_lm_load, mock_clear_cache
    ) -> None:
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

        # Request new 7B model (needs ~4GB)
        # Current: 6GB, Limit: 8GB, Available: 2GB
        # Should evict existing -> Available: 8GB
        loaded = await pool.get_model("new-7B-model")

        # Existing should be evicted
        assert "existing" not in pool._models
        # New model loaded
        assert loaded.model_id == "new-7B-model"
        assert pool.is_loaded("new-7B-model")


# ============================================================================
# Group A: Model loading by type
# ============================================================================


class TestModelLoadingByType:
    """Tests for loading different model types with their respective loaders."""

    @pytest.mark.asyncio
    async def test_load_text_gen_model(self, mock_detect, mock_memory) -> None:
        """Verify TEXT_GEN model uses mlx_lm.load."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.TEXT_GEN

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)):
            loaded = await pool.get_model("test/text-model")

        assert loaded.model_type == "text-gen"
        assert loaded.model is mock_model
        assert loaded.tokenizer is mock_tokenizer

    @pytest.mark.asyncio
    async def test_load_vision_model(self, mock_detect, mock_memory) -> None:
        """Verify VISION model uses mlx_vlm.load."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.VISION

        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch("asyncio.to_thread", return_value=(mock_model, mock_processor)):
            loaded = await pool.get_model("test/vision-model")

        assert loaded.model_type == "vision"
        assert loaded.model is mock_model
        assert loaded.tokenizer is mock_processor  # processor stored as tokenizer

    @pytest.mark.asyncio
    async def test_load_embeddings_model(self, mock_detect, mock_memory) -> None:
        """Verify EMBEDDINGS model uses mlx_embeddings.utils.load."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.EMBEDDINGS

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)):
            loaded = await pool.get_model("test/embed-model")

        assert loaded.model_type == "embeddings"
        assert loaded.model is mock_model
        assert loaded.tokenizer is mock_tokenizer

    @pytest.mark.asyncio
    async def test_load_audio_model(self, mock_detect, mock_memory) -> None:
        """Verify AUDIO model uses mlx_audio.utils.load_model with tokenizer=None."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.AUDIO

        mock_model = MagicMock()

        # Audio load returns only the model (not a tuple)
        with patch("asyncio.to_thread", return_value=mock_model):
            loaded = await pool.get_model("test/audio-model")

        assert loaded.model_type == "audio"
        assert loaded.model is mock_model
        assert loaded.tokenizer is None  # Audio models have no tokenizer


# ============================================================================
# Group B: LoRA adapter support
# ============================================================================


class TestLoRAAdapterValidation:
    """Tests for _validate_adapter_path()."""

    def test_validate_adapter_nonexistent_path(self, pool: ModelPoolManager) -> None:
        """Raise ValueError if adapter path does not exist."""
        with pytest.raises(ValueError, match="does not exist"):
            pool._validate_adapter_path("/nonexistent/path/adapter")

    def test_validate_adapter_not_directory(
        self, pool: ModelPoolManager, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Raise ValueError if adapter path is a file, not a directory."""
        file_path = tmp_path / "adapter_file.txt"
        file_path.write_text("not a directory")

        with pytest.raises(ValueError, match="not a directory"):
            pool._validate_adapter_path(str(file_path))

    def test_validate_adapter_missing_config(
        self, pool: ModelPoolManager, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Raise ValueError if adapter_config.json is missing."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        with pytest.raises(ValueError, match="adapter_config.json not found"):
            pool._validate_adapter_path(str(adapter_dir))

    def test_validate_adapter_invalid_json(
        self, pool: ModelPoolManager, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Raise ValueError if adapter_config.json is invalid JSON."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config_file = adapter_dir / "adapter_config.json"
        config_file.write_text("not valid json {{{")

        with pytest.raises(ValueError, match="Invalid adapter_config.json"):
            pool._validate_adapter_path(str(adapter_dir))

    def test_validate_adapter_success(
        self, pool: ModelPoolManager, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Successfully validate adapter path and return AdapterInfo."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {
            "base_model_name_or_path": "base/model-id",
            "description": "Test adapter",
        }
        config_file = adapter_dir / "adapter_config.json"
        config_file.write_text(json.dumps(config))

        info = pool._validate_adapter_path(str(adapter_dir))

        assert isinstance(info, AdapterInfo)
        assert info.adapter_path == str(adapter_dir)
        assert info.base_model == "base/model-id"
        assert info.description == "Test adapter"

    def test_validate_adapter_no_base_model(
        self, pool: ModelPoolManager, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Adapter config without base_model_name_or_path returns None for base_model."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"lora_alpha": 16}
        config_file = adapter_dir / "adapter_config.json"
        config_file.write_text(json.dumps(config))

        info = pool._validate_adapter_path(str(adapter_dir))

        assert info.base_model is None
        assert info.description is None


class TestAdapterCacheKey:
    """Tests for _get_adapter_cache_key()."""

    def test_cache_key_format(self, pool: ModelPoolManager) -> None:
        """Verify composite cache key uses '::' separator."""
        key = pool._get_adapter_cache_key("model/id", "/path/to/adapter")
        assert key == "model/id::/path/to/adapter"


class TestModelWithAdapter:
    """Tests for get_model_with_adapter() and _load_model_with_adapter()."""

    @pytest.mark.asyncio
    async def test_load_model_with_adapter(
        self, mock_detect, mock_memory, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Load a text-gen model with LoRA adapter successfully."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.TEXT_GEN

        # Create valid adapter directory
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"base_model_name_or_path": "base/model"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)):
            loaded = await pool.get_model_with_adapter("test/model", str(adapter_dir))

        expected_key = f"test/model::{adapter_dir}"
        assert loaded.model_id == expected_key
        assert loaded.adapter_path == str(adapter_dir)
        assert loaded.adapter_info is not None
        assert loaded.adapter_info.base_model == "base/model"
        assert loaded.model_type == "text-gen"

    @pytest.mark.asyncio
    async def test_adapter_cache_hit(
        self, pool: ModelPoolManager, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Return cached model+adapter without reloading."""
        # Create valid adapter directory
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"base_model_name_or_path": "base/model"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        cache_key = f"test/model::{adapter_dir}"
        original_time = time.time() - 100

        pool._models[cache_key] = LoadedModel(
            model_id=cache_key,
            model=MagicMock(),
            tokenizer=MagicMock(),
            adapter_path=str(adapter_dir),
            last_used=original_time,
        )

        loaded = await pool.get_model_with_adapter("test/model", str(adapter_dir))

        assert loaded.model_id == cache_key
        assert loaded.last_used > original_time  # touch() was called

    @pytest.mark.asyncio
    async def test_adapter_only_text_gen(
        self, mock_detect, mock_memory, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Raise RuntimeError when applying adapter to non-TEXT_GEN model."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.VISION

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"base_model_name_or_path": "base/model"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        with pytest.raises(RuntimeError, match="Failed to load model with adapter"):
            await pool.get_model_with_adapter("test/vision-model", str(adapter_dir))

    @pytest.mark.asyncio
    async def test_adapter_load_failure_cleans_up_events(
        self, mock_detect, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Verify loading event is cleaned up when adapter load fails."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.TEXT_GEN

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"base_model_name_or_path": "base/model"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        with patch("asyncio.to_thread", side_effect=Exception("Load failed")):
            with pytest.raises(RuntimeError, match="Failed to load model with adapter"):
                await pool.get_model_with_adapter("test/model", str(adapter_dir))

        # Loading event should be cleaned up
        cache_key = f"test/model::{adapter_dir}"
        assert cache_key not in pool._loading


# ============================================================================
# Group C: Concurrent loading
# ============================================================================


class TestConcurrentLoading:
    """Tests for concurrent model loading with asyncio.Event synchronization."""

    @pytest.mark.asyncio
    async def test_concurrent_get_model_same_id(self, mock_detect, mock_memory) -> None:
        """Two concurrent get_model() calls for same model should load only once."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.TEXT_GEN

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        load_count = 0

        async def slow_to_thread(func, *args, **kwargs):
            nonlocal load_count
            load_count += 1
            # Simulate a slow load
            await asyncio.sleep(0.05)
            return (mock_model, mock_tokenizer)

        with patch("asyncio.to_thread", side_effect=slow_to_thread):
            # Launch two concurrent loads for the same model
            results = await asyncio.gather(
                pool.get_model("test/model"),
                pool.get_model("test/model"),
            )

        # Both should return valid LoadedModel
        assert results[0].model_id == "test/model"
        assert results[1].model_id == "test/model"
        # Model should only be loaded once
        assert load_count == 1


# ============================================================================
# Group D: Cache hits
# ============================================================================


class TestCacheHits:
    """Tests for returning cached models and updating last_used."""

    @pytest.mark.asyncio
    async def test_get_model_cache_hit(self) -> None:
        """Return already-loaded model from cache and update last_used."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        original_time = time.time() - 100
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        pool._models["cached-model"] = LoadedModel(
            model_id="cached-model",
            model=mock_model,
            tokenizer=mock_tokenizer,
            last_used=original_time,
            size_gb=4.0,
        )

        loaded = await pool.get_model("cached-model")

        assert loaded.model is mock_model
        assert loaded.tokenizer is mock_tokenizer
        assert loaded.last_used > original_time  # touch() was called

    @pytest.mark.asyncio
    async def test_get_model_cache_hit_does_not_reload(self) -> None:
        """Cached model is not reloaded through the loading path."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        pool._models["cached-model"] = LoadedModel(
            model_id="cached-model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        with patch("asyncio.to_thread") as mock_thread:
            await pool.get_model("cached-model")
            # to_thread should never be called for a cached model
            mock_thread.assert_not_called()


# ============================================================================
# Group E: Reload as type
# ============================================================================


class TestReloadAsType:
    """Tests for reload_as_type() and _load_model_as_type()."""

    @pytest.mark.asyncio
    async def test_reload_as_vision(self, mock_memory, mock_clear_cache) -> None:
        """Pre-load as TEXT_GEN, reload as VISION, verify new type."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # Pre-populate pool with a text-gen model
        pool._models["test/model"] = LoadedModel(
            model_id="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            model_type="text-gen",
        )

        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch("asyncio.to_thread", return_value=(mock_model, mock_processor)):
            loaded = await pool.reload_as_type("test/model", ModelType.VISION)

        assert loaded.model_type == "vision"
        assert loaded.model is mock_model
        assert loaded.tokenizer is mock_processor

    @pytest.mark.asyncio
    async def test_reload_as_embeddings(self, mock_memory, mock_clear_cache) -> None:
        """Reload model as EMBEDDINGS type."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        pool._models["test/model"] = LoadedModel(
            model_id="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            model_type="text-gen",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)):
            loaded = await pool.reload_as_type("test/model", ModelType.EMBEDDINGS)

        assert loaded.model_type == "embeddings"

    @pytest.mark.asyncio
    async def test_reload_as_audio(self, mock_memory, mock_clear_cache) -> None:
        """Reload model as AUDIO type, tokenizer should be None."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        pool._models["test/model"] = LoadedModel(
            model_id="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            model_type="text-gen",
        )

        mock_model = MagicMock()

        with patch("asyncio.to_thread", return_value=mock_model):
            loaded = await pool.reload_as_type("test/model", ModelType.AUDIO)

        assert loaded.model_type == "audio"
        assert loaded.tokenizer is None

    @pytest.mark.asyncio
    async def test_reload_as_text_gen(self, mock_memory, mock_clear_cache) -> None:
        """Reload model as TEXT_GEN type (the else branch in _load_model_as_type)."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        pool._models["test/model"] = LoadedModel(
            model_id="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            model_type="vision",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)):
            loaded = await pool.reload_as_type("test/model", ModelType.TEXT_GEN)

        assert loaded.model_type == "text-gen"

    @pytest.mark.asyncio
    async def test_reload_not_loaded(self, mock_memory) -> None:
        """Reload a model that is not currently loaded (no unload needed)."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("asyncio.to_thread", return_value=(mock_model, mock_tokenizer)),
            patch("mlx_manager.mlx_server.utils.memory.clear_cache"),
        ):
            loaded = await pool.reload_as_type("test/new-model", ModelType.TEXT_GEN)

        assert loaded.model_type == "text-gen"

    @pytest.mark.asyncio
    async def test_load_model_as_type_already_loaded(self, mock_memory) -> None:
        """If model appears in pool during _load_model_as_type, return it immediately."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        existing = LoadedModel(
            model_id="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            model_type="vision",
        )
        pool._models["test/model"] = existing

        # _load_model_as_type should find it in the double-check and return it
        result = await pool._load_model_as_type("test/model", ModelType.VISION)

        assert result is existing

    @pytest.mark.asyncio
    async def test_load_model_as_type_failure(self, mock_memory) -> None:
        """Verify loading event cleanup on _load_model_as_type failure."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        with patch("asyncio.to_thread", side_effect=Exception("Load error")):
            with pytest.raises(RuntimeError, match="Failed to load model as vision"):
                await pool._load_model_as_type("test/model", ModelType.VISION)

        # Loading event should be cleaned up
        assert "test/model" not in pool._loading


# ============================================================================
# Group F: Dynamic configuration
# ============================================================================


class TestDynamicConfig:
    """Tests for update_memory_limit() and update_max_models()."""

    def test_update_memory_limit_absolute(
        self, pool: ModelPoolManager, mock_set_memory_limit
    ) -> None:
        """Update memory limit with absolute GB value."""
        pool.update_memory_limit(memory_gb=32.0)

        assert pool.max_memory_gb == 32.0
        assert pool._memory_limit_pct is None
        mock_set_memory_limit.assert_called_once_with(32.0)

    def test_update_memory_limit_percentage(
        self, pool: ModelPoolManager, mock_set_memory_limit
    ) -> None:
        """Update memory limit with percentage of device memory."""
        with patch(
            "mlx_manager.mlx_server.utils.memory.get_device_memory_gb",
            return_value=64.0,
        ):
            pool.update_memory_limit(memory_pct=0.75)

        assert pool._memory_limit_pct == 0.75
        assert pool.max_memory_gb == pytest.approx(48.0)
        mock_set_memory_limit.assert_called_once()

    def test_update_memory_limit_no_args(
        self, pool: ModelPoolManager, mock_set_memory_limit
    ) -> None:
        """Calling update_memory_limit with no args keeps the same limit but still syncs to MLX."""
        original = pool.max_memory_gb
        pool.update_memory_limit()

        assert pool.max_memory_gb == original
        # set_memory_limit is still called unconditionally (syncs current limit to MLX)
        mock_set_memory_limit.assert_called_once_with(original)

    def test_update_max_models(self, pool: ModelPoolManager) -> None:
        """Update max_models at runtime."""
        pool.update_max_models(8)

        assert pool.max_models == 8

    def test_update_max_models_to_one(self, pool: ModelPoolManager) -> None:
        """Set max_models to 1."""
        pool.update_max_models(1)

        assert pool.max_models == 1


# ============================================================================
# Group G: Preload management
# ============================================================================


class TestPreloadManagement:
    """Tests for apply_preload_list()."""

    @pytest.mark.asyncio
    async def test_apply_preload_list_new_models(
        self, mock_detect, mock_memory, mock_mlx_lm_load
    ) -> None:
        """Preload new models and mark them as protected."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        results = await pool.apply_preload_list(["model/a", "model/b"])

        assert results["model/a"] == "loaded"
        assert results["model/b"] == "loaded"
        assert pool._models["model/a"].preloaded is True
        assert pool._models["model/b"].preloaded is True

    @pytest.mark.asyncio
    async def test_apply_preload_list_already_loaded(self) -> None:
        """Already loaded models get marked as preloaded without reloading."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        pool._models["model/a"] = LoadedModel(
            model_id="model/a",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=False,
        )

        results = await pool.apply_preload_list(["model/a"])

        assert results["model/a"] == "already_loaded"
        assert pool._models["model/a"].preloaded is True

    @pytest.mark.asyncio
    async def test_apply_preload_list_unmarks_others(
        self, mock_detect, mock_memory, mock_mlx_lm_load
    ) -> None:
        """Models NOT in the preload list get marked as evictable."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # Pre-populate a model marked as preloaded
        pool._models["old-model"] = LoadedModel(
            model_id="old-model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=True,
        )

        results = await pool.apply_preload_list(["new-model"])

        assert results["new-model"] == "loaded"
        # old-model should be unmarked from preloaded
        assert pool._models["old-model"].preloaded is False

    @pytest.mark.asyncio
    async def test_apply_preload_list_failure(self, mock_detect, mock_memory) -> None:
        """Failed preloads are recorded with error message."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        with patch("asyncio.to_thread", side_effect=Exception("Network error")):
            results = await pool.apply_preload_list(["bad-model"])

        assert "failed" in results["bad-model"]
        assert "bad-model" not in pool._models


# ============================================================================
# Group H: Status and cleanup
# ============================================================================


class TestStatusAndCleanup:
    """Tests for get_status() and cleanup()."""

    def test_get_status_empty_pool(self, pool: ModelPoolManager) -> None:
        """Verify get_status() returns correct dict for empty pool."""
        status = pool.get_status()

        assert status["max_memory_gb"] == 48.0
        assert status["current_memory_gb"] == 0.0
        assert status["max_models"] == 4
        assert status["loaded_models"] == []

    def test_get_status_with_models(self, pool: ModelPoolManager) -> None:
        """Verify get_status() includes loaded model info."""
        pool._models["model1"] = LoadedModel(
            model_id="model1",
            model=MagicMock(),
            tokenizer=MagicMock(),
            size_gb=4.0,
            preloaded=True,
        )

        status = pool.get_status()

        assert status["current_memory_gb"] == 4.0
        assert len(status["loaded_models"]) == 1
        assert status["loaded_models"][0]["model_id"] == "model1"
        assert status["loaded_models"][0]["size_gb"] == 4.0
        assert status["loaded_models"][0]["preloaded"] is True
        assert "last_used" in status["loaded_models"][0]

    def test_get_status_with_adapter_model(self, pool: ModelPoolManager) -> None:
        """Verify get_status() includes adapter info for adapted models."""
        adapter_info = AdapterInfo(
            adapter_path="/path/to/adapter",
            base_model="base/model",
            description="My adapter",
        )
        pool._models["model::adapter"] = LoadedModel(
            model_id="model::adapter",
            model=MagicMock(),
            tokenizer=MagicMock(),
            adapter_path="/path/to/adapter",
            adapter_info=adapter_info,
        )

        status = pool.get_status()

        model_info = status["loaded_models"][0]
        assert model_info["adapter_path"] == "/path/to/adapter"
        assert model_info["adapter_info"]["adapter_path"] == "/path/to/adapter"
        assert model_info["adapter_info"]["base_model"] == "base/model"
        assert model_info["adapter_info"]["description"] == "My adapter"

    def test_get_status_model_without_adapter_info(self, pool: ModelPoolManager) -> None:
        """Model with adapter_path but no adapter_info includes only adapter_path."""
        pool._models["model1"] = LoadedModel(
            model_id="model1",
            model=MagicMock(),
            tokenizer=MagicMock(),
            adapter_path="/some/path",
            adapter_info=None,
        )

        status = pool.get_status()

        model_info = status["loaded_models"][0]
        assert model_info["adapter_path"] == "/some/path"
        assert "adapter_info" not in model_info

    @pytest.mark.asyncio
    async def test_cleanup_unloads_all(self, mock_clear_cache) -> None:
        """Verify cleanup() unloads all models."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        pool._models["model1"] = LoadedModel(
            model_id="model1",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        pool._models["model2"] = LoadedModel(
            model_id="model2",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        await pool.cleanup()

        assert len(pool._models) == 0
        assert mock_clear_cache.call_count == 2  # once per model

    @pytest.mark.asyncio
    async def test_cleanup_empty_pool(self) -> None:
        """Cleanup on empty pool is a no-op."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        await pool.cleanup()

        assert len(pool._models) == 0


# ============================================================================
# Group I: Error handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling during model loading."""

    @pytest.mark.asyncio
    async def test_load_model_failure_cleans_up_event(self, mock_detect) -> None:
        """Verify loading event is cleaned up when _load_model fails."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.TEXT_GEN

        with patch("asyncio.to_thread", side_effect=Exception("CUDA error")):
            with pytest.raises(RuntimeError, match="Failed to load model"):
                await pool.get_model("bad-model")

        # Loading event should be cleaned up
        assert "bad-model" not in pool._loading

    @pytest.mark.asyncio
    async def test_load_model_failure_propagates_original_error(
        self, mock_detect
    ) -> None:
        """Verify the original exception is chained."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.TEXT_GEN

        with patch("asyncio.to_thread", side_effect=ValueError("Bad model format")):
            with pytest.raises(RuntimeError) as exc_info:
                await pool.get_model("bad-model")

        assert exc_info.value.__cause__ is not None
        assert "Bad model format" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_load_model_double_check_returns_cached(
        self, mock_detect, mock_memory
    ) -> None:
        """If model appears in pool during _load_model lock acquisition, return it."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        # Pre-populate pool so the double-check in _load_model finds it
        existing = LoadedModel(
            model_id="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        pool._models["test/model"] = existing

        # _load_model should find it in the double-check and return
        result = await pool._load_model("test/model")

        assert result is existing


# ============================================================================
# Unload and basic operations
# ============================================================================


class TestUnloadAndBasicOps:
    """Tests for unload_model, get_loaded_models, is_loaded."""

    @pytest.mark.asyncio
    async def test_unload_model_success(self, mock_clear_cache) -> None:
        """Successfully unload a loaded model."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        pool._models["test-model"] = LoadedModel(
            model_id="test-model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )

        result = await pool.unload_model("test-model")

        assert result is True
        assert "test-model" not in pool._models
        mock_clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload_model_not_found(self) -> None:
        """Return False when trying to unload a model not in pool."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        result = await pool.unload_model("nonexistent")

        assert result is False

    def test_get_loaded_models(self) -> None:
        """Return list of loaded model IDs."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        pool._models["model-a"] = LoadedModel(
            model_id="model-a", model=MagicMock(), tokenizer=MagicMock()
        )
        pool._models["model-b"] = LoadedModel(
            model_id="model-b", model=MagicMock(), tokenizer=MagicMock()
        )

        loaded = pool.get_loaded_models()

        assert set(loaded) == {"model-a", "model-b"}

    def test_is_loaded_true(self) -> None:
        """Return True for loaded model."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        pool._models["test"] = LoadedModel(
            model_id="test", model=MagicMock(), tokenizer=MagicMock()
        )

        assert pool.is_loaded("test") is True

    def test_is_loaded_false(self) -> None:
        """Return False for not loaded model."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        assert pool.is_loaded("nonexistent") is False


# ============================================================================
# Singleton accessor
# ============================================================================


class TestGetModelPool:
    """Tests for the get_model_pool() singleton accessor."""

    def test_get_model_pool_raises_when_none(self) -> None:
        """Raise RuntimeError when model_pool is None."""
        with patch("mlx_manager.mlx_server.models.pool.model_pool", None):
            with pytest.raises(RuntimeError, match="Model pool not initialized"):
                get_model_pool()

    def test_get_model_pool_returns_instance(self) -> None:
        """Return the singleton when initialized."""
        mock_pool = ModelPoolManager(max_memory_gb=16.0)
        with patch("mlx_manager.mlx_server.models.pool.model_pool", mock_pool):
            result = get_model_pool()
            assert result is mock_pool


# ============================================================================
# Preload model (covers preload_model method)
# ============================================================================


class TestPreloadModel:
    """Tests for preload_model()."""

    @pytest.mark.asyncio
    async def test_preload_sets_protected_flag(
        self, mock_detect, mock_memory, mock_mlx_lm_load
    ) -> None:
        """Verify preload_model loads and sets preloaded=True."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        loaded = await pool.preload_model("test/preload-model")

        assert loaded.preloaded is True
        assert loaded.model_id == "test/preload-model"
        assert pool.is_loaded("test/preload-model")

    @pytest.mark.asyncio
    async def test_preload_already_cached_sets_protected(self) -> None:
        """Preloading a cached model sets preloaded=True without reloading."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        pool._models["cached"] = LoadedModel(
            model_id="cached",
            model=MagicMock(),
            tokenizer=MagicMock(),
            preloaded=False,
        )

        loaded = await pool.preload_model("cached")

        assert loaded.preloaded is True


# ============================================================================
# Adapter loading wait path (concurrent adapter loads)
# ============================================================================


class TestAdapterConcurrentLoading:
    """Tests for concurrent adapter loading wait paths."""

    @pytest.mark.asyncio
    async def test_concurrent_adapter_loads(
        self, mock_detect, mock_memory, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """Two concurrent loads of same model+adapter should load once."""
        pool = ModelPoolManager(max_memory_gb=48.0)
        mock_detect.return_value = ModelType.TEXT_GEN

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        config = {"base_model_name_or_path": "base/model"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        load_count = 0

        async def slow_to_thread(func, *args, **kwargs):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.05)
            return (mock_model, mock_tokenizer)

        with patch("asyncio.to_thread", side_effect=slow_to_thread):
            results = await asyncio.gather(
                pool.get_model_with_adapter("test/model", str(adapter_dir)),
                pool.get_model_with_adapter("test/model", str(adapter_dir)),
            )

        cache_key = f"test/model::{adapter_dir}"
        assert results[0].model_id == cache_key
        assert results[1].model_id == cache_key
        assert load_count == 1


# ============================================================================
# _load_model_with_adapter double-check path
# ============================================================================


class TestAdapterDoubleCheck:
    """Tests for _load_model_with_adapter double-check path."""

    @pytest.mark.asyncio
    async def test_adapter_double_check_returns_cached(self) -> None:
        """If model+adapter appears in pool during lock, return immediately."""
        pool = ModelPoolManager(max_memory_gb=48.0)

        cache_key = "test/model::/path/adapter"
        adapter_info = AdapterInfo(adapter_path="/path/adapter")
        existing = LoadedModel(
            model_id=cache_key,
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        pool._models[cache_key] = existing

        result = await pool._load_model_with_adapter(
            "test/model", "/path/adapter", adapter_info, cache_key
        )

        assert result is existing
