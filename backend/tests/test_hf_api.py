"""Tests for HuggingFace REST API wrapper."""

import pytest

from mlx_manager.services.hf_api import (
    ModelInfo,
    estimate_size_from_name,
    get_model_size_gb,
)


class TestEstimateSizeFromName:
    """Tests for name-based size estimation (returns GiB)."""

    def test_4bit_model(self):
        """4-bit quantized models: ~0.5 bytes per param."""
        # 8B params at 4-bit = 8 * 0.5 * 1.1 * 1e9 / 1024^3 ≈ 4.1 GiB
        size = estimate_size_from_name("mlx-community/Qwen3-8B-4bit")
        assert size is not None
        assert 3.8 <= size <= 4.5

    def test_8bit_model(self):
        """8-bit quantized models: ~1 byte per param."""
        # 8B params at 8-bit = 8 * 1.0 * 1.1 * 1e9 / 1024^3 ≈ 8.2 GiB
        size = estimate_size_from_name("mlx-community/Llama-3.1-8B-8bit")
        assert size is not None
        assert 7.5 <= size <= 9.0

    def test_bf16_model(self):
        """BF16 models: ~2 bytes per param."""
        # 7B params at bf16 = 7 * 2.0 * 1.1 * 1e9 / 1024^3 ≈ 14.3 GiB
        size = estimate_size_from_name("mlx-community/Mistral-7B-bf16")
        assert size is not None
        assert 13.0 <= size <= 16.0

    def test_3bit_model(self):
        """3-bit quantized models: ~0.375 bytes per param."""
        # 8B params at 3-bit = 8 * 0.375 * 1.1 * 1e9 / 1024^3 ≈ 3.1 GiB
        size = estimate_size_from_name("mlx-community/Qwen3-8B-3bit")
        assert size is not None
        assert 2.8 <= size <= 3.5

    def test_large_model(self):
        """Large models like 70B."""
        # 70B params at 4-bit = 70 * 0.5 * 1.1 * 1e9 / 1024^3 ≈ 35.9 GiB
        size = estimate_size_from_name("mlx-community/Llama-3.1-70B-4bit")
        assert size is not None
        assert 33.0 <= size <= 40.0

    def test_decimal_params(self):
        """Models with decimal parameter counts like 1.7B."""
        # 1.7B at 4-bit = 1.7 * 0.5 * 1.1 * 1e9 / 1024^3 ≈ 0.87 GiB
        size = estimate_size_from_name("mlx-community/Qwen3-1.7B-4bit")
        assert size is not None
        assert 0.7 <= size <= 1.1

    def test_no_params_in_name(self):
        """Models without parameter count in name return None."""
        size = estimate_size_from_name("mlx-community/SomeModel-4bit")
        assert size is None

    def test_default_to_4bit(self):
        """Models without quantization info default to 4-bit."""
        # 8B params, assumed 4-bit ≈ 4.1 GiB
        size = estimate_size_from_name("mlx-community/Qwen3-8B")
        assert size is not None
        assert 3.8 <= size <= 4.5

    def test_hyphenated_quantization(self):
        """Handle hyphenated quantization like '4-bit'."""
        # 8B at 4-bit ≈ 4.1 GiB
        size = estimate_size_from_name("mlx-community/Model-8B-4-bit")
        assert size is not None
        assert 3.8 <= size <= 4.5


class TestGetModelSizeGb:
    """Tests for get_model_size_gb with ModelInfo."""

    def test_with_safetensors_size(self):
        """Use safetensors total when available."""
        model = ModelInfo(
            model_id="mlx-community/Test-8B-4bit",
            author="mlx-community",
            downloads=1000,
            likes=10,
            tags=[],
            last_modified=None,
            size_bytes=5_000_000_000,  # 5 billion bytes
        )
        size = get_model_size_gb(model)
        # 5_000_000_000 / 1024^3 = 4.66 GiB
        assert size == 4.66

    def test_fallback_to_name_estimation(self):
        """Fall back to name estimation when safetensors not available."""
        model = ModelInfo(
            model_id="mlx-community/Qwen3-8B-4bit",
            author="mlx-community",
            downloads=1000,
            likes=10,
            tags=[],
            last_modified=None,
            size_bytes=None,  # No safetensors info
        )
        size = get_model_size_gb(model)
        # 8B at 4-bit ≈ 4.1 GiB
        assert 3.8 <= size <= 4.5

    def test_unknown_size(self):
        """Return 0 when size cannot be determined."""
        model = ModelInfo(
            model_id="mlx-community/UnknownModel",
            author="mlx-community",
            downloads=1000,
            likes=10,
            tags=[],
            last_modified=None,
            size_bytes=None,
        )
        size = get_model_size_gb(model)
        assert size == 0.0


class TestSearchModels:
    """Integration tests for search_models (requires network)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access")
    async def test_search_returns_results(self):
        """Search returns results with size info."""
        from mlx_manager.services.hf_api import search_models

        results = await search_models("Qwen", limit=3)
        assert len(results) > 0
        assert all(r.model_id for r in results)
