"""Tests for models listing endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mlx_manager.mlx_server.api.v1.models import (
    get_available_models,
    get_model,
    list_models,
)
from mlx_manager.mlx_server.schemas.openai import ModelInfo, ModelListResponse

# ============================================================================
# Tests for get_available_models
# ============================================================================


class TestGetAvailableModels:
    """Tests for the get_available_models helper function."""

    @patch("mlx_manager.mlx_server.api.v1.models.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.models.get_settings")
    def test_returns_configured_models(self, mock_settings, mock_pool):
        """Returns models from settings.available_models."""
        settings = MagicMock()
        settings.available_models = [
            "mlx-community/model-a",
            "mlx-community/model-b",
        ]
        mock_settings.return_value = settings

        pool = MagicMock()
        pool.get_loaded_models.return_value = []
        mock_pool.return_value = pool

        result = get_available_models()
        assert "mlx-community/model-a" in result
        assert "mlx-community/model-b" in result

    @patch("mlx_manager.mlx_server.api.v1.models.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.models.get_settings")
    def test_includes_loaded_models(self, mock_settings, mock_pool):
        """Includes currently loaded models from the pool."""
        settings = MagicMock()
        settings.available_models = ["mlx-community/model-a"]
        mock_settings.return_value = settings

        pool = MagicMock()
        pool.get_loaded_models.return_value = ["mlx-community/loaded-model"]
        mock_pool.return_value = pool

        result = get_available_models()
        assert "mlx-community/model-a" in result
        assert "mlx-community/loaded-model" in result

    @patch("mlx_manager.mlx_server.api.v1.models.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.models.get_settings")
    def test_deduplicates_models(self, mock_settings, mock_pool):
        """Models appearing in both settings and pool are deduplicated."""
        settings = MagicMock()
        settings.available_models = ["mlx-community/model-a"]
        mock_settings.return_value = settings

        pool = MagicMock()
        pool.get_loaded_models.return_value = ["mlx-community/model-a"]
        mock_pool.return_value = pool

        result = get_available_models()
        assert result.count("mlx-community/model-a") == 1

    @patch("mlx_manager.mlx_server.api.v1.models.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.models.get_settings")
    def test_returns_sorted(self, mock_settings, mock_pool):
        """Results are returned sorted alphabetically."""
        settings = MagicMock()
        settings.available_models = [
            "mlx-community/z-model",
            "mlx-community/a-model",
        ]
        mock_settings.return_value = settings

        pool = MagicMock()
        pool.get_loaded_models.return_value = ["mlx-community/m-model"]
        mock_pool.return_value = pool

        result = get_available_models()
        assert result == sorted(result)

    @patch("mlx_manager.mlx_server.api.v1.models.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.models.get_settings")
    def test_empty_when_no_models(self, mock_settings, mock_pool):
        """Returns empty list when no models configured or loaded."""
        settings = MagicMock()
        settings.available_models = []
        mock_settings.return_value = settings

        pool = MagicMock()
        pool.get_loaded_models.return_value = []
        mock_pool.return_value = pool

        result = get_available_models()
        assert result == []

    @patch("mlx_manager.mlx_server.api.v1.models.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.models.get_settings")
    def test_pool_not_initialized_graceful(self, mock_settings, mock_pool):
        """When pool is not initialized, still returns configured models."""
        settings = MagicMock()
        settings.available_models = ["mlx-community/model-a"]
        mock_settings.return_value = settings

        # Pool not initialized raises RuntimeError
        mock_pool.side_effect = RuntimeError("Pool not initialized")

        result = get_available_models()
        assert "mlx-community/model-a" in result


# ============================================================================
# Tests for list_models endpoint
# ============================================================================


class TestListModels:
    """Tests for the list_models endpoint."""

    @patch("mlx_manager.mlx_server.api.v1.models.get_available_models")
    async def test_returns_model_list_response(self, mock_get_models):
        """Returns ModelListResponse with correct structure."""
        mock_get_models.return_value = [
            "mlx-community/model-a",
            "mlx-community/model-b",
        ]

        result = await list_models()
        assert isinstance(result, ModelListResponse)
        assert result.object == "list"
        assert len(result.data) == 2

    @patch("mlx_manager.mlx_server.api.v1.models.get_available_models")
    async def test_models_have_correct_fields(self, mock_get_models):
        """Each model has id, object, owned_by fields."""
        mock_get_models.return_value = ["mlx-community/test-model"]

        result = await list_models()
        model = result.data[0]
        assert isinstance(model, ModelInfo)
        assert model.id == "mlx-community/test-model"
        assert model.object == "model"
        assert model.owned_by == "mlx-community"

    @patch("mlx_manager.mlx_server.api.v1.models.get_available_models")
    async def test_empty_pool_returns_empty_list(self, mock_get_models):
        """When no models available, returns empty data list."""
        mock_get_models.return_value = []

        result = await list_models()
        assert result.data == []
        assert result.object == "list"


# ============================================================================
# Tests for get_model endpoint
# ============================================================================


class TestGetModel:
    """Tests for the get_model endpoint."""

    @patch("mlx_manager.mlx_server.api.v1.models.get_available_models")
    async def test_returns_model_info(self, mock_get_models):
        """Returns ModelInfo for available model."""
        mock_get_models.return_value = [
            "mlx-community/test-model",
            "mlx-community/other-model",
        ]

        result = await get_model("mlx-community/test-model")
        assert isinstance(result, ModelInfo)
        assert result.id == "mlx-community/test-model"
        assert result.object == "model"
        assert result.owned_by == "mlx-community"

    @patch("mlx_manager.mlx_server.api.v1.models.get_available_models")
    async def test_model_not_found_raises_404(self, mock_get_models):
        """Returns 404 for unknown model."""
        mock_get_models.return_value = ["mlx-community/existing-model"]

        with pytest.raises(HTTPException) as exc_info:
            await get_model("mlx-community/nonexistent-model")
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail

    @patch("mlx_manager.mlx_server.api.v1.models.get_available_models")
    async def test_model_not_found_includes_available(self, mock_get_models):
        """404 error includes list of available models."""
        mock_get_models.return_value = [
            "mlx-community/model-a",
            "mlx-community/model-b",
        ]

        with pytest.raises(HTTPException) as exc_info:
            await get_model("mlx-community/missing")
        assert "mlx-community/model-a" in exc_info.value.detail
        assert "mlx-community/model-b" in exc_info.value.detail

    @patch("mlx_manager.mlx_server.api.v1.models.get_available_models")
    async def test_model_with_slash_in_id(self, mock_get_models):
        """Model IDs with slashes (org/name) work correctly."""
        mock_get_models.return_value = ["mlx-community/Llama-3.2-3B-Instruct-4bit"]

        result = await get_model("mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert result.id == "mlx-community/Llama-3.2-3B-Instruct-4bit"
