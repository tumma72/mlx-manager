"""Tests for admin API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mlx_manager.mlx_server.api.v1.admin import (
    pool_status,
    preload_model,
    unload_model,
    admin_health,
    PoolStatusResponse,
    ModelLoadResponse,
    ModelUnloadResponse,
)


class TestPoolStatus:
    """Tests for /admin/models/status endpoint."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.admin.get_memory_usage")
    async def test_pool_status_empty(self, mock_memory, mock_get_pool):
        """Test pool status with no models loaded."""
        mock_pool = MagicMock()
        mock_pool._models = {}
        mock_pool.max_memory_gb = 48.0
        mock_pool.max_models = 4
        mock_get_pool.return_value = mock_pool

        mock_memory.return_value = {"active_gb": 0.0, "cache_gb": 0.0}

        response = await pool_status()

        assert isinstance(response, PoolStatusResponse)
        assert response.total_models == 0
        assert len(response.loaded_models) == 0
        assert response.max_memory_gb == 48.0

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    @patch("mlx_manager.mlx_server.api.v1.admin.get_memory_usage")
    async def test_pool_status_with_models(self, mock_memory, mock_get_pool):
        """Test pool status with loaded models."""
        # Create mock loaded model
        mock_loaded = MagicMock()
        mock_loaded.model_type = "text-gen"
        mock_loaded.size_gb = 4.5
        mock_loaded.preloaded = True
        mock_loaded.last_used = 1234567890.0
        mock_loaded.loaded_at = 1234567800.0

        mock_pool = MagicMock()
        mock_pool._models = {"test-model": mock_loaded}
        mock_pool.max_memory_gb = 48.0
        mock_pool.max_models = 4
        mock_get_pool.return_value = mock_pool

        mock_memory.return_value = {"active_gb": 4.5, "cache_gb": 0.5}

        response = await pool_status()

        assert response.total_models == 1
        assert len(response.loaded_models) == 1
        assert response.loaded_models[0].model_id == "test-model"
        assert response.loaded_models[0].model_type == "text-gen"
        assert response.loaded_models[0].preloaded is True


class TestPreloadModel:
    """Tests for /admin/models/load endpoint."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    async def test_preload_model_success(self, mock_get_pool):
        """Test successful model preload."""
        mock_loaded = MagicMock()
        mock_loaded.model_type = "text-gen"
        mock_loaded.size_gb = 4.0
        mock_loaded.preloaded = True

        mock_pool = MagicMock()
        mock_pool.preload_model = AsyncMock(return_value=mock_loaded)
        mock_get_pool.return_value = mock_pool

        response = await preload_model("mlx-community/test-model")

        assert isinstance(response, ModelLoadResponse)
        assert response.status == "loaded"
        assert response.model_id == "mlx-community/test-model"
        assert response.preloaded is True
        mock_pool.preload_model.assert_called_once_with("mlx-community/test-model")

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    async def test_preload_model_failure(self, mock_get_pool):
        """Test preload failure returns 500."""
        from fastapi import HTTPException

        mock_pool = MagicMock()
        mock_pool.preload_model = AsyncMock(side_effect=RuntimeError("Load failed"))
        mock_get_pool.return_value = mock_pool

        with pytest.raises(HTTPException) as exc_info:
            await preload_model("bad-model")

        assert exc_info.value.status_code == 500
        assert "Load failed" in exc_info.value.detail


class TestUnloadModel:
    """Tests for /admin/models/unload endpoint."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    async def test_unload_model_success(self, mock_get_pool):
        """Test successful model unload."""
        mock_pool = MagicMock()
        mock_pool.unload_model = AsyncMock(return_value=True)
        mock_get_pool.return_value = mock_pool

        response = await unload_model("test-model")

        assert isinstance(response, ModelUnloadResponse)
        assert response.status == "unloaded"
        assert response.model_id == "test-model"
        mock_pool.unload_model.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.admin.get_model_pool")
    async def test_unload_model_not_found(self, mock_get_pool):
        """Test unload of non-existent model returns 404."""
        from fastapi import HTTPException

        mock_pool = MagicMock()
        mock_pool.unload_model = AsyncMock(return_value=False)
        mock_get_pool.return_value = mock_pool

        with pytest.raises(HTTPException) as exc_info:
            await unload_model("not-loaded-model")

        assert exc_info.value.status_code == 404
        assert "not loaded" in exc_info.value.detail.lower()


class TestAdminHealth:
    """Tests for /admin/health endpoint."""

    @pytest.mark.asyncio
    async def test_admin_health(self):
        """Test admin health endpoint returns healthy."""
        response = await admin_health()
        assert response["status"] == "healthy"
