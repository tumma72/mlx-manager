"""Tests for the models API router."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_search_models(client, mock_hf_client):
    """Test searching for models."""
    response = await client.get("/api/models/search?query=test")
    assert response.status_code == 200

    models = response.json()
    assert len(models) == 1
    assert models[0]["model_id"] == "mlx-community/test-model"
    mock_hf_client.search_mlx_models.assert_called_once_with(
        query="test", max_size_gb=None, limit=20
    )


@pytest.mark.asyncio
async def test_search_models_with_size_filter(client, mock_hf_client):
    """Test searching for models with size filter."""
    response = await client.get("/api/models/search?query=test&max_size_gb=50")
    assert response.status_code == 200

    mock_hf_client.search_mlx_models.assert_called_once_with(
        query="test", max_size_gb=50.0, limit=20
    )


@pytest.mark.asyncio
async def test_search_models_with_limit(client, mock_hf_client):
    """Test searching for models with custom limit."""
    response = await client.get("/api/models/search?query=test&limit=10")
    assert response.status_code == 200

    mock_hf_client.search_mlx_models.assert_called_once_with(
        query="test", max_size_gb=None, limit=10
    )


@pytest.mark.asyncio
async def test_search_models_empty_query(client, mock_hf_client):
    """Test that empty query returns validation error."""
    response = await client.get("/api/models/search?query=")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_models_error(client):
    """Test search error handling."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.search_mlx_models = AsyncMock(side_effect=Exception("Search failed"))
        response = await client.get("/api/models/search?query=test")
        assert response.status_code == 500
        assert "Search failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_local_models(client, mock_hf_client):
    """Test listing local models."""
    response = await client.get("/api/models/local")
    assert response.status_code == 200

    models = response.json()
    assert len(models) == 1
    assert models[0]["model_id"] == "mlx-community/local-model"
    mock_hf_client.list_local_models.assert_called_once()


@pytest.mark.asyncio
async def test_start_download(client):
    """Test starting a model download."""
    response = await client.post(
        "/api/models/download?model_id=mlx-community/test-model",
    )
    assert response.status_code == 200

    data = response.json()
    assert "task_id" in data
    assert data["model_id"] == "mlx-community/test-model"


@pytest.mark.asyncio
async def test_delete_model(client, mock_hf_client):
    """Test deleting a local model."""
    response = await client.delete("/api/models/mlx-community/test-model")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    mock_hf_client.delete_model.assert_called_once_with("mlx-community/test-model")


@pytest.mark.asyncio
async def test_delete_model_not_found(client):
    """Test deleting a non-existent model."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.delete_model = AsyncMock(return_value=False)
        response = await client.delete("/api/models/mlx-community/nonexistent")
        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]
