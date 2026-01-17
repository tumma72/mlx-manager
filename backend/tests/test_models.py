"""Tests for the models API router."""

from unittest.mock import AsyncMock, MagicMock, patch

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
        "/api/models/download",
        json={"model_id": "mlx-community/test-model"},
    )
    assert response.status_code == 200

    data = response.json()
    assert "task_id" in data
    assert data["model_id"] == "mlx-community/test-model"


@pytest.mark.asyncio
async def test_start_download_existing_returns_same_task(client):
    """Test that starting a download for model with active download returns existing task."""
    from mlx_manager.routers import models
    from mlx_manager.models import Download
    from datetime import datetime, UTC

    model_id = "mlx-community/duplicate-download-test"

    # First, start a download to create the DB record
    response1 = await client.post(
        "/api/models/download",
        json={"model_id": model_id},
    )
    assert response1.status_code == 200
    first_task_id = response1.json()["task_id"]

    # Manually update the task status to simulate an active download
    models.download_tasks[first_task_id]["status"] = "downloading"
    models.download_tasks[first_task_id]["progress"] = 50

    try:
        # Try to start a new download for the same model
        response2 = await client.post(
            "/api/models/download",
            json={"model_id": model_id},
        )
        assert response2.status_code == 200

        data = response2.json()
        # Should return the same task_id, not create a new one
        assert data["task_id"] == first_task_id
        assert data["model_id"] == model_id
    finally:
        # Clean up
        models.download_tasks.pop(first_task_id, None)


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


@pytest.mark.asyncio
async def test_get_download_progress_task_not_found(client):
    """Test getting download progress for non-existent task."""
    with patch("mlx_manager.routers.models.download_tasks", {}):
        response = await client.get("/api/models/download/nonexistent-task/progress")
        assert response.status_code == 200
        # SSE response
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


@pytest.mark.asyncio
async def test_get_download_progress_with_valid_task(client):
    """Test getting download progress for valid task."""
    from mlx_manager.routers import models

    # Create a download task
    task_id = "test-task-123"
    models.download_tasks[task_id] = {
        "model_id": "mlx-community/test-model",
        "status": "starting",
        "progress": 0,
    }

    async def mock_download_model(model_id):
        yield {"status": "downloading", "progress": 50}
        yield {"status": "completed", "progress": 100}

    with (
        patch("mlx_manager.routers.models.hf_client") as mock,
        patch("mlx_manager.routers.models.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock.download_model = mock_download_model

        response = await client.get(f"/api/models/download/{task_id}/progress")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Clean up
    models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_get_download_progress_with_error(client):
    """Test download progress when download fails."""
    from mlx_manager.routers import models

    # Create a download task
    task_id = "test-task-error"
    models.download_tasks[task_id] = {
        "model_id": "mlx-community/failing-model",
        "status": "starting",
        "progress": 0,
    }

    async def mock_download_model(model_id):
        raise Exception("Download failed: network error")
        yield  # Make it a generator (unreachable)

    with (
        patch("mlx_manager.routers.models.hf_client") as mock,
        patch("mlx_manager.routers.models.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock.download_model = mock_download_model

        response = await client.get(f"/api/models/download/{task_id}/progress")
        assert response.status_code == 200
        # SSE should contain error
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Clean up
    models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_start_download_creates_task(client):
    """Test that start_download creates a task entry."""
    from mlx_manager.routers import models

    len(models.download_tasks)

    response = await client.post(
        "/api/models/download",
        json={"model_id": "mlx-community/new-model"},
    )
    assert response.status_code == 200

    data = response.json()
    task_id = data["task_id"]
    assert data["model_id"] == "mlx-community/new-model"

    # Verify task was created
    assert task_id in models.download_tasks
    assert models.download_tasks[task_id]["model_id"] == "mlx-community/new-model"
    assert models.download_tasks[task_id]["status"] == "starting"

    # Clean up
    models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_list_local_models_empty(client):
    """Test listing local models when none exist."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.list_local_models = MagicMock(return_value=[])
        response = await client.get("/api/models/local")
        assert response.status_code == 200
        assert response.json() == []


@pytest.mark.asyncio
async def test_search_models_with_all_params(client):
    """Test searching models with all parameters."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.search_mlx_models = AsyncMock(return_value=[])
        response = await client.get("/api/models/search?query=llama&max_size_gb=10&limit=5")
        assert response.status_code == 200
        mock.search_mlx_models.assert_called_once_with(query="llama", max_size_gb=10.0, limit=5)


@pytest.mark.asyncio
async def test_delete_model_with_path_separator(client, mock_hf_client):
    """Test deleting a model with path separator in ID."""
    response = await client.delete("/api/models/mlx-community/some/nested/model")
    assert response.status_code == 200
    mock_hf_client.delete_model.assert_called_once_with("mlx-community/some/nested/model")


@pytest.mark.asyncio
async def test_detect_model_options_minimax(client, tmp_path):
    """Test detecting parser options for MiniMax model."""
    with patch("mlx_manager.routers.models.get_model_detection_info") as mock_detect:
        mock_detect.return_value = {
            "model_family": "minimax",
            "recommended_options": {
                "tool_call_parser": "minimax_m2",
                "reasoning_parser": "minimax_m2",
                "message_converter": "minimax_m2",
            },
            "is_downloaded": True,
            "available_parsers": ["minimax_m2", "qwen3", "glm4"],
        }

        response = await client.get("/api/models/detect-options/mlx-community/MiniMax-M2")
        assert response.status_code == 200

        data = response.json()
        assert data["model_family"] == "minimax"
        assert data["recommended_options"]["tool_call_parser"] == "minimax_m2"
        assert data["recommended_options"]["reasoning_parser"] == "minimax_m2"
        assert data["recommended_options"]["message_converter"] == "minimax_m2"
        assert data["is_downloaded"] is True


@pytest.mark.asyncio
async def test_detect_model_options_unknown(client, tmp_path):
    """Test detecting parser options for unknown model."""
    with patch("mlx_manager.routers.models.get_model_detection_info") as mock_detect:
        mock_detect.return_value = {
            "model_family": None,
            "recommended_options": {},
            "is_downloaded": False,
            "available_parsers": ["minimax_m2", "qwen3", "glm4"],
        }

        response = await client.get("/api/models/detect-options/mlx-community/Unknown-Model")
        assert response.status_code == 200

        data = response.json()
        assert data["model_family"] is None
        assert data["recommended_options"] == {}
        assert data["is_downloaded"] is False
        assert "available_parsers" in data


@pytest.mark.asyncio
async def test_detect_model_options_with_nested_path(client, tmp_path):
    """Test detecting parser options for model with nested path."""
    with patch("mlx_manager.routers.models.get_model_detection_info") as mock_detect:
        mock_detect.return_value = {
            "model_family": "qwen",
            "recommended_options": {
                "tool_call_parser": "qwen3",
                "reasoning_parser": "qwen3",
                "message_converter": "qwen3",
            },
            "is_downloaded": False,
            "available_parsers": ["minimax_m2", "qwen3", "glm4"],
        }

        response = await client.get(
            "/api/models/detect-options/mlx-community/Qwen/Qwen2.5-72B-4bit"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model_family"] == "qwen"
        mock_detect.assert_called_once_with("mlx-community/Qwen/Qwen2.5-72B-4bit")
