"""Tests for the models API router."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_search_models(auth_client, mock_hf_client):
    """Test searching for models."""
    response = await auth_client.get("/api/models/search?query=test")
    assert response.status_code == 200

    models = response.json()
    assert len(models) == 1
    assert models[0]["model_id"] == "mlx-community/test-model"
    mock_hf_client.search_mlx_models.assert_called_once_with(
        query="test", max_size_gb=None, limit=20
    )


@pytest.mark.asyncio
async def test_search_models_with_size_filter(auth_client, mock_hf_client):
    """Test searching for models with size filter."""
    response = await auth_client.get("/api/models/search?query=test&max_size_gb=50")
    assert response.status_code == 200

    mock_hf_client.search_mlx_models.assert_called_once_with(
        query="test", max_size_gb=50.0, limit=20
    )


@pytest.mark.asyncio
async def test_search_models_with_limit(auth_client, mock_hf_client):
    """Test searching for models with custom limit."""
    response = await auth_client.get("/api/models/search?query=test&limit=10")
    assert response.status_code == 200

    mock_hf_client.search_mlx_models.assert_called_once_with(
        query="test", max_size_gb=None, limit=10
    )


@pytest.mark.asyncio
async def test_search_models_empty_query(auth_client, mock_hf_client):
    """Test that empty query returns validation error."""
    response = await auth_client.get("/api/models/search?query=")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_models_error(auth_client):
    """Test search error handling."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.search_mlx_models = AsyncMock(side_effect=Exception("Search failed"))
        response = await auth_client.get("/api/models/search?query=test")
        assert response.status_code == 500
        assert "Search failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_local_models(auth_client, mock_hf_client):
    """Test listing local models."""
    response = await auth_client.get("/api/models/local")
    assert response.status_code == 200

    models = response.json()
    assert len(models) == 1
    assert models[0]["model_id"] == "mlx-community/local-model"
    mock_hf_client.list_local_models.assert_called_once()


@pytest.mark.asyncio
async def test_start_download(auth_client):
    """Test starting a model download."""
    response = await auth_client.post(
        "/api/models/download",
        json={"model_id": "mlx-community/test-model"},
    )
    assert response.status_code == 200

    data = response.json()
    assert "task_id" in data
    assert data["model_id"] == "mlx-community/test-model"


@pytest.mark.asyncio
async def test_start_download_existing_returns_same_task(auth_client):
    """Test that starting a download for model with active download returns existing task."""

    from mlx_manager.routers import models

    model_id = "mlx-community/duplicate-download-test"

    # First, start a download to create the DB record
    response1 = await auth_client.post(
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
        response2 = await auth_client.post(
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
async def test_delete_model(auth_client, mock_hf_client):
    """Test deleting a local model."""
    response = await auth_client.delete("/api/models/mlx-community/test-model")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    mock_hf_client.delete_model.assert_called_once_with("mlx-community/test-model")


@pytest.mark.asyncio
async def test_delete_model_not_found(auth_client):
    """Test deleting a non-existent model."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.delete_model = AsyncMock(return_value=False)
        response = await auth_client.delete("/api/models/mlx-community/nonexistent")
        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_download_progress_task_not_found(auth_client):
    """Test getting download progress for non-existent task."""
    with patch("mlx_manager.routers.models.download_tasks", {}):
        response = await auth_client.get("/api/models/download/nonexistent-task/progress")
        assert response.status_code == 200
        # SSE response
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


@pytest.mark.asyncio
async def test_get_download_progress_with_valid_task(auth_client):
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

        response = await auth_client.get(f"/api/models/download/{task_id}/progress")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Clean up
    models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_get_download_progress_with_error(auth_client):
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

        response = await auth_client.get(f"/api/models/download/{task_id}/progress")
        assert response.status_code == 200
        # SSE should contain error
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Clean up
    models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_start_download_creates_task(auth_client):
    """Test that start_download creates a task entry."""
    from mlx_manager.routers import models

    len(models.download_tasks)

    response = await auth_client.post(
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
async def test_list_local_models_empty(auth_client):
    """Test listing local models when none exist."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.list_local_models = MagicMock(return_value=[])
        response = await auth_client.get("/api/models/local")
        assert response.status_code == 200
        assert response.json() == []


@pytest.mark.asyncio
async def test_search_models_with_all_params(auth_client):
    """Test searching models with all parameters."""
    with patch("mlx_manager.routers.models.hf_client") as mock:
        mock.search_mlx_models = AsyncMock(return_value=[])
        response = await auth_client.get("/api/models/search?query=llama&max_size_gb=10&limit=5")
        assert response.status_code == 200
        mock.search_mlx_models.assert_called_once_with(query="llama", max_size_gb=10.0, limit=5)


@pytest.mark.asyncio
async def test_delete_model_with_path_separator(auth_client, mock_hf_client):
    """Test deleting a model with path separator in ID."""
    response = await auth_client.delete("/api/models/mlx-community/some/nested/model")
    assert response.status_code == 200
    mock_hf_client.delete_model.assert_called_once_with("mlx-community/some/nested/model")


@pytest.mark.asyncio
async def test_detect_model_options_minimax(auth_client, tmp_path):
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

        response = await auth_client.get("/api/models/detect-options/mlx-community/MiniMax-M2")
        assert response.status_code == 200

        data = response.json()
        assert data["model_family"] == "minimax"
        assert data["recommended_options"]["tool_call_parser"] == "minimax_m2"
        assert data["recommended_options"]["reasoning_parser"] == "minimax_m2"
        assert data["recommended_options"]["message_converter"] == "minimax_m2"
        assert data["is_downloaded"] is True


@pytest.mark.asyncio
async def test_detect_model_options_unknown(auth_client, tmp_path):
    """Test detecting parser options for unknown model."""
    with patch("mlx_manager.routers.models.get_model_detection_info") as mock_detect:
        mock_detect.return_value = {
            "model_family": None,
            "recommended_options": {},
            "is_downloaded": False,
            "available_parsers": ["minimax_m2", "qwen3", "glm4"],
        }

        response = await auth_client.get("/api/models/detect-options/mlx-community/Unknown-Model")
        assert response.status_code == 200

        data = response.json()
        assert data["model_family"] is None
        assert data["recommended_options"] == {}
        assert data["is_downloaded"] is False
        assert "available_parsers" in data


@pytest.mark.asyncio
async def test_detect_model_options_with_nested_path(auth_client, tmp_path):
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

        response = await auth_client.get(
            "/api/models/detect-options/mlx-community/Qwen/Qwen2.5-72B-4bit"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model_family"] == "qwen"
        mock_detect.assert_called_once_with("mlx-community/Qwen/Qwen2.5-72B-4bit")


# =============================================================================
# Tests for _update_download_record()
# =============================================================================


@pytest.mark.asyncio
async def test_update_download_record_completed():
    """Test _update_download_record with completed=True sets completed_at."""
    from contextlib import asynccontextmanager
    from datetime import UTC, datetime

    from mlx_manager.models import Download
    from mlx_manager.routers.models import _update_download_record

    # Create a mock download record
    mock_download = Download(
        id=1,
        model_id="mlx-community/test-model",
        status="downloading",
        downloaded_bytes=5000,
        total_bytes=10000,
        started_at=datetime.now(tz=UTC),
    )

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = mock_download
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    with patch("mlx_manager.database.get_session", mock_get_session):
        await _update_download_record(
            download_id=1,
            status="completed",
            downloaded_bytes=10000,
            total_bytes=10000,
            completed=True,
        )

        # Verify the download was updated correctly
        assert mock_download.status == "completed"
        assert mock_download.downloaded_bytes == 10000
        assert mock_download.total_bytes == 10000
        assert mock_download.completed_at is not None
        mock_session.add.assert_called_once_with(mock_download)
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_update_download_record_with_error():
    """Test _update_download_record sets error field correctly."""
    from contextlib import asynccontextmanager
    from datetime import UTC, datetime

    from mlx_manager.models import Download
    from mlx_manager.routers.models import _update_download_record

    mock_download = Download(
        id=2,
        model_id="mlx-community/failing-model",
        status="downloading",
        downloaded_bytes=1000,
        started_at=datetime.now(tz=UTC),
    )

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = mock_download
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    with patch("mlx_manager.database.get_session", mock_get_session):
        await _update_download_record(
            download_id=2,
            status="failed",
            error="Network connection lost",
        )

        assert mock_download.status == "failed"
        assert mock_download.error == "Network connection lost"
        # completed_at should NOT be set for failures
        assert mock_download.completed_at is None
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_update_download_record_not_found():
    """Test _update_download_record when record doesn't exist."""
    from contextlib import asynccontextmanager

    from mlx_manager.routers.models import _update_download_record

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None  # Not found
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    with patch("mlx_manager.database.get_session", mock_get_session):
        # Should not raise, just silently do nothing
        await _update_download_record(
            download_id=999,
            status="completed",
            downloaded_bytes=10000,
        )

        # add and commit should NOT be called since record wasn't found
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_update_download_record_partial_update():
    """Test _update_download_record with only some fields updated."""
    from contextlib import asynccontextmanager
    from datetime import UTC, datetime

    from mlx_manager.models import Download
    from mlx_manager.routers.models import _update_download_record

    mock_download = Download(
        id=3,
        model_id="mlx-community/partial-model",
        status="downloading",
        downloaded_bytes=5000,
        total_bytes=None,  # Not set initially
        started_at=datetime.now(tz=UTC),
    )

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = mock_download
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    with patch("mlx_manager.database.get_session", mock_get_session):
        # Update with total_bytes but without error
        await _update_download_record(
            download_id=3,
            status="downloading",
            downloaded_bytes=7500,
            total_bytes=15000,
        )

        assert mock_download.status == "downloading"
        assert mock_download.downloaded_bytes == 7500
        assert mock_download.total_bytes == 15000
        assert mock_download.error is None
        assert mock_download.completed_at is None
        mock_session.commit.assert_called_once()


# =============================================================================
# Tests for get_active_downloads()
# =============================================================================


@pytest.mark.asyncio
async def test_get_active_downloads_in_memory_tasks(auth_client):
    """Test get_active_downloads returns in-memory active tasks."""
    from mlx_manager.routers import models

    # Set up in-memory tasks
    task_id = "test-active-task-123"
    models.download_tasks[task_id] = {
        "model_id": "mlx-community/active-model",
        "status": "downloading",
        "progress": 75,
        "downloaded_bytes": 7500000,
        "total_bytes": 10000000,
    }

    try:
        response = await auth_client.get("/api/models/downloads/active")
        assert response.status_code == 200

        data = response.json()
        assert len(data) >= 1

        # Find our task in the response
        task = next((d for d in data if d["task_id"] == task_id), None)
        assert task is not None
        assert task["model_id"] == "mlx-community/active-model"
        assert task["status"] == "downloading"
        assert task["progress"] == 75
        assert task["downloaded_bytes"] == 7500000
        assert task["total_bytes"] == 10000000
    finally:
        models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_get_active_downloads_pending_status(auth_client):
    """Test get_active_downloads includes pending tasks."""
    from mlx_manager.routers import models

    task_id = "test-pending-task"
    models.download_tasks[task_id] = {
        "model_id": "mlx-community/pending-model",
        "status": "pending",
        "progress": 0,
    }

    try:
        response = await auth_client.get("/api/models/downloads/active")
        assert response.status_code == 200

        data = response.json()
        task = next((d for d in data if d["task_id"] == task_id), None)
        assert task is not None
        assert task["status"] == "pending"
    finally:
        models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_get_active_downloads_starting_status(auth_client):
    """Test get_active_downloads includes starting tasks."""
    from mlx_manager.routers import models

    task_id = "test-starting-task"
    models.download_tasks[task_id] = {
        "model_id": "mlx-community/starting-model",
        "status": "starting",
        "progress": 0,
    }

    try:
        response = await auth_client.get("/api/models/downloads/active")
        assert response.status_code == 200

        data = response.json()
        task = next((d for d in data if d["task_id"] == task_id), None)
        assert task is not None
        assert task["status"] == "starting"
    finally:
        models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_get_active_downloads_excludes_completed(auth_client):
    """Test get_active_downloads excludes completed tasks."""
    from mlx_manager.routers import models

    task_id = "test-completed-task"
    models.download_tasks[task_id] = {
        "model_id": "mlx-community/completed-model",
        "status": "completed",
        "progress": 100,
    }

    try:
        response = await auth_client.get("/api/models/downloads/active")
        assert response.status_code == 200

        data = response.json()
        # Completed task should not be in the list
        task = next((d for d in data if d["task_id"] == task_id), None)
        assert task is None
    finally:
        models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_get_active_downloads_db_backed_needs_resume(auth_client, test_engine):
    """Test get_active_downloads returns DB-backed downloads that need resume."""
    from datetime import UTC, datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models import Download
    from mlx_manager.routers import models

    # Clear any existing in-memory tasks
    original_tasks = models.download_tasks.copy()
    models.download_tasks.clear()

    # Create a session from the same engine the client uses
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Insert data using a session from the same engine
    async with async_session() as session:
        download = Download(
            model_id="mlx-community/resume-needed-model",
            status="downloading",
            downloaded_bytes=5000000,
            total_bytes=10000000,
            started_at=datetime.now(tz=UTC),
        )
        session.add(download)
        await session.commit()
        await session.refresh(download)
        download_id = download.id

    try:
        response = await auth_client.get("/api/models/downloads/active")
        assert response.status_code == 200

        data = response.json()

        # Find the DB-backed download
        resume_task = next(
            (d for d in data if d["model_id"] == "mlx-community/resume-needed-model"),
            None,
        )
        assert resume_task is not None
        assert resume_task["task_id"] == f"resume-{download_id}"
        assert resume_task["status"] == "downloading"
        assert resume_task["needs_resume"] is True
        # Progress should be calculated: (5000000 / 10000000) * 100 = 50
        assert resume_task["progress"] == 50
        assert resume_task["downloaded_bytes"] == 5000000
        assert resume_task["total_bytes"] == 10000000
    finally:
        models.download_tasks.update(original_tasks)


@pytest.mark.asyncio
async def test_get_active_downloads_db_progress_calculation_no_total(auth_client, test_engine):
    """Test progress calculation when total_bytes is 0 or None."""
    from datetime import UTC, datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models import Download
    from mlx_manager.routers import models

    original_tasks = models.download_tasks.copy()
    models.download_tasks.clear()

    # Create a session from the same engine the client uses
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Insert data using a session from the same engine
    async with async_session() as session:
        download = Download(
            model_id="mlx-community/unknown-size-model",
            status="pending",
            downloaded_bytes=1000,
            total_bytes=None,  # Unknown total
            started_at=datetime.now(tz=UTC),
        )
        session.add(download)
        await session.commit()

    try:
        response = await auth_client.get("/api/models/downloads/active")
        assert response.status_code == 200

        data = response.json()
        resume_task = next(
            (d for d in data if d["model_id"] == "mlx-community/unknown-size-model"),
            None,
        )
        assert resume_task is not None
        # Progress should be 0 when total_bytes is None
        assert resume_task["progress"] == 0
        assert resume_task["total_bytes"] == 0
    finally:
        models.download_tasks.update(original_tasks)


@pytest.mark.asyncio
async def test_get_active_downloads_in_memory_takes_precedence(auth_client, test_engine):
    """Test that in-memory tasks take precedence over DB records for same model."""
    from datetime import UTC, datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models import Download
    from mlx_manager.routers import models

    model_id = "mlx-community/precedence-test-model"

    # Create both an in-memory task and a DB record for the same model
    task_id = "test-precedence-task"
    models.download_tasks[task_id] = {
        "model_id": model_id,
        "status": "downloading",
        "progress": 80,
        "downloaded_bytes": 8000000,
        "total_bytes": 10000000,
    }

    # Create a session from the same engine the client uses
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Also create a DB record for the same model
    async with async_session() as session:
        download = Download(
            model_id=model_id,
            status="downloading",
            downloaded_bytes=5000000,
            total_bytes=10000000,
            started_at=datetime.now(tz=UTC),
        )
        session.add(download)
        await session.commit()

    try:
        response = await auth_client.get("/api/models/downloads/active")
        assert response.status_code == 200

        data = response.json()

        # Only in-memory version should be returned (not duplicated)
        matching = [d for d in data if d["model_id"] == model_id]
        assert len(matching) == 1
        assert matching[0]["task_id"] == task_id
        assert matching[0]["progress"] == 80  # In-memory value, not DB value
        assert "needs_resume" not in matching[0]  # In-memory tasks don't have this flag
    finally:
        models.download_tasks.pop(task_id, None)


# =============================================================================
# Tests for get_available_parsers() (deprecated - always returns empty list)
# =============================================================================


@pytest.mark.asyncio
async def test_get_available_parsers_returns_empty_list(auth_client):
    """Test get_available_parsers returns empty list (feature deprecated)."""
    response = await auth_client.get("/api/models/available-parsers")
    assert response.status_code == 200

    data = response.json()
    assert "parsers" in data
    assert data["parsers"] == []


# =============================================================================
# Tests for SSE progress with DB updates and error handling
# =============================================================================


@pytest.mark.asyncio
async def test_download_progress_sse_updates_db_periodically(auth_client):
    """Test SSE progress updates the database periodically."""
    from mlx_manager.routers import models

    task_id = "test-sse-db-update-task"
    models.download_tasks[task_id] = {
        "model_id": "mlx-community/sse-db-model",
        "download_id": 123,  # Has a download_id
        "status": "starting",
        "progress": 0,
    }

    # Track calls to _update_download_record
    update_calls = []

    async def mock_download_model(model_id):
        yield {
            "status": "downloading",
            "progress": 50,
            "downloaded_bytes": 5000,
            "total_bytes": 10000,
        }
        yield {
            "status": "completed",
            "progress": 100,
            "downloaded_bytes": 10000,
            "total_bytes": 10000,
        }

    async def mock_update_record(*args, **kwargs):
        update_calls.append({"args": args, "kwargs": kwargs})

    with (
        patch("mlx_manager.routers.models.hf_client") as mock_hf,
        patch("mlx_manager.routers.models.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "mlx_manager.routers.models._update_download_record",
            side_effect=mock_update_record,
        ),
    ):
        mock_hf.download_model = mock_download_model

        response = await auth_client.get(f"/api/models/download/{task_id}/progress")
        assert response.status_code == 200

    # Should have called _update_download_record at least once for the completed status
    assert len(update_calls) >= 1

    # Find the completed call
    completed_call = next(
        (c for c in update_calls if c["kwargs"].get("status") == "completed"), None
    )
    assert completed_call is not None
    assert completed_call["kwargs"]["completed"] is True

    models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_download_progress_sse_error_updates_db(auth_client):
    """Test SSE error path updates database with failure status."""
    from mlx_manager.routers import models

    task_id = "test-sse-error-db-task"
    models.download_tasks[task_id] = {
        "model_id": "mlx-community/sse-error-model",
        "download_id": 456,  # Has a download_id
        "status": "starting",
        "progress": 0,
    }

    update_calls = []

    async def mock_download_model_error(model_id):
        raise Exception("Simulated network error")
        yield  # Make it a generator

    async def mock_update_record(*args, **kwargs):
        update_calls.append({"args": args, "kwargs": kwargs})

    with (
        patch("mlx_manager.routers.models.hf_client") as mock_hf,
        patch("mlx_manager.routers.models.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "mlx_manager.routers.models._update_download_record",
            side_effect=mock_update_record,
        ),
    ):
        mock_hf.download_model = mock_download_model_error

        response = await auth_client.get(f"/api/models/download/{task_id}/progress")
        assert response.status_code == 200
        # SSE content should contain error
        assert b"failed" in response.content or b"error" in response.content

    # Should have called _update_download_record with failed status
    assert len(update_calls) == 1
    assert update_calls[0]["kwargs"]["status"] == "failed"
    assert "Simulated network error" in update_calls[0]["kwargs"]["error"]

    models.download_tasks.pop(task_id, None)


@pytest.mark.asyncio
async def test_download_progress_sse_without_download_id(auth_client):
    """Test SSE progress works even without download_id (no DB updates)."""
    from mlx_manager.routers import models

    task_id = "test-sse-no-download-id"
    models.download_tasks[task_id] = {
        "model_id": "mlx-community/no-db-model",
        # No download_id key
        "status": "starting",
        "progress": 0,
    }

    async def mock_download_model(model_id):
        yield {"status": "completed", "progress": 100}

    with (
        patch("mlx_manager.routers.models.hf_client") as mock_hf,
        patch("mlx_manager.routers.models.asyncio.sleep", new_callable=AsyncMock),
        patch("mlx_manager.routers.models._update_download_record") as mock_update,
    ):
        mock_hf.download_model = mock_download_model

        response = await auth_client.get(f"/api/models/download/{task_id}/progress")
        assert response.status_code == 200

        # _update_download_record should NOT be called when there's no download_id
        mock_update.assert_not_called()

    models.download_tasks.pop(task_id, None)


# =============================================================================
# Direct function tests (bypassing HTTP layer for coverage tracking)
# =============================================================================


@pytest.mark.asyncio
async def test_start_download_function_no_existing(test_engine):
    """Test start_download function directly when no existing download."""

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models import User, UserStatus
    from mlx_manager.routers import models
    from mlx_manager.routers.models import DownloadRequest, start_download

    # Clear any existing tasks
    original_tasks = models.download_tasks.copy()
    models.download_tasks.clear()

    # Create session from test engine
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create a mock user for the dependency
    mock_user = User(
        id=1,
        email="test@example.com",
        hashed_password="hashed",
        is_admin=False,
        status=UserStatus.APPROVED,
    )

    try:
        async with async_session() as session:
            request = DownloadRequest(model_id="mlx-community/direct-test-model")
            result = await start_download(mock_user, request, session)
            await session.commit()

            assert "task_id" in result
            assert result["model_id"] == "mlx-community/direct-test-model"

            task_id = result["task_id"]
            assert task_id in models.download_tasks
            assert models.download_tasks[task_id]["status"] == "starting"
            assert "download_id" in models.download_tasks[task_id]
    finally:
        models.download_tasks.clear()
        models.download_tasks.update(original_tasks)


@pytest.mark.asyncio
async def test_start_download_function_existing_download(test_engine):
    """Test start_download function returns existing task when download exists."""
    from datetime import UTC, datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models import Download, User, UserStatus
    from mlx_manager.routers import models
    from mlx_manager.routers.models import DownloadRequest, start_download

    original_tasks = models.download_tasks.copy()
    models.download_tasks.clear()

    model_id = "mlx-community/existing-download-direct"

    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create a mock user for the dependency
    mock_user = User(
        id=1,
        email="test@example.com",
        hashed_password="hashed",
        is_admin=False,
        status=UserStatus.APPROVED,
    )

    try:
        # Create existing download in DB
        async with async_session() as session:
            existing_download = Download(
                model_id=model_id,
                status="downloading",
                downloaded_bytes=5000,
                started_at=datetime.now(tz=UTC),
            )
            session.add(existing_download)
            await session.commit()
            await session.refresh(existing_download)
            download_id = existing_download.id

        # Add corresponding in-memory task
        existing_task_id = "existing-direct-task"
        models.download_tasks[existing_task_id] = {
            "model_id": model_id,
            "download_id": download_id,
            "status": "downloading",
            "progress": 50,
        }

        # Try to start a new download for same model
        async with async_session() as session:
            request = DownloadRequest(model_id=model_id)
            result = await start_download(mock_user, request, session)

            # Should return existing task_id
            assert result["task_id"] == existing_task_id
            assert result["model_id"] == model_id
    finally:
        models.download_tasks.clear()
        models.download_tasks.update(original_tasks)


@pytest.mark.asyncio
async def test_get_active_downloads_function_with_db_records(test_engine):
    """Test get_active_downloads function directly with DB records."""
    from datetime import UTC, datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models import Download, User, UserStatus
    from mlx_manager.routers import models
    from mlx_manager.routers.models import get_active_downloads

    original_tasks = models.download_tasks.copy()
    models.download_tasks.clear()

    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create a mock user for the dependency
    mock_user = User(
        id=1,
        email="test@example.com",
        hashed_password="hashed",
        is_admin=False,
        status=UserStatus.APPROVED,
    )

    try:
        # Create DB-backed download
        async with async_session() as session:
            download = Download(
                model_id="mlx-community/db-active-download",
                status="downloading",
                downloaded_bytes=5000000,
                total_bytes=10000000,
                started_at=datetime.now(tz=UTC),
            )
            session.add(download)
            await session.commit()
            await session.refresh(download)
            download_id = download.id

        # Call function directly
        async with async_session() as session:
            result = await get_active_downloads(mock_user, session)

            # Find our download
            found = next(
                (d for d in result if d["model_id"] == "mlx-community/db-active-download"),
                None,
            )
            assert found is not None
            assert found["task_id"] == f"resume-{download_id}"
            assert found["needs_resume"] is True
            assert found["progress"] == 50  # 5000000/10000000 * 100
    finally:
        models.download_tasks.clear()
        models.download_tasks.update(original_tasks)


@pytest.mark.asyncio
async def test_get_active_downloads_function_no_total_bytes(test_engine):
    """Test get_active_downloads progress calculation when total_bytes is None."""
    from datetime import UTC, datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models import Download, User, UserStatus
    from mlx_manager.routers import models
    from mlx_manager.routers.models import get_active_downloads

    original_tasks = models.download_tasks.copy()
    models.download_tasks.clear()

    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create a mock user for the dependency
    mock_user = User(
        id=1,
        email="test@example.com",
        hashed_password="hashed",
        is_admin=False,
        status=UserStatus.APPROVED,
    )

    try:
        # Create DB-backed download with no total_bytes
        async with async_session() as session:
            download = Download(
                model_id="mlx-community/no-total-bytes",
                status="pending",
                downloaded_bytes=1000,
                total_bytes=None,
                started_at=datetime.now(tz=UTC),
            )
            session.add(download)
            await session.commit()
            await session.refresh(download)

        # Call function directly
        async with async_session() as session:
            result = await get_active_downloads(mock_user, session)

            found = next(
                (d for d in result if d["model_id"] == "mlx-community/no-total-bytes"),
                None,
            )
            assert found is not None
            assert found["progress"] == 0  # No total_bytes means 0 progress
            assert found["total_bytes"] == 0
    finally:
        models.download_tasks.clear()
        models.download_tasks.update(original_tasks)


@pytest.mark.asyncio
async def test_start_download_no_existing_creates_db_record(auth_client, test_engine):
    """Test start_download creates DB record and task when no existing download."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlmodel import select

    from mlx_manager.models import Download
    from mlx_manager.routers import models

    # Clear any existing tasks for this test
    original_tasks = models.download_tasks.copy()

    try:
        # Start a download
        response = await auth_client.post(
            "/api/models/download",
            json={"model_id": "mlx-community/new-test-model-db"},
        )
        assert response.status_code == 200

        data = response.json()
        task_id = data["task_id"]
        assert "task_id" in data
        assert data["model_id"] == "mlx-community/new-test-model-db"

        # Verify task was created in memory
        assert task_id in models.download_tasks
        task = models.download_tasks[task_id]
        assert task["model_id"] == "mlx-community/new-test-model-db"
        assert task["status"] == "starting"
        assert "download_id" in task  # Should have DB reference

        # Verify DB record was created
        async_session = sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session() as session:
            result = await session.execute(
                select(Download).where(Download.model_id == "mlx-community/new-test-model-db")
            )
            download = result.scalars().first()
            assert download is not None
            assert download.status == "pending"
            assert download.id == task["download_id"]
    finally:
        # Clean up
        for tid in list(models.download_tasks.keys()):
            task_model = models.download_tasks.get(tid, {}).get("model_id")
            if task_model == "mlx-community/new-test-model-db":
                models.download_tasks.pop(tid, None)
        models.download_tasks.update(original_tasks)


@pytest.mark.asyncio
async def test_start_download_existing_download_in_db(auth_client, test_engine):
    """Test start_download returns existing task when download exists in DB."""
    from datetime import UTC, datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from mlx_manager.models import Download
    from mlx_manager.routers import models

    model_id = "mlx-community/already-downloading"

    # Create a session from the same engine the client uses
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create an existing download in DB with active status
    async with async_session() as session:
        download = Download(
            model_id=model_id,
            status="downloading",
            downloaded_bytes=5000,
            total_bytes=10000,
            started_at=datetime.now(tz=UTC),
        )
        session.add(download)
        await session.commit()
        await session.refresh(download)
        download_id = download.id

    # Also add to in-memory tasks
    existing_task_id = "existing-task-for-db-model"
    models.download_tasks[existing_task_id] = {
        "model_id": model_id,
        "download_id": download_id,
        "status": "downloading",
        "progress": 50,
    }

    try:
        response = await auth_client.post(
            "/api/models/download",
            json={"model_id": model_id},
        )
        assert response.status_code == 200

        data = response.json()
        # Should return the existing task_id, not create a new one
        assert data["task_id"] == existing_task_id
        assert data["model_id"] == model_id
    finally:
        models.download_tasks.pop(existing_task_id, None)


# =============================================================================
# Tests for get_model_config()
# =============================================================================


@pytest.mark.asyncio
async def test_get_model_config_local_model(auth_client):
    """Test get_model_config returns characteristics for local model."""
    with patch(
        "mlx_manager.utils.model_detection.extract_characteristics_from_model"
    ) as mock_extract_local:
        mock_extract_local.return_value = {
            "architecture": "llama",
            "context_window": 4096,
            "quantization": "4bit",
        }

        response = await auth_client.get("/api/models/config/mlx-community/local-model")
        assert response.status_code == 200

        data = response.json()
        assert data["architecture"] == "llama"
        assert data["context_window"] == 4096
        assert data["quantization"] == "4bit"


@pytest.mark.asyncio
async def test_get_model_config_remote_model(auth_client):
    """Test get_model_config fetches and returns characteristics for remote model."""
    with (
        patch(
            "mlx_manager.utils.model_detection.extract_characteristics_from_model"
        ) as mock_extract_local,
        patch("mlx_manager.services.hf_api.fetch_remote_config") as mock_fetch_remote,
        patch("mlx_manager.utils.model_detection.extract_characteristics") as mock_extract_chars,
    ):
        # Local returns None
        mock_extract_local.return_value = None

        # Remote returns config
        mock_fetch_remote.return_value = {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 8192,
        }

        # Extract from config
        mock_extract_chars.return_value = {
            "architecture": "llama",
            "context_window": 8192,
        }

        response = await auth_client.get("/api/models/config/mlx-community/remote-model")
        assert response.status_code == 200

        data = response.json()
        assert data["architecture"] == "llama"
        assert data["context_window"] == 8192


@pytest.mark.asyncio
async def test_get_model_config_not_found(auth_client):
    """Test get_model_config returns 204 when config not available."""
    with (
        patch(
            "mlx_manager.utils.model_detection.extract_characteristics_from_model"
        ) as mock_extract_local,
        patch("mlx_manager.services.hf_api.fetch_remote_config") as mock_fetch_remote,
    ):
        # Local returns None
        mock_extract_local.return_value = None

        # Remote also returns None
        mock_fetch_remote.return_value = None

        response = await auth_client.get("/api/models/config/mlx-community/unknown-model")
        assert response.status_code == 204
        assert response.content == b""


@pytest.mark.asyncio
async def test_get_model_config_with_tags(auth_client):
    """Test get_model_config passes tags parameter correctly."""
    with patch(
        "mlx_manager.utils.model_detection.extract_characteristics_from_model"
    ) as mock_extract_local:
        mock_extract_local.return_value = {
            "architecture": "llama",
            "context_window": 4096,
        }

        response = await auth_client.get(
            "/api/models/config/mlx-community/test-model?tags=mlx,quantized"
        )
        assert response.status_code == 200

        # Verify tags were parsed and passed
        mock_extract_local.assert_called_once_with(
            "mlx-community/test-model", tags=["mlx", "quantized"]
        )
