"""Tests for the servers API router.

With the embedded MLX Server, this router provides status information
about the model pool and loaded models. Start/stop/restart endpoints
return informative messages since the embedded server is always running.

The main /api/servers endpoint returns an empty list for UI compatibility
(the frontend expects RunningServer objects with profile-based fields).
Use /api/servers/embedded for actual embedded server status.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_list_servers_returns_empty_for_ui_compatibility(auth_client):
    """Test listing servers returns empty list for UI compatibility.

    The frontend expects RunningServer objects with profile_id, profile_name,
    pid, port, etc. Since embedded mode doesn't use profiles for server
    management, we return an empty list.
    """
    response = await auth_client.get("/api/servers")
    assert response.status_code == 200

    data = response.json()
    assert data == []


@pytest.mark.asyncio
async def test_get_embedded_status_running(auth_client):
    """Test getting embedded server status when running."""
    mock_pool = MagicMock()
    mock_pool.get_loaded_models.return_value = []
    mock_pool.max_memory_gb = 32.0

    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool
    ):
        with patch(
            "mlx_manager.mlx_server.utils.memory.get_memory_usage",
            return_value={"active_gb": 0.0},
        ):
            response = await auth_client.get("/api/servers/embedded")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "running"
            assert data["type"] == "embedded"


@pytest.mark.asyncio
async def test_get_embedded_status_not_initialized(auth_client):
    """Test getting embedded server status when pool not initialized."""
    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool",
        side_effect=RuntimeError("Pool not initialized"),
    ):
        response = await auth_client.get("/api/servers/embedded")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "not_initialized"


@pytest.mark.asyncio
async def test_list_loaded_models_empty(auth_client):
    """Test listing loaded models when none are loaded."""
    mock_pool = MagicMock()
    mock_pool.get_loaded_models.return_value = []
    mock_pool._models = {}

    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool
    ):
        response = await auth_client.get("/api/servers/models")
        assert response.status_code == 200
        assert response.json() == []


@pytest.mark.asyncio
async def test_list_loaded_models_with_data(auth_client):
    """Test listing loaded models with loaded models."""
    mock_loaded_model = MagicMock()
    mock_loaded_model.model_id = "test-model"
    mock_loaded_model.model_type = "lm"
    mock_loaded_model.size_gb = 4.0
    mock_loaded_model.loaded_at = 1704067200.0
    mock_loaded_model.last_used = 1704067200.0
    mock_loaded_model.preloaded = False

    mock_pool = MagicMock()
    mock_pool.get_loaded_models.return_value = ["test-model"]
    mock_pool._models = {"test-model": mock_loaded_model}

    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool
    ):
        response = await auth_client.get("/api/servers/models")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["model_id"] == "test-model"
        assert data[0]["model_type"] == "lm"


@pytest.mark.asyncio
async def test_check_server_health_healthy(auth_client):
    """Test checking server health when healthy."""
    mock_pool = MagicMock()
    mock_pool.get_loaded_models.return_value = ["model1"]
    mock_pool.max_memory_gb = 32.0

    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool
    ):
        with patch(
            "mlx_manager.mlx_server.utils.memory.get_memory_usage",
            return_value={"active_gb": 8.0},
        ):
            response = await auth_client.get("/api/servers/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_pool_initialized"] is True
            assert data["loaded_model_count"] == 1


@pytest.mark.asyncio
async def test_check_server_health_degraded(auth_client):
    """Test checking server health when memory is low."""
    mock_pool = MagicMock()
    mock_pool.get_loaded_models.return_value = ["model1"]
    mock_pool.max_memory_gb = 8.0

    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool
    ):
        with patch(
            "mlx_manager.mlx_server.utils.memory.get_memory_usage",
            return_value={"active_gb": 7.5},  # Less than 1GB available
        ):
            response = await auth_client.get("/api/servers/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "degraded"


@pytest.mark.asyncio
async def test_check_server_health_unhealthy(auth_client):
    """Test checking server health when pool not initialized."""
    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool",
        side_effect=RuntimeError("Pool not initialized"),
    ):
        response = await auth_client.get("/api/servers/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_pool_initialized"] is False


@pytest.mark.asyncio
async def test_get_memory_status(auth_client):
    """Test getting memory status."""
    mock_pool = MagicMock()
    mock_pool.max_memory_gb = 32.0
    mock_pool.max_models = 5
    mock_pool.get_loaded_models.return_value = ["model1", "model2"]

    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool
    ):
        with patch(
            "mlx_manager.mlx_server.utils.memory.get_memory_usage",
            return_value={
                "cache_gb": 0.5,
                "active_gb": 8.0,
                "peak_gb": 10.0,
            },
        ):
            response = await auth_client.get("/api/servers/memory")
            assert response.status_code == 200

            data = response.json()
            assert data["active_gb"] == 8.0
            assert data["limit_gb"] == 32.0
            assert data["available_gb"] == 24.0
            assert data["loaded_models"] == 2


@pytest.mark.asyncio
async def test_legacy_start_endpoint(auth_client, sample_profile_data):
    """Test legacy start endpoint returns informative message."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    response = await auth_client.post(f"/api/servers/{profile_id}/start")
    assert response.status_code == 200

    data = response.json()
    assert "embedded" in data["message"].lower() or "always running" in data["message"].lower()
    assert data["pid"] == os.getpid()


@pytest.mark.asyncio
async def test_legacy_stop_endpoint(auth_client, sample_profile_data):
    """Test legacy stop endpoint returns informative message."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    response = await auth_client.post(f"/api/servers/{profile_id}/stop")
    assert response.status_code == 200

    data = response.json()
    assert "embedded" in data["message"].lower() or "cannot be stopped" in data["message"].lower()


@pytest.mark.asyncio
async def test_legacy_restart_endpoint(auth_client, sample_profile_data):
    """Test legacy restart endpoint returns informative message."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    response = await auth_client.post(f"/api/servers/{profile_id}/restart")
    assert response.status_code == 200

    data = response.json()
    assert "embedded" in data["message"].lower() or "cannot be restarted" in data["message"].lower()


@pytest.mark.asyncio
async def test_get_server_status(auth_client, sample_profile_data):
    """Test getting server status for a profile.

    In embedded mode, the server is always running. This endpoint supports
    the frontend's polling logic during "startup".
    """
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    response = await auth_client.get(f"/api/servers/{profile_id}/status")
    assert response.status_code == 200

    data = response.json()
    assert data["profile_id"] == profile_id
    assert data["running"] is True
    assert data["pid"] == os.getpid()
    assert data["failed"] is False
    assert data["error_message"] is None


@pytest.mark.asyncio
async def test_get_server_health_for_profile(auth_client, sample_profile_data):
    """Test getting server health for a profile.

    In embedded mode, the server is always healthy. The model_loaded field
    is True to indicate the server is ready to accept requests (models load
    on-demand when a chat request is made).
    """
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    response = await auth_client.get(f"/api/servers/{profile_id}/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["error"] is None
