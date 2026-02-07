"""Tests for the servers API router.

With the embedded MLX Server, this router provides status information
about the model pool and loaded models. Start/stop/restart endpoints
return informative messages since the embedded server is always running.

The main /api/servers endpoint returns RunningServer objects for profiles
whose models are loaded in the model pool.
Use /api/servers/embedded for actual embedded server status.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from mlx_manager.config import DEFAULT_PORT


@pytest.mark.asyncio
async def test_list_servers_returns_empty_when_no_models_loaded(auth_client):
    """Test listing servers returns empty list when no models are loaded."""
    mock_pool = MagicMock()
    mock_pool.get_loaded_models.return_value = []

    with patch("mlx_manager.routers.servers.get_model_pool", return_value=mock_pool):
        response = await auth_client.get("/api/servers")
        assert response.status_code == 200

        data = response.json()
        assert data == []


@pytest.mark.asyncio
async def test_list_servers_returns_running_servers_when_models_loaded(auth_client, test_session):
    """Test listing servers returns RunningServer objects for profiles with loaded models."""
    from mlx_manager.models import ServerProfile

    # Create a test profile
    profile = ServerProfile(
        name="Test Profile",
        model_path="mlx-community/test-model",
        port=DEFAULT_PORT,
    )
    test_session.add(profile)
    await test_session.commit()
    await test_session.refresh(profile)

    # Mock the model pool to show the model is loaded
    mock_loaded_model = MagicMock()
    mock_loaded_model.loaded_at = time.time() - 60  # Loaded 60 seconds ago
    mock_loaded_model.size_gb = 4.0  # Per-model memory size

    mock_pool = MagicMock()
    mock_pool.get_loaded_models.return_value = ["mlx-community/test-model"]
    mock_pool.get_loaded_model.return_value = mock_loaded_model
    mock_pool.max_memory_gb = 32.0

    with patch("mlx_manager.routers.servers.get_model_pool", return_value=mock_pool):
        response = await auth_client.get("/api/servers")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["profile_id"] == profile.id
        assert data[0]["profile_name"] == "Test Profile"
        assert data[0]["health_status"] == "healthy"
        assert data[0]["uptime_seconds"] >= 59  # Approximately 60 seconds
        # Verify per-model memory metrics
        assert data[0]["memory_mb"] == 4096.0  # 4 GB in MB
        assert data[0]["memory_percent"] == 12.5  # 4 / 32 * 100
        assert data[0]["memory_limit_percent"] == 12.5  # 4 / 32 * 100


@pytest.mark.asyncio
async def test_get_embedded_status_running(auth_client):
    """Test getting embedded server status when running."""
    mock_pool = MagicMock()
    mock_pool.get_loaded_models.return_value = []
    mock_pool.max_memory_gb = 32.0

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
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
    mock_pool.get_loaded_model.return_value = None

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
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
    mock_pool.get_loaded_model.return_value = mock_loaded_model

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
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

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
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

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
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

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
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
async def test_start_endpoint_triggers_model_loading(auth_client, sample_profile_data):
    """Test start endpoint triggers model loading in background."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]
    model_path = sample_profile_data["model_path"]

    mock_pool = MagicMock()
    mock_pool.is_loaded.return_value = False

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
        response = await auth_client.post(f"/api/servers/{profile_id}/start")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "loading"
        assert data["model"] == model_path
        assert data["pid"] == os.getpid()


@pytest.mark.asyncio
async def test_start_endpoint_already_loaded(auth_client, sample_profile_data):
    """Test start endpoint when model is already loaded."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]
    model_path = sample_profile_data["model_path"]

    mock_pool = MagicMock()
    mock_pool.is_loaded.return_value = True

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
        response = await auth_client.post(f"/api/servers/{profile_id}/start")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "already_loaded"
        assert data["model"] == model_path


@pytest.mark.asyncio
async def test_start_endpoint_pool_not_initialized(auth_client, sample_profile_data):
    """Test start endpoint when model pool is not initialized."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool",
        side_effect=RuntimeError("Pool not initialized"),
    ):
        response = await auth_client.post(f"/api/servers/{profile_id}/start")
        assert response.status_code == 503


@pytest.mark.asyncio
async def test_start_endpoint_profile_not_found(auth_client):
    """Test start endpoint with non-existent profile."""
    mock_pool = MagicMock()

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
        response = await auth_client.post("/api/servers/99999/start")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_stop_endpoint_unloads_model(auth_client, sample_profile_data):
    """Test stop endpoint unloads the model from memory."""
    from unittest.mock import AsyncMock

    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Mock loaded model
    mock_loaded_model = MagicMock()
    mock_loaded_model.preloaded = False

    # Mock the model pool
    mock_pool = MagicMock()
    mock_pool.is_loaded.return_value = True
    mock_pool.get_loaded_model.return_value = mock_loaded_model
    mock_pool.unload_model = AsyncMock(return_value=True)

    with patch("mlx_manager.routers.servers.get_model_pool", return_value=mock_pool):
        response = await auth_client.post(f"/api/servers/{profile_id}/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "unloaded successfully" in data["message"]


@pytest.mark.asyncio
async def test_stop_endpoint_model_not_loaded(auth_client, sample_profile_data):
    """Test stop endpoint when model is not loaded."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    mock_pool = MagicMock()
    mock_pool.is_loaded.return_value = False

    with patch("mlx_manager.routers.servers.get_model_pool", return_value=mock_pool):
        response = await auth_client.post(f"/api/servers/{profile_id}/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "not currently loaded" in data["message"]


@pytest.mark.asyncio
async def test_stop_endpoint_preloaded_model(auth_client, sample_profile_data):
    """Test stop endpoint returns error for preloaded models."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Mock preloaded model
    mock_loaded_model = MagicMock()
    mock_loaded_model.preloaded = True

    mock_pool = MagicMock()
    mock_pool.is_loaded.return_value = True
    mock_pool.get_loaded_model.return_value = mock_loaded_model

    with patch("mlx_manager.routers.servers.get_model_pool", return_value=mock_pool):
        response = await auth_client.post(f"/api/servers/{profile_id}/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False
        assert "preloaded and protected" in data["message"]


@pytest.mark.asyncio
async def test_stop_endpoint_profile_not_found(auth_client):
    """Test stop endpoint with non-existent profile."""
    mock_pool = MagicMock()

    with patch("mlx_manager.routers.servers.get_model_pool", return_value=mock_pool):
        response = await auth_client.post("/api/servers/99999/stop")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_stop_endpoint_pool_not_initialized(auth_client, sample_profile_data):
    """Test stop endpoint when model pool is not initialized."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    with patch(
        "mlx_manager.routers.servers.get_model_pool",
        side_effect=RuntimeError("Pool not initialized"),
    ):
        response = await auth_client.post(f"/api/servers/{profile_id}/stop")
        assert response.status_code == 503


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
async def test_get_server_health_for_profile_model_loaded(auth_client, sample_profile_data):
    """Test getting server health when profile's model is loaded."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    mock_pool = MagicMock()
    mock_pool.is_loaded.return_value = True

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
        response = await auth_client.get(f"/api/servers/{profile_id}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["error"] is None


@pytest.mark.asyncio
async def test_get_server_health_for_profile_model_not_loaded(auth_client, sample_profile_data):
    """Test getting server health when profile's model is not loaded."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    mock_pool = MagicMock()
    mock_pool.is_loaded.return_value = False

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
        response = await auth_client.get(f"/api/servers/{profile_id}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
        assert data["error"] is None


@pytest.mark.asyncio
async def test_get_server_health_for_profile_pool_not_initialized(auth_client, sample_profile_data):
    """Test getting server health when model pool is not initialized."""
    # Create a profile first
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    with patch(
        "mlx_manager.mlx_server.models.pool.get_model_pool",
        side_effect=RuntimeError("Pool not initialized"),
    ):
        response = await auth_client.get(f"/api/servers/{profile_id}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert data["error"] == "Model pool not initialized"


@pytest.mark.asyncio
async def test_get_server_health_for_profile_not_found(auth_client):
    """Test getting server health for non-existent profile."""
    mock_pool = MagicMock()

    with patch("mlx_manager.mlx_server.models.pool.get_model_pool", return_value=mock_pool):
        response = await auth_client.get("/api/servers/99999/health")
        assert response.status_code == 404
