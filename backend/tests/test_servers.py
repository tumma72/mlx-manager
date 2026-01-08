"""Tests for the servers API router."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_list_running_servers_empty(client, mock_server_manager):
    """Test listing running servers when none exist."""
    response = await client.get("/api/servers")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_start_server(client, sample_profile_data, mock_server_manager):
    """Test starting a server."""
    # Create a profile first
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Start the server
    response = await client.post(f"/api/servers/{profile_id}/start")
    assert response.status_code == 200

    data = response.json()
    assert data["pid"] == 12345
    assert data["port"] == sample_profile_data["port"]


@pytest.mark.asyncio
async def test_start_server_profile_not_found(client, mock_server_manager):
    """Test starting a server for non-existent profile."""
    response = await client.post("/api/servers/999/start")
    assert response.status_code == 404
    assert "Profile not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_start_server_already_running(client, sample_profile_data):
    """Test starting a server that's already running."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.start_server = AsyncMock(side_effect=RuntimeError("Server already running"))

        # Create a profile
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # Try to start
        response = await client.post(f"/api/servers/{profile_id}/start")
        assert response.status_code == 409


@pytest.mark.asyncio
async def test_stop_server(client, sample_profile_data, mock_server_manager):
    """Test stopping a server."""
    # Create and start a server
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]
    await client.post(f"/api/servers/{profile_id}/start")

    # Stop the server
    response = await client.post(f"/api/servers/{profile_id}/stop")
    assert response.status_code == 200
    assert response.json()["stopped"] is True


@pytest.mark.asyncio
async def test_stop_server_force(client, sample_profile_data, mock_server_manager):
    """Test force stopping a server."""
    # Create and start a server
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]
    await client.post(f"/api/servers/{profile_id}/start")

    # Force stop the server
    response = await client.post(f"/api/servers/{profile_id}/stop?force=true")
    assert response.status_code == 200
    mock_server_manager.stop_server.assert_called_with(profile_id, force=True)


@pytest.mark.asyncio
async def test_stop_server_not_running(client):
    """Test stopping a server that's not running."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.stop_server = AsyncMock(return_value=False)

        response = await client.post("/api/servers/1/stop")
        assert response.status_code == 404
        assert "Server not running" in response.json()["detail"]


@pytest.mark.asyncio
async def test_restart_server(client, sample_profile_data, mock_server_manager):
    """Test restarting a server."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Restart the server
    response = await client.post(f"/api/servers/{profile_id}/restart")
    assert response.status_code == 200
    assert "pid" in response.json()


@pytest.mark.asyncio
async def test_restart_server_profile_not_found(client, mock_server_manager):
    """Test restarting a server for non-existent profile."""
    response = await client.post("/api/servers/999/restart")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_check_server_health(client, sample_profile_data, mock_server_manager):
    """Test checking server health."""
    # Configure mock to indicate server is running
    mock_server_manager.is_running = MagicMock(return_value=True)

    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Check health
    response = await client.get(f"/api/servers/{profile_id}/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["response_time_ms"] == 45.0


@pytest.mark.asyncio
async def test_check_server_health_stopped(client, sample_profile_data):
    """Test checking health of a stopped server."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.is_running = MagicMock(return_value=False)

        # Create a profile
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # Check health
        response = await client.get(f"/api/servers/{profile_id}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "stopped"


@pytest.mark.asyncio
async def test_check_server_health_profile_not_found(client, mock_server_manager):
    """Test checking health for non-existent profile."""
    response = await client.get("/api/servers/999/health")
    assert response.status_code == 404
