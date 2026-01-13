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


@pytest.mark.asyncio
async def test_list_running_servers_with_data(client, sample_profile_data):
    """Test listing running servers when servers are running."""
    import time

    with patch("mlx_manager.routers.servers.server_manager") as mock:
        # Create a profile first
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # Configure mock to return running server
        mock.get_all_running = MagicMock(
            return_value=[
                {
                    "profile_id": profile_id,
                    "pid": 12345,
                    "status": "healthy",
                    "memory_mb": 1024.0,
                    "create_time": time.time() - 100,  # Started 100 seconds ago
                }
            ]
        )

        response = await client.get("/api/servers")
        assert response.status_code == 200

        servers = response.json()
        assert len(servers) == 1
        assert servers[0]["profile_id"] == profile_id
        assert servers[0]["pid"] == 12345
        assert servers[0]["health_status"] == "healthy"
        assert servers[0]["memory_mb"] == 1024.0
        assert servers[0]["uptime_seconds"] >= 100


@pytest.mark.asyncio
async def test_list_running_servers_filters_missing_profiles(client, sample_profile_data):
    """Test that list_running_servers filters out servers with missing profiles."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        # Return a running server for a profile that doesn't exist
        mock.get_all_running = MagicMock(
            return_value=[
                {
                    "profile_id": 999,  # Non-existent profile
                    "pid": 12345,
                    "status": "healthy",
                    "memory_mb": 1024.0,
                }
            ]
        )

        response = await client.get("/api/servers")
        assert response.status_code == 200
        # Should return empty list since profile doesn't exist
        assert response.json() == []


@pytest.mark.asyncio
async def test_start_server_cleans_stale_instance(client, sample_profile_data):
    """Test that starting a server cleans up stale instance records."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        # Create a profile
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # First, mock server not running (stale record)
        mock.is_running = MagicMock(return_value=False)
        mock.start_server = AsyncMock(return_value=99999)

        # Start server (should clean up stale record and create new one)
        response = await client.post(f"/api/servers/{profile_id}/start")
        assert response.status_code == 200
        assert response.json()["pid"] == 99999


@pytest.mark.asyncio
async def test_start_server_generic_error(client, sample_profile_data):
    """Test starting a server with a generic exception."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.start_server = AsyncMock(side_effect=Exception("Process spawn failed"))
        mock.is_running = MagicMock(return_value=False)

        # Create a profile
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # Try to start
        response = await client.post(f"/api/servers/{profile_id}/start")
        assert response.status_code == 500
        assert "Process spawn failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_stop_server_removes_db_record(client, sample_profile_data):
    """Test that stopping a server removes the database record."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.start_server = AsyncMock(return_value=12345)
        mock.stop_server = AsyncMock(return_value=True)
        mock.is_running = MagicMock(return_value=False)

        # Create and start
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]
        await client.post(f"/api/servers/{profile_id}/start")

        # Stop
        response = await client.post(f"/api/servers/{profile_id}/stop")
        assert response.status_code == 200
        assert response.json()["stopped"] is True


@pytest.mark.asyncio
async def test_restart_server_updates_existing_instance(client, sample_profile_data):
    """Test restarting a server updates existing instance record."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.start_server = AsyncMock(return_value=12345)
        mock.stop_server = AsyncMock(return_value=True)
        mock.is_running = MagicMock(return_value=False)

        # Create and start
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]
        await client.post(f"/api/servers/{profile_id}/start")

        # Restart (should update existing instance)
        mock.start_server = AsyncMock(return_value=54321)  # New PID
        response = await client.post(f"/api/servers/{profile_id}/restart")
        assert response.status_code == 200
        assert response.json()["pid"] == 54321


@pytest.mark.asyncio
async def test_restart_server_creates_new_instance(client, sample_profile_data):
    """Test restarting a server creates new instance if none exists."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.start_server = AsyncMock(return_value=12345)
        mock.stop_server = AsyncMock(return_value=False)  # Server wasn't running
        mock.is_running = MagicMock(return_value=False)

        # Create profile but don't start
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # Restart (should create new instance)
        response = await client.post(f"/api/servers/{profile_id}/restart")
        assert response.status_code == 200
        assert response.json()["pid"] == 12345


@pytest.mark.asyncio
async def test_restart_server_error(client, sample_profile_data):
    """Test restarting a server with an error."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.stop_server = AsyncMock(return_value=False)
        mock.start_server = AsyncMock(side_effect=Exception("Restart failed"))
        mock.is_running = MagicMock(return_value=False)

        # Create a profile
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # Try to restart
        response = await client.post(f"/api/servers/{profile_id}/restart")
        assert response.status_code == 500
        assert "Restart failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_stream_logs(client, sample_profile_data):
    """Test streaming server logs."""

    with patch("mlx_manager.routers.servers.server_manager") as mock:
        # Setup mock to return some log lines then stop
        log_lines_calls = [0]

        def get_log_lines(profile_id):
            log_lines_calls[0] += 1
            if log_lines_calls[0] == 1:
                return ["Log line 1", "Log line 2"]
            return []

        def is_running(profile_id):
            # Stop after first call to get_log_lines
            return log_lines_calls[0] < 2

        mock.get_log_lines = MagicMock(side_effect=get_log_lines)
        mock.is_running = MagicMock(side_effect=is_running)

        # Create a profile
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # Stream logs - need to handle SSE response
        response = await client.get(f"/api/servers/{profile_id}/logs")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


@pytest.mark.asyncio
async def test_check_health_returns_health_data(client, sample_profile_data):
    """Test health check returns full health data when server is running."""
    with patch("mlx_manager.routers.servers.server_manager") as mock:
        mock.is_running = MagicMock(return_value=True)
        mock.check_health = AsyncMock(
            return_value={
                "status": "healthy",
                "response_time_ms": 25.5,
                "model_loaded": True,
            }
        )

        # Create a profile
        create_response = await client.post("/api/profiles", json=sample_profile_data)
        profile_id = create_response.json()["id"]

        # Check health
        response = await client.get(f"/api/servers/{profile_id}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["response_time_ms"] == 25.5
        assert data["model_loaded"] is True
