"""Tests for the system API router."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_memory(client):
    """Test getting system memory information."""
    with patch("mlx_manager.routers.system.psutil") as mock_psutil:
        # Mock memory info
        mock_mem = MagicMock()
        mock_mem.total = 128 * 1e9  # 128 GB
        mock_mem.available = 64 * 1e9  # 64 GB
        mock_mem.used = 64 * 1e9  # 64 GB
        mock_mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_mem

        response = await client.get("/api/system/memory")
        assert response.status_code == 200

        data = response.json()
        assert data["total_gb"] == 128.0
        assert data["available_gb"] == 64.0
        assert data["used_gb"] == 64.0
        assert data["percent_used"] == 50.0
        # Default max_memory_percent is 80
        assert data["mlx_recommended_gb"] == 102.4


@pytest.mark.asyncio
async def test_get_system_info(client):
    """Test getting system information."""
    response = await client.get("/api/system/info")
    assert response.status_code == 200

    data = response.json()
    # Check that required fields are present
    assert "os_version" in data
    assert "chip" in data
    assert "memory_gb" in data
    assert "python_version" in data
    # Memory should be a positive number
    assert data["memory_gb"] > 0


@pytest.mark.asyncio
async def test_install_launchd_service(client, sample_profile_data, mock_launchd_manager):
    """Test installing a launchd service."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Install launchd service
    response = await client.post(f"/api/system/launchd/install/{profile_id}")
    assert response.status_code == 200

    data = response.json()
    assert "plist_path" in data
    assert data["label"] == "com.mlx-manager.test"


@pytest.mark.asyncio
async def test_install_launchd_service_profile_not_found(client, mock_launchd_manager):
    """Test installing launchd service for non-existent profile."""
    response = await client.post("/api/system/launchd/install/999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_uninstall_launchd_service(client, sample_profile_data, mock_launchd_manager):
    """Test uninstalling a launchd service."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # First install
    await client.post(f"/api/system/launchd/install/{profile_id}")

    # Then uninstall
    response = await client.post(f"/api/system/launchd/uninstall/{profile_id}")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_uninstall_launchd_service_profile_not_found(client, mock_launchd_manager):
    """Test uninstalling launchd service for non-existent profile."""
    response = await client.post("/api/system/launchd/uninstall/999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_launchd_status(client, sample_profile_data, mock_launchd_manager):
    """Test getting launchd service status."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Get status
    response = await client.get(f"/api/system/launchd/status/{profile_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["installed"] is False
    assert data["running"] is False
    assert data["label"] == "com.mlx-manager.test"


@pytest.mark.asyncio
async def test_get_launchd_status_profile_not_found(client, mock_launchd_manager):
    """Test getting launchd status for non-existent profile."""
    response = await client.get("/api/system/launchd/status/999")
    assert response.status_code == 404
