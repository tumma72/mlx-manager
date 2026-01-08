"""Tests for the profiles API router."""

import pytest


@pytest.mark.asyncio
async def test_list_profiles_empty(client):
    """Test listing profiles when none exist."""
    response = await client.get("/api/profiles")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_create_profile(client, sample_profile_data):
    """Test creating a new profile."""
    response = await client.post("/api/profiles", json=sample_profile_data)
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == sample_profile_data["name"]
    assert data["model_path"] == sample_profile_data["model_path"]
    assert data["port"] == sample_profile_data["port"]
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data
    assert data["launchd_installed"] is False


@pytest.mark.asyncio
async def test_list_profiles_with_data(client, sample_profile_data):
    """Test listing profiles after creating one."""
    # Create a profile
    await client.post("/api/profiles", json=sample_profile_data)

    # List profiles
    response = await client.get("/api/profiles")
    assert response.status_code == 200

    profiles = response.json()
    assert len(profiles) == 1
    assert profiles[0]["name"] == sample_profile_data["name"]


@pytest.mark.asyncio
async def test_get_profile(client, sample_profile_data):
    """Test getting a specific profile."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Get the profile
    response = await client.get(f"/api/profiles/{profile_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == profile_id
    assert data["name"] == sample_profile_data["name"]


@pytest.mark.asyncio
async def test_get_profile_not_found(client):
    """Test getting a non-existent profile."""
    response = await client.get("/api/profiles/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Profile not found"


@pytest.mark.asyncio
async def test_update_profile(client, sample_profile_data):
    """Test updating a profile."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Update the profile
    update_data = {"name": "Updated Profile", "description": "Updated description"}
    response = await client.put(f"/api/profiles/{profile_id}", json=update_data)
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Updated Profile"
    assert data["description"] == "Updated description"
    # Original fields should remain unchanged
    assert data["model_path"] == sample_profile_data["model_path"]


@pytest.mark.asyncio
async def test_update_profile_not_found(client):
    """Test updating a non-existent profile."""
    response = await client.put("/api/profiles/999", json={"name": "New Name"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_profile(client, sample_profile_data):
    """Test deleting a profile."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Delete the profile
    response = await client.delete(f"/api/profiles/{profile_id}")
    assert response.status_code == 204

    # Verify it's deleted
    get_response = await client.get(f"/api/profiles/{profile_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_profile_not_found(client):
    """Test deleting a non-existent profile."""
    response = await client.delete("/api/profiles/999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_profile_duplicate_name(client, sample_profile_data):
    """Test that creating a profile with duplicate name fails."""
    # Create first profile
    await client.post("/api/profiles", json=sample_profile_data)

    # Try to create another with same name
    response = await client.post("/api/profiles", json=sample_profile_data)
    assert response.status_code == 409
    assert "name already exists" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_profile_duplicate_port(client, sample_profile_data, sample_profile_data_alt):
    """Test that creating a profile with duplicate port fails."""
    # Create first profile
    await client.post("/api/profiles", json=sample_profile_data)

    # Try to create another with same port
    sample_profile_data_alt["port"] = sample_profile_data["port"]
    response = await client.post("/api/profiles", json=sample_profile_data_alt)
    assert response.status_code == 409
    assert "Port already in use" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_profile_duplicate_name(client, sample_profile_data, sample_profile_data_alt):
    """Test that updating a profile to a duplicate name fails."""
    # Create two profiles
    await client.post("/api/profiles", json=sample_profile_data)
    create_response = await client.post("/api/profiles", json=sample_profile_data_alt)
    profile_id = create_response.json()["id"]

    # Try to update second profile with first profile's name
    response = await client.put(
        f"/api/profiles/{profile_id}",
        json={"name": sample_profile_data["name"]},
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_update_profile_duplicate_port(client, sample_profile_data, sample_profile_data_alt):
    """Test that updating a profile to a duplicate port fails."""
    # Create two profiles
    await client.post("/api/profiles", json=sample_profile_data)
    create_response = await client.post("/api/profiles", json=sample_profile_data_alt)
    profile_id = create_response.json()["id"]

    # Try to update second profile with first profile's port
    response = await client.put(
        f"/api/profiles/{profile_id}",
        json={"port": sample_profile_data["port"]},
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_get_next_port_empty(client):
    """Test getting next port when no profiles exist."""
    response = await client.get("/api/profiles/next-port")
    assert response.status_code == 200
    # Should return default starting port (10240)
    assert response.json()["port"] == 10240


@pytest.mark.asyncio
async def test_get_next_port_with_profiles(client, sample_profile_data):
    """Test getting next port when profiles exist."""
    # Create a profile
    await client.post("/api/profiles", json=sample_profile_data)

    response = await client.get("/api/profiles/next-port")
    assert response.status_code == 200
    # Should return port after the existing one
    assert response.json()["port"] == sample_profile_data["port"] + 1


@pytest.mark.asyncio
async def test_duplicate_profile(client, sample_profile_data):
    """Test duplicating a profile."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Duplicate it
    response = await client.post(
        f"/api/profiles/{profile_id}/duplicate?new_name=Duplicated%20Profile"
    )
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "Duplicated Profile"
    assert data["model_path"] == sample_profile_data["model_path"]
    # Port should be different
    assert data["port"] != sample_profile_data["port"]
    # Should have new ID
    assert data["id"] != profile_id


@pytest.mark.asyncio
async def test_duplicate_profile_not_found(client):
    """Test duplicating a non-existent profile."""
    response = await client.post("/api/profiles/999/duplicate?new_name=New%20Name")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_duplicate_profile_name_conflict(
    client, sample_profile_data, sample_profile_data_alt
):
    """Test duplicating with an existing name fails."""
    # Create two profiles
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]
    await client.post("/api/profiles", json=sample_profile_data_alt)

    # Try to duplicate with existing name
    response = await client.post(
        f"/api/profiles/{profile_id}/duplicate?new_name={sample_profile_data_alt['name']}"
    )
    assert response.status_code == 409
