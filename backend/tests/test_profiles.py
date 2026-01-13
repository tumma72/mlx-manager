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


@pytest.mark.asyncio
async def test_get_next_port_multiple_profiles(
    client, sample_profile_data, sample_profile_data_alt
):
    """Test getting next port with multiple profiles (gap detection)."""
    # Create profiles with ports 10240 and 10241
    await client.post("/api/profiles", json=sample_profile_data)  # port 10240
    await client.post("/api/profiles", json=sample_profile_data_alt)  # port 10241

    response = await client.get("/api/profiles/next-port")
    assert response.status_code == 200
    # Should return 10242 (after the highest port)
    assert response.json()["port"] == 10242


@pytest.mark.asyncio
async def test_update_profile_all_fields(client, sample_profile_data):
    """Test updating a profile with all fields."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Update with multiple fields
    update_data = {
        "name": "Updated Profile",
        "description": "Updated description",
        "model_path": "mlx-community/updated-model",
        "model_type": "vlm",
        "port": 12345,
        "host": "0.0.0.0",
        "max_concurrency": 4,
        "queue_timeout": 600,
        "queue_size": 200,
        "log_level": "DEBUG",
        "auto_start": True,
        "context_length": 4096,
        "tool_call_parser": "native",
        "reasoning_parser": "deepseek",
        "enable_auto_tool_choice": True,
        "trust_remote_code": True,
    }
    response = await client.put(f"/api/profiles/{profile_id}", json=update_data)
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Updated Profile"
    assert data["description"] == "Updated description"
    assert data["model_path"] == "mlx-community/updated-model"
    assert data["model_type"] == "vlm"
    assert data["port"] == 12345
    assert data["host"] == "0.0.0.0"
    assert data["max_concurrency"] == 4
    assert data["auto_start"] is True
    assert data["context_length"] == 4096


@pytest.mark.asyncio
async def test_duplicate_profile_copies_all_fields(client):
    """Test that duplicating a profile copies all fields correctly."""
    # Create a profile with many fields set
    profile_data = {
        "name": "Full Profile",
        "description": "Full description",
        "model_path": "mlx-community/full-model",
        "model_type": "vlm",
        "port": 10250,
        "host": "0.0.0.0",
        "max_concurrency": 4,
        "queue_timeout": 600,
        "queue_size": 200,
        "log_level": "DEBUG",
        "context_length": 8192,
        "tool_call_parser": "native",
        "reasoning_parser": "deepseek",
        "enable_auto_tool_choice": True,
        "trust_remote_code": True,
        "chat_template_file": "/path/to/template.jinja",
        "log_file": "/path/to/log.txt",
        "no_log_file": True,
    }
    create_response = await client.post("/api/profiles", json=profile_data)
    assert create_response.status_code == 201
    profile_id = create_response.json()["id"]

    # Duplicate
    response = await client.post(
        f"/api/profiles/{profile_id}/duplicate?new_name=Duplicated%20Full%20Profile"
    )
    assert response.status_code == 201

    data = response.json()
    # Verify all fields were copied (except name, port, auto_start)
    assert data["name"] == "Duplicated Full Profile"
    assert data["description"] == profile_data["description"]
    assert data["model_path"] == profile_data["model_path"]
    assert data["model_type"] == profile_data["model_type"]
    assert data["port"] != profile_data["port"]  # Should be different
    assert data["host"] == profile_data["host"]
    assert data["max_concurrency"] == profile_data["max_concurrency"]
    assert data["queue_timeout"] == profile_data["queue_timeout"]
    assert data["queue_size"] == profile_data["queue_size"]
    assert data["log_level"] == profile_data["log_level"]
    assert data["context_length"] == profile_data["context_length"]
    assert data["tool_call_parser"] == profile_data["tool_call_parser"]
    assert data["reasoning_parser"] == profile_data["reasoning_parser"]
    assert data["enable_auto_tool_choice"] == profile_data["enable_auto_tool_choice"]
    assert data["trust_remote_code"] == profile_data["trust_remote_code"]
    assert data["chat_template_file"] == profile_data["chat_template_file"]
    assert data["log_file"] == profile_data["log_file"]
    assert data["no_log_file"] == profile_data["no_log_file"]
    assert data["auto_start"] is False  # auto_start should NOT be copied


@pytest.mark.asyncio
async def test_update_profile_same_name_allowed(client, sample_profile_data):
    """Test that updating a profile with its own name is allowed."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Update with same name should succeed
    response = await client.put(
        f"/api/profiles/{profile_id}",
        json={"name": sample_profile_data["name"], "description": "New description"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == sample_profile_data["name"]


@pytest.mark.asyncio
async def test_update_profile_same_port_allowed(client, sample_profile_data):
    """Test that updating a profile with its own port is allowed."""
    # Create a profile
    create_response = await client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Update with same port should succeed
    response = await client.put(
        f"/api/profiles/{profile_id}",
        json={"port": sample_profile_data["port"], "description": "New description"},
    )
    assert response.status_code == 200
    assert response.json()["port"] == sample_profile_data["port"]


@pytest.mark.asyncio
async def test_create_profile_validates_required_fields(client):
    """Test that creating a profile without required fields fails."""
    # Missing model_path and port
    response = await client.post("/api/profiles", json={"name": "Test"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_profiles_returns_all(client, sample_profile_data, sample_profile_data_alt):
    """Test that list_profiles returns all profiles."""
    # Create two profiles
    await client.post("/api/profiles", json=sample_profile_data)
    await client.post("/api/profiles", json=sample_profile_data_alt)

    # List should return both
    response = await client.get("/api/profiles")
    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) == 2
    names = {p["name"] for p in profiles}
    assert names == {sample_profile_data["name"], sample_profile_data_alt["name"]}
