"""Tests for the profiles API router."""

import pytest


@pytest.mark.asyncio
async def test_list_profiles_empty(auth_client):
    """Test listing profiles when none exist."""
    response = await auth_client.get("/api/profiles")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_create_profile(auth_client, sample_profile_data):
    """Test creating a new profile."""
    response = await auth_client.post("/api/profiles", json=sample_profile_data)
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == sample_profile_data["name"]
    assert data["model_id"] == sample_profile_data["model_id"]
    assert data["model_repo_id"] == "mlx-community/test-model-4bit"
    assert data["model_type"] == "text-gen"
    assert data["profile_type"] == "inference"
    assert data["inference"]["temperature"] == sample_profile_data["inference"]["temperature"]
    assert data["inference"]["max_tokens"] == sample_profile_data["inference"]["max_tokens"]
    assert data["inference"]["top_p"] == sample_profile_data["inference"]["top_p"]
    assert data["context"]["context_length"] is None
    assert data["context"]["system_prompt"] is None
    assert data["context"]["enable_tool_injection"] is False
    assert data["audio"] is None
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data
    assert data["launchd_installed"] is False


@pytest.mark.asyncio
async def test_list_profiles_with_data(auth_client, sample_profile_data):
    """Test listing profiles after creating one."""
    # Create a profile
    await auth_client.post("/api/profiles", json=sample_profile_data)

    # List profiles
    response = await auth_client.get("/api/profiles")
    assert response.status_code == 200

    profiles = response.json()
    assert len(profiles) == 1
    assert profiles[0]["name"] == sample_profile_data["name"]


@pytest.mark.asyncio
async def test_get_profile(auth_client, sample_profile_data):
    """Test getting a specific profile."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Get the profile
    response = await auth_client.get(f"/api/profiles/{profile_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == profile_id
    assert data["name"] == sample_profile_data["name"]


@pytest.mark.asyncio
async def test_get_profile_not_found(auth_client):
    """Test getting a non-existent profile."""
    response = await auth_client.get("/api/profiles/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Profile not found"


@pytest.mark.asyncio
async def test_update_profile(auth_client, sample_profile_data):
    """Test updating a profile."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Update the profile
    update_data = {"name": "Updated Profile", "description": "Updated description"}
    response = await auth_client.put(f"/api/profiles/{profile_id}", json=update_data)
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Updated Profile"
    assert data["description"] == "Updated description"
    # Original fields should remain unchanged
    assert data["model_id"] == sample_profile_data["model_id"]
    assert data["inference"]["temperature"] == sample_profile_data["inference"]["temperature"]


@pytest.mark.asyncio
async def test_update_profile_not_found(auth_client):
    """Test updating a non-existent profile."""
    response = await auth_client.put("/api/profiles/999", json={"name": "New Name"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_profile(auth_client, sample_profile_data):
    """Test deleting a profile."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Delete the profile
    response = await auth_client.delete(f"/api/profiles/{profile_id}")
    assert response.status_code == 204

    # Verify it's deleted
    get_response = await auth_client.get(f"/api/profiles/{profile_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_profile_not_found(auth_client):
    """Test deleting a non-existent profile."""
    response = await auth_client.delete("/api/profiles/999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_profile_duplicate_name(auth_client, sample_profile_data):
    """Test that creating a profile with duplicate name fails."""
    # Create first profile
    await auth_client.post("/api/profiles", json=sample_profile_data)

    # Try to create another with same name
    response = await auth_client.post("/api/profiles", json=sample_profile_data)
    assert response.status_code == 409
    assert "name already exists" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_profile_duplicate_name(
    auth_client, sample_profile_data, sample_profile_data_alt
):
    """Test that updating a profile to a duplicate name fails."""
    # Create two profiles
    await auth_client.post("/api/profiles", json=sample_profile_data)
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data_alt)
    profile_id = create_response.json()["id"]

    # Try to update second profile with first profile's name
    response = await auth_client.put(
        f"/api/profiles/{profile_id}",
        json={"name": sample_profile_data["name"]},
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_duplicate_profile(auth_client, sample_profile_data):
    """Test duplicating a profile."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Duplicate it
    response = await auth_client.post(
        f"/api/profiles/{profile_id}/duplicate?new_name=Duplicated%20Profile"
    )
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "Duplicated Profile"
    assert data["model_id"] == sample_profile_data["model_id"]
    assert data["model_repo_id"] == "mlx-community/test-model-4bit"
    assert data["inference"]["temperature"] == sample_profile_data["inference"]["temperature"]
    # Should have new ID
    assert data["id"] != profile_id


@pytest.mark.asyncio
async def test_duplicate_profile_not_found(auth_client):
    """Test duplicating a non-existent profile."""
    response = await auth_client.post("/api/profiles/999/duplicate?new_name=New%20Name")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_duplicate_profile_name_conflict(
    auth_client, sample_profile_data, sample_profile_data_alt
):
    """Test duplicating with an existing name fails."""
    # Create two profiles
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]
    await auth_client.post("/api/profiles", json=sample_profile_data_alt)

    # Try to duplicate with existing name
    response = await auth_client.post(
        f"/api/profiles/{profile_id}/duplicate?new_name={sample_profile_data_alt['name']}"
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_update_profile_all_fields(auth_client, sample_profile_data):
    """Test updating a profile with all fields."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Update with multiple fields
    update_data = {
        "name": "Updated Profile",
        "description": "Updated description",
        "model_id": 4,  # mlx-community/updated-model from fixtures
        "auto_start": True,
        "context": {"context_length": 4096},
        "inference": {"temperature": 0.5, "max_tokens": 2048, "top_p": 0.9},
    }
    response = await auth_client.put(f"/api/profiles/{profile_id}", json=update_data)
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Updated Profile"
    assert data["description"] == "Updated description"
    assert data["model_id"] == 4
    assert data["model_repo_id"] == "mlx-community/updated-model"
    assert data["auto_start"] is True
    assert data["context"]["context_length"] == 4096
    assert data["inference"]["temperature"] == 0.5
    assert data["inference"]["max_tokens"] == 2048
    assert data["inference"]["top_p"] == 0.9


@pytest.mark.asyncio
async def test_duplicate_profile_copies_all_fields(auth_client):
    """Test that duplicating a profile copies all fields correctly."""
    # Create a profile with many fields set
    profile_data = {
        "name": "Full Profile",
        "description": "Full description",
        "model_id": 3,  # mlx-community/full-model (vision) from fixtures
        "context": {"context_length": 8192, "system_prompt": "You are a helpful assistant."},
        "inference": {"temperature": 0.5, "max_tokens": 2048, "top_p": 0.9},
    }
    create_response = await auth_client.post("/api/profiles", json=profile_data)
    assert create_response.status_code == 201
    profile_id = create_response.json()["id"]

    # Duplicate
    response = await auth_client.post(
        f"/api/profiles/{profile_id}/duplicate?new_name=Duplicated%20Full%20Profile"
    )
    assert response.status_code == 201

    data = response.json()
    # Verify all fields were copied (except name, auto_start)
    assert data["name"] == "Duplicated Full Profile"
    assert data["description"] == profile_data["description"]
    assert data["model_id"] == profile_data["model_id"]
    assert data["model_repo_id"] == "mlx-community/full-model"
    assert data["model_type"] == "vision"
    assert data["context"]["context_length"] == profile_data["context"]["context_length"]
    assert data["context"]["system_prompt"] == profile_data["context"]["system_prompt"]
    assert data["inference"]["temperature"] == profile_data["inference"]["temperature"]
    assert data["inference"]["max_tokens"] == profile_data["inference"]["max_tokens"]
    assert data["inference"]["top_p"] == profile_data["inference"]["top_p"]
    assert data["auto_start"] is False  # auto_start should NOT be copied


@pytest.mark.asyncio
async def test_update_profile_same_name_allowed(auth_client, sample_profile_data):
    """Test that updating a profile with its own name is allowed."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Update with same name should succeed
    response = await auth_client.put(
        f"/api/profiles/{profile_id}",
        json={"name": sample_profile_data["name"], "description": "New description"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == sample_profile_data["name"]


@pytest.mark.asyncio
async def test_create_profile_validates_required_fields(auth_client):
    """Test that creating a profile without required fields fails."""
    # Missing model_id
    response = await auth_client.post("/api/profiles", json={"name": "Test"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_profiles_returns_all(auth_client, sample_profile_data, sample_profile_data_alt):
    """Test that list_profiles returns all profiles."""
    # Create two profiles
    await auth_client.post("/api/profiles", json=sample_profile_data)
    await auth_client.post("/api/profiles", json=sample_profile_data_alt)

    # List should return both
    response = await auth_client.get("/api/profiles")
    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) == 2
    names = {p["name"] for p in profiles}
    assert names == {sample_profile_data["name"], sample_profile_data_alt["name"]}


@pytest.mark.asyncio
async def test_profile_generation_parameters_defaults(auth_client):
    """Test that generation parameters have sensible defaults."""
    # Create profile with minimal data
    minimal_data = {
        "name": "Minimal Profile",
        "model_id": 1,
    }
    response = await auth_client.post("/api/profiles", json=minimal_data)
    assert response.status_code == 201

    data = response.json()
    # Generation params are now nullable (only meaningful for text/vision)
    assert data["inference"] is None or data["inference"]["temperature"] is None
    assert data["inference"] is None or data["inference"]["max_tokens"] is None
    assert data["inference"] is None or data["inference"]["top_p"] is None


@pytest.mark.asyncio
async def test_profile_generation_parameters_validation(auth_client):
    """Test that generation parameters are validated."""
    # Temperature out of range
    invalid_data = {
        "name": "Invalid Profile",
        "model_id": 1,
        "inference": {"temperature": 3.0},  # Max is 2.0
    }
    response = await auth_client.post("/api/profiles", json=invalid_data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_profile_invalid_model_id(auth_client):
    """Test that creating a profile with non-existent model_id fails."""
    data = {
        "name": "Bad Model Profile",
        "model_id": 999,
    }
    response = await auth_client.post("/api/profiles", json=data)
    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"


@pytest.mark.asyncio
async def test_update_profile_invalid_model_id(auth_client, sample_profile_data):
    """Test that updating a profile with non-existent model_id fails."""
    # Create a profile
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Try to update with invalid model_id
    response = await auth_client.put(
        f"/api/profiles/{profile_id}",
        json={"model_id": 999},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"
