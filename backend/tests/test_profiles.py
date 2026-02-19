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


@pytest.mark.asyncio
async def test_create_profile_with_model_options(auth_client, sample_profile_data):
    """Test creating a profile with model_options."""
    profile_data = sample_profile_data.copy()
    profile_data["model_options"] = {"enable_thinking": False, "custom_param": "value"}

    response = await auth_client.post("/api/profiles", json=profile_data)
    assert response.status_code == 201

    data = response.json()
    assert data["model_options"] == {"enable_thinking": False, "custom_param": "value"}
    assert data["name"] == profile_data["name"]
    assert data["model_id"] == profile_data["model_id"]


@pytest.mark.asyncio
async def test_update_profile_model_options(auth_client, sample_profile_data):
    """Test updating a profile's model_options."""
    # Create a profile without model_options
    create_response = await auth_client.post("/api/profiles", json=sample_profile_data)
    profile_id = create_response.json()["id"]

    # Update with model_options
    update_data = {"model_options": {"enable_thinking": True, "max_thinking_tokens": 1000}}
    response = await auth_client.put(f"/api/profiles/{profile_id}", json=update_data)
    assert response.status_code == 200

    data = response.json()
    assert data["model_options"] == {"enable_thinking": True, "max_thinking_tokens": 1000}
    # Other fields should remain unchanged
    assert data["name"] == sample_profile_data["name"]
    assert data["model_id"] == sample_profile_data["model_id"]

    # Update model_options again with different values
    update_data = {"model_options": {"enable_thinking": False}}
    response = await auth_client.put(f"/api/profiles/{profile_id}", json=update_data)
    assert response.status_code == 200

    data = response.json()
    assert data["model_options"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_create_profile_without_model_options(auth_client, sample_profile_data):
    """Test creating a profile without model_options (backward compatibility)."""
    # Ensure sample_profile_data doesn't have model_options
    profile_data = sample_profile_data.copy()
    profile_data.pop("model_options", None)

    response = await auth_client.post("/api/profiles", json=profile_data)
    assert response.status_code == 201

    data = response.json()
    # model_options should be null when not provided
    assert data["model_options"] is None
    assert data["name"] == profile_data["name"]
    assert data["model_id"] == profile_data["model_id"]


@pytest.mark.asyncio
async def test_duplicate_profile_with_model_options(auth_client):
    """Test that duplicating a profile copies model_options correctly."""
    # Create a profile with model_options
    profile_data = {
        "name": "Profile with Options",
        "description": "Test description",
        "model_id": 1,
        "inference": {"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9},
        "model_options": {"enable_thinking": True, "custom_setting": "test"},
    }
    create_response = await auth_client.post("/api/profiles", json=profile_data)
    assert create_response.status_code == 201
    profile_id = create_response.json()["id"]

    # Duplicate the profile
    response = await auth_client.post(
        f"/api/profiles/{profile_id}/duplicate?new_name=Duplicated%20Options%20Profile"
    )
    assert response.status_code == 201

    data = response.json()
    # Verify model_options was copied
    assert data["model_options"] == {"enable_thinking": True, "custom_setting": "test"}
    assert data["name"] == "Duplicated Options Profile"
    assert data["model_id"] == profile_data["model_id"]
    assert data["inference"]["temperature"] == profile_data["inference"]["temperature"]
    # Should have new ID
    assert data["id"] != profile_id


@pytest.mark.asyncio
async def test_update_profile_clear_model_options(auth_client, sample_profile_data):
    """Test clearing model_options by setting to empty dict."""
    # Create a profile with model_options
    profile_data = sample_profile_data.copy()
    profile_data["model_options"] = {"enable_thinking": True}

    create_response = await auth_client.post("/api/profiles", json=profile_data)
    profile_id = create_response.json()["id"]

    # Clear model_options with empty dict
    update_data = {"model_options": {}}
    response = await auth_client.put(f"/api/profiles/{profile_id}", json=update_data)
    assert response.status_code == 200

    data = response.json()
    # Empty dict should be stored as None
    assert data["model_options"] is None


# ============================================================================
# Audio Profile Tests (uncovered lines 219-220, 226, 249-254)
# ============================================================================


@pytest.mark.asyncio
async def test_create_audio_profile_with_audio_params(auth_client):
    """Test creating an audio profile sets audio params correctly (lines 249-254).

    Creates an audio model in the DB, then creates a profile with audio defaults.
    """

    from mlx_manager.database import get_db
    from mlx_manager.main import app
    from mlx_manager.models import Model

    # Add an audio model to the DB by calling the session directly via the override
    audio_model_id = None
    db_override = app.dependency_overrides.get(get_db)
    if db_override is not None:
        async for session in db_override():
            audio_model = Model(
                repo_id="mlx-community/Kokoro-82M-4bit",
                model_type="audio",
                local_path="/fake/path/to/kokoro",
            )
            session.add(audio_model)
            await session.commit()
            await session.refresh(audio_model)
            audio_model_id = audio_model.id
            break

    if audio_model_id is None:
        pytest.skip("Could not create audio model via DB override")

    # Create an audio profile with audio defaults
    profile_data = {
        "name": "Audio Profile",
        "model_id": audio_model_id,
        "audio": {
            "tts_voice": "af_sky",
            "tts_speed": 1.0,
            "tts_sample_rate": 24000,
            "stt_language": "en",
        },
    }
    response = await auth_client.post("/api/profiles", json=profile_data)
    assert response.status_code == 201

    data = response.json()
    assert data["profile_type"] == "audio"
    assert data["audio"] is not None
    assert data["audio"]["tts_voice"] == "af_sky"
    assert data["audio"]["tts_speed"] == 1.0
    assert data["audio"]["tts_sample_rate"] == 24000
    assert data["audio"]["stt_language"] == "en"
    assert data["inference"] is None
    assert data["context"] is None


@pytest.mark.asyncio
async def test_audio_profile_rejects_inference_params(auth_client):
    """Test that audio profile with inference params raises 422 (lines 219-220)."""
    from mlx_manager.database import get_db
    from mlx_manager.main import app
    from mlx_manager.models import Model

    # Add an audio model to the DB
    audio_model_id = None
    db_override = app.dependency_overrides.get(get_db)
    if db_override is not None:
        async for session in db_override():
            audio_model = Model(
                repo_id="mlx-community/Kokoro-audio-test",
                model_type="audio",
                local_path="/fake/path/to/audio",
            )
            session.add(audio_model)
            await session.commit()
            await session.refresh(audio_model)
            audio_model_id = audio_model.id
            break

    if audio_model_id is None:
        pytest.skip("Could not create audio model via DB override")

    # Try to create audio profile with inference params - should fail
    profile_data = {
        "name": "Bad Audio Profile",
        "model_id": audio_model_id,
        "inference": {"temperature": 0.7, "max_tokens": 100},
    }
    response = await auth_client.post("/api/profiles", json=profile_data)
    assert response.status_code == 422
    assert "Audio profiles cannot have inference" in response.json()["detail"]


@pytest.mark.asyncio
async def test_audio_profile_rejects_context_params(auth_client):
    """Test that audio profile with context params raises 422 (lines 219-220)."""
    from mlx_manager.database import get_db
    from mlx_manager.main import app
    from mlx_manager.models import Model

    # Add an audio model to the DB
    audio_model_id = None
    db_override = app.dependency_overrides.get(get_db)
    if db_override is not None:
        async for session in db_override():
            audio_model = Model(
                repo_id="mlx-community/Kokoro-audio-ctx-test",
                model_type="audio",
                local_path="/fake/path/to/audio-ctx",
            )
            session.add(audio_model)
            await session.commit()
            await session.refresh(audio_model)
            audio_model_id = audio_model.id
            break

    if audio_model_id is None:
        pytest.skip("Could not create audio model via DB override")

    # Try to create audio profile with context params - should fail
    profile_data = {
        "name": "Bad Audio Context Profile",
        "model_id": audio_model_id,
        "context": {"system_prompt": "You are a TTS system."},
    }
    response = await auth_client.post("/api/profiles", json=profile_data)
    assert response.status_code == 422
    assert "Audio profiles cannot have inference" in response.json()["detail"]


@pytest.mark.asyncio
async def test_inference_profile_rejects_audio_params(auth_client):
    """Test that inference profile with audio params raises 422 (line 226)."""
    # model_id=1 is text-gen which becomes inference profile type
    profile_data = {
        "name": "Bad Inference Profile",
        "model_id": 1,
        "audio": {
            "tts_voice": "af_sky",
        },
    }
    response = await auth_client.post("/api/profiles", json=profile_data)
    assert response.status_code == 422
    assert "Inference profiles cannot have audio" in response.json()["detail"]


@pytest.mark.asyncio
async def test_vision_profile_rejects_audio_params(auth_client):
    """Test that vision profile (inference type) with audio params raises 422 (line 226)."""
    # model_id=3 is vision which becomes inference profile type
    profile_data = {
        "name": "Bad Vision Profile",
        "model_id": 3,
        "audio": {
            "tts_voice": "voice1",
        },
    }
    response = await auth_client.post("/api/profiles", json=profile_data)
    assert response.status_code == 422
    assert "Inference profiles cannot have audio" in response.json()["detail"]


# ============================================================================
# _validate_profile_fields and _apply_dto_to_entity - Direct function tests
# ============================================================================


def test_validate_profile_fields_audio_with_inference_raises():
    """_validate_profile_fields raises 422 when audio profile has inference params (lines 219-220)."""
    from fastapi import HTTPException

    from mlx_manager.models.profiles import ExecutionProfileCreate
    from mlx_manager.models.value_objects import InferenceParams
    from mlx_manager.routers.profiles import _validate_profile_fields

    dto = ExecutionProfileCreate(
        name="test",
        model_id=1,
        inference=InferenceParams(temperature=0.7),
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_profile_fields("audio", dto)
    assert exc_info.value.status_code == 422
    assert "Audio profiles" in exc_info.value.detail


def test_validate_profile_fields_audio_with_context_raises():
    """_validate_profile_fields raises 422 when audio profile has context params (lines 219-220)."""
    from fastapi import HTTPException

    from mlx_manager.models.profiles import ExecutionProfileCreate
    from mlx_manager.models.value_objects import InferenceContext
    from mlx_manager.routers.profiles import _validate_profile_fields

    dto = ExecutionProfileCreate(
        name="test",
        model_id=1,
        context=InferenceContext(system_prompt="hello"),
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_profile_fields("audio", dto)
    assert exc_info.value.status_code == 422
    assert "Audio profiles" in exc_info.value.detail


def test_validate_profile_fields_inference_with_audio_raises():
    """_validate_profile_fields raises 422 when inference profile has audio params (line 226)."""
    from fastapi import HTTPException

    from mlx_manager.models.profiles import ExecutionProfileCreate
    from mlx_manager.models.value_objects import AudioDefaults
    from mlx_manager.routers.profiles import _validate_profile_fields

    dto = ExecutionProfileCreate(
        name="test",
        model_id=1,
        audio=AudioDefaults(tts_voice="voice1"),
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_profile_fields("inference", dto)
    assert exc_info.value.status_code == 422
    assert "Inference profiles" in exc_info.value.detail


def test_validate_profile_fields_base_with_audio_raises():
    """_validate_profile_fields raises 422 when base profile has audio params (line 226)."""
    from fastapi import HTTPException

    from mlx_manager.models.profiles import ExecutionProfileCreate
    from mlx_manager.models.value_objects import AudioDefaults
    from mlx_manager.routers.profiles import _validate_profile_fields

    dto = ExecutionProfileCreate(
        name="test",
        model_id=1,
        audio=AudioDefaults(tts_voice="voice1"),
    )
    with pytest.raises(HTTPException) as exc_info:
        _validate_profile_fields("base", dto)
    assert exc_info.value.status_code == 422
    assert "Inference profiles" in exc_info.value.detail


def test_apply_dto_to_entity_audio_sets_tts_stt_fields():
    """_apply_dto_to_entity sets tts/stt fields on audio profile (lines 249-254)."""
    from mlx_manager.models.profiles import ExecutionProfile, ExecutionProfileCreate
    from mlx_manager.models.value_objects import AudioDefaults
    from mlx_manager.routers.profiles import _apply_dto_to_entity

    profile = ExecutionProfile(
        name="audio-profile",
        model_id=1,
        profile_type="audio",
    )
    dto = ExecutionProfileCreate(
        name="audio-profile",
        model_id=1,
        audio=AudioDefaults(
            tts_voice="af_sky",
            tts_speed=1.2,
            tts_sample_rate=22050,
            stt_language="en",
        ),
    )
    _apply_dto_to_entity(profile, dto, "audio")

    assert profile.default_tts_voice == "af_sky"
    assert profile.default_tts_speed == 1.2
    assert profile.default_tts_sample_rate == 22050
    assert profile.default_stt_language == "en"


def test_apply_dto_to_entity_audio_with_none_audio_does_not_set():
    """_apply_dto_to_entity does not set tts/stt fields when audio is None."""
    from mlx_manager.models.profiles import ExecutionProfile, ExecutionProfileCreate
    from mlx_manager.routers.profiles import _apply_dto_to_entity

    profile = ExecutionProfile(
        name="audio-profile",
        model_id=1,
        profile_type="audio",
        default_tts_voice="original_voice",
    )
    dto = ExecutionProfileCreate(
        name="audio-profile",
        model_id=1,
        audio=None,
    )
    _apply_dto_to_entity(profile, dto, "audio")

    # Fields should remain unchanged
    assert profile.default_tts_voice == "original_voice"
