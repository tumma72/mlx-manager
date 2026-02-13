"""Tests for model capabilities API endpoints."""

import pytest

from mlx_manager.models import Model
from mlx_manager.models.capabilities import ModelCapabilities


async def _create_model_with_caps(
    session, repo_id: str, capability_type: str = "text-gen", **cap_fields
) -> tuple[Model, ModelCapabilities]:
    """Helper to create a Model + linked ModelCapabilities record."""
    from datetime import UTC, datetime

    model = Model(repo_id=repo_id, model_type=capability_type)
    session.add(model)
    await session.flush()  # Get the ID

    caps = ModelCapabilities(
        model_id=model.id,
        capability_type=capability_type,
        probed_at=datetime.now(tz=UTC),
        **cap_fields,
    )
    session.add(caps)
    await session.commit()
    return model, caps


@pytest.mark.asyncio
async def test_get_all_capabilities_empty(client, test_user, auth_headers):
    """Test getting capabilities when none exist."""
    response = await client.get("/api/models/capabilities", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_capabilities_not_found(client, test_user, auth_headers):
    """Test getting capabilities for unprobed model."""
    response = await client.get(
        "/api/models/capabilities/some/model",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.asyncio
async def test_get_capabilities_after_insert(client, test_user, auth_headers, test_session):
    """Test getting capabilities after manual DB insert."""
    await _create_model_with_caps(
        test_session,
        repo_id="test/model",
        supports_native_tools=True,
        supports_thinking=True,
        practical_max_tokens=4096,
    )

    response = await client.get(
        "/api/models/capabilities/test/model",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == "test/model"
    assert data["model_id"] == "test/model"  # Frontend-compatible alias
    assert data["supports_native_tools"] is True
    assert data["supports_thinking"] is True
    assert data["practical_max_tokens"] == 4096


@pytest.mark.asyncio
async def test_get_all_capabilities(client, test_user, auth_headers, test_session):
    """Test listing all capabilities."""
    await _create_model_with_caps(
        test_session,
        repo_id="model/a",
        supports_native_tools=True,
        supports_thinking=False,
        practical_max_tokens=8192,
    )
    await _create_model_with_caps(
        test_session,
        repo_id="model/b",
        supports_native_tools=False,
        supports_thinking=True,
        practical_max_tokens=16384,
    )

    response = await client.get("/api/models/capabilities", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    # Check both entries exist (via both repo_id and model_id alias)
    repo_ids = {item["repo_id"] for item in data}
    assert "model/a" in repo_ids
    assert "model/b" in repo_ids
    model_ids = {item["model_id"] for item in data}
    assert "model/a" in model_ids
    assert "model/b" in model_ids


@pytest.mark.asyncio
async def test_get_capabilities_with_all_fields(client, test_user, auth_headers, test_session):
    """Test getting capabilities with all fields populated."""
    await _create_model_with_caps(
        test_session,
        repo_id="test/full-model",
        supports_native_tools=True,
        supports_thinking=True,
        practical_max_tokens=32768,
    )

    response = await client.get(
        "/api/models/capabilities/test/full-model",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == "test/full-model"
    assert data["model_id"] == "test/full-model"  # Frontend-compatible alias
    assert data["supports_native_tools"] is True
    assert data["supports_thinking"] is True
    assert data["practical_max_tokens"] == 32768
    assert "probed_at" in data
    assert data["probed_at"] is not None


@pytest.mark.asyncio
async def test_get_capabilities_with_partial_fields(client, test_user, auth_headers, test_session):
    """Test getting capabilities with some fields as None."""
    await _create_model_with_caps(
        test_session,
        repo_id="test/partial-model",
        supports_native_tools=None,
        supports_thinking=True,
        practical_max_tokens=None,
    )

    response = await client.get(
        "/api/models/capabilities/test/partial-model",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == "test/partial-model"
    assert data["supports_native_tools"] is None
    assert data["supports_thinking"] is True
    assert data["practical_max_tokens"] is None


@pytest.mark.asyncio
async def test_get_capabilities_requires_auth(client):
    """Test that capabilities endpoints require authentication."""
    response = await client.get("/api/models/capabilities")
    assert response.status_code == 401

    response = await client.get("/api/models/capabilities/some/model")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_probe_endpoint_requires_auth(client):
    """Test that probe endpoint requires authentication via token query param.

    Returns 422 (validation error) when token is missing, since the endpoint
    uses get_current_user_from_token which expects a token query parameter.
    """
    response = await client.post("/api/models/probe/test/model")
    assert response.status_code == 422  # Missing required token query param


@pytest.mark.asyncio
async def test_get_capabilities_url_encoding(client, test_user, auth_headers, test_session):
    """Test getting capabilities for model with slashes in ID."""
    await _create_model_with_caps(
        test_session,
        repo_id="mlx-community/Qwen3-0.6B-4bit-DWQ",
        supports_native_tools=True,
        supports_thinking=False,
        practical_max_tokens=2048,
    )

    # Test with URL-encoded path
    response = await client.get(
        "/api/models/capabilities/mlx-community/Qwen3-0.6B-4bit-DWQ",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == "mlx-community/Qwen3-0.6B-4bit-DWQ"
    assert data["model_id"] == "mlx-community/Qwen3-0.6B-4bit-DWQ"


@pytest.mark.asyncio
async def test_get_all_capabilities_empty_after_delete(
    client, test_user, auth_headers, test_session
):
    """Test that deleted capabilities don't appear in list."""
    model, caps = await _create_model_with_caps(
        test_session,
        repo_id="test/deleted",
        supports_native_tools=True,
    )

    # Verify it exists
    response = await client.get("/api/models/capabilities", headers=auth_headers)
    assert len(response.json()) == 1

    # Delete the model (cascades to capabilities)
    await test_session.delete(model)
    await test_session.commit()

    # Verify it's gone
    response = await client.get("/api/models/capabilities", headers=auth_headers)
    assert response.json() == []
