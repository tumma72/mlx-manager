"""Direct unit tests for profiles router functions.

These tests call router functions directly with mock sessions to ensure
coverage is properly tracked (avoiding ASGI transport coverage issues).
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from mlx_manager.models import (
    ExecutionProfile,
    ExecutionProfileCreate,
    ExecutionProfileResponse,
    ExecutionProfileUpdate,
    Model,
    User,
    UserStatus,
)
from mlx_manager.routers.profiles import (
    create_profile,
    duplicate_profile,
    list_profiles,
    update_profile,
)


# Helper to create a mock user for auth
def create_mock_user():
    """Create a mock user for testing."""
    return User(
        id=1,
        email="test@example.com",
        hashed_password="hashed",
        is_admin=False,
        status=UserStatus.APPROVED,
    )


def _create_mock_model(model_id=1, repo_id="mlx-community/test-model", model_type="text-gen"):
    """Create a mock Model for testing."""
    model = MagicMock(spec=Model)
    model.id = model_id
    model.repo_id = repo_id
    model.model_type = model_type
    return model


def _create_mock_profile(
    profile_id=1,
    name="Test Profile",
    model_id=1,
    model=None,
):
    """Create a mock ExecutionProfile with model relationship for testing."""
    mock_model = model or _create_mock_model(model_id=model_id)
    profile = MagicMock(spec=ExecutionProfile)
    profile.id = profile_id
    profile.name = name
    profile.description = None
    profile.model_id = model_id
    profile.model = mock_model
    profile.profile_type = "inference"
    profile.default_context_length = None
    profile.auto_start = False
    profile.default_system_prompt = None
    profile.default_temperature = None
    profile.default_max_tokens = None
    profile.default_top_p = None
    profile.default_enable_tool_injection = False
    profile.default_tts_voice = None
    profile.default_tts_speed = None
    profile.default_tts_sample_rate = None
    profile.default_stt_language = None
    profile.launchd_installed = False
    profile.created_at = datetime.now(tz=UTC)
    profile.updated_at = datetime.now(tz=UTC)
    return profile


def _mock_response(profile_id=1, name="Test Profile"):
    """Create a mock ExecutionProfileResponse."""
    return ExecutionProfileResponse(
        id=profile_id,
        name=name,
        model_id=1,
        model_repo_id="mlx-community/test-model",
        model_type="text-gen",
        profile_type="inference",
        auto_start=False,
        launchd_installed=False,
        created_at=datetime.now(tz=UTC),
        updated_at=datetime.now(tz=UTC),
    )


class TestListProfilesDirect:
    """Direct tests for list_profiles function."""

    @pytest.mark.asyncio
    async def test_list_profiles_returns_all(self):
        """Test list_profiles returns all profiles from database."""
        mock_user = create_mock_user()

        # Create mock profiles with model relationships
        mock_profiles = [
            _create_mock_profile(profile_id=1, name="Profile 1"),
            _create_mock_profile(profile_id=2, name="Profile 2"),
        ]

        # Setup mock session
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_profiles
        mock_session.execute.return_value = mock_result

        # Call function directly
        result = await list_profiles(current_user=mock_user, session=mock_session)

        assert len(result) == 2
        mock_session.execute.assert_called_once()


class TestCreateProfileDirect:
    """Direct tests for create_profile function."""

    @pytest.mark.asyncio
    async def test_create_profile_success(self):
        """Test create_profile creates and returns new profile."""
        mock_user = create_mock_user()

        profile_data = ExecutionProfileCreate(
            name="Test Profile",
            model_id=1,
        )

        mock_model = _create_mock_model()
        mock_session = AsyncMock()
        # session.add is synchronous, so use MagicMock
        mock_session.add = MagicMock()
        # Query for name conflict - none
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        # session.get for model validation
        mock_session.get.return_value = mock_model

        # Create a side effect that sets the ID on the profile when add is called
        def set_profile_id(profile):
            # Set attributes directly on the instance without going through SQLAlchemy
            object.__setattr__(profile, "id", 1)

        mock_session.add.side_effect = set_profile_id

        mock_resp = _mock_response(name="Test Profile")
        with patch("mlx_manager.models.profiles.profile_to_response", return_value=mock_resp):
            result = await create_profile(
                current_user=mock_user, profile_data=profile_data, session=mock_session
            )

        assert result.name == "Test Profile"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_profile_name_conflict(self):
        """Test create_profile raises 409 on name conflict."""
        mock_user = create_mock_user()

        profile_data = ExecutionProfileCreate(
            name="Existing",
            model_id=1,
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()  # Existing profile
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await create_profile(
                current_user=mock_user, profile_data=profile_data, session=mock_session
            )

        assert exc_info.value.status_code == 409
        assert "name already exists" in exc_info.value.detail


class TestUpdateProfileDirect:
    """Direct tests for update_profile function."""

    @pytest.mark.asyncio
    async def test_update_profile_success(self):
        """Test update_profile updates and returns profile."""
        mock_user = create_mock_user()

        existing_profile = _create_mock_profile(profile_id=1, name="Old Name")

        profile_data = ExecutionProfileUpdate(name="New Name")

        mock_session = AsyncMock()
        # session.add is synchronous, so use MagicMock
        mock_session.add = MagicMock()
        # First query - find existing profile
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = existing_profile
        # Second query - check name conflict (none)
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = None
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        mock_resp = _mock_response(name="New Name")
        with patch("mlx_manager.models.profiles.profile_to_response", return_value=mock_resp):
            result = await update_profile(
                current_user=mock_user,
                profile_id=1,
                profile_data=profile_data,
                session=mock_session,
            )

        assert result.name == "New Name"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_profile_not_found(self):
        """Test update_profile raises 404 when profile not found."""
        mock_user = create_mock_user()

        profile_data = ExecutionProfileUpdate(name="New Name")

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await update_profile(
                current_user=mock_user,
                profile_id=999,
                profile_data=profile_data,
                session=mock_session,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_update_profile_name_conflict(self):
        """Test update_profile raises 409 on name conflict."""
        mock_user = create_mock_user()

        existing_profile = _create_mock_profile(profile_id=1, name="Old Name")

        profile_data = ExecutionProfileUpdate(name="Taken Name")

        mock_session = AsyncMock()
        # First query - find existing profile
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = existing_profile
        # Second query - name conflict!
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = MagicMock()
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        with pytest.raises(HTTPException) as exc_info:
            await update_profile(
                current_user=mock_user,
                profile_id=1,
                profile_data=profile_data,
                session=mock_session,
            )

        assert exc_info.value.status_code == 409
        assert "name already exists" in exc_info.value.detail


class TestDuplicateProfileDirect:
    """Direct tests for duplicate_profile function."""

    @pytest.mark.asyncio
    async def test_duplicate_profile_success(self):
        """Test duplicate_profile creates copy with new name."""
        mock_user = create_mock_user()

        existing_profile = _create_mock_profile(
            profile_id=1,
            name="Original",
        )

        mock_session = AsyncMock()
        # session.add is synchronous, so use MagicMock
        mock_session.add = MagicMock()
        # Query - check name conflict (none)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Create a side effect that sets the ID on the profile when add is called
        def set_profile_id(profile):
            # Set attributes directly on the instance without going through SQLAlchemy
            object.__setattr__(profile, "id", 2)

        mock_session.add.side_effect = set_profile_id

        mock_resp = _mock_response(profile_id=2, name="Copy")
        with patch("mlx_manager.models.profiles.profile_to_response", return_value=mock_resp):
            result = await duplicate_profile(
                current_user=mock_user,
                new_name="Copy",
                profile=existing_profile,
                session=mock_session,
            )

        assert result.name == "Copy"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_profile_name_conflict(self):
        """Test duplicate_profile raises 409 on name conflict."""
        mock_user = create_mock_user()

        existing_profile = _create_mock_profile(
            profile_id=1,
            name="Original",
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()  # Name exists
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await duplicate_profile(
                current_user=mock_user,
                new_name="Existing",
                profile=existing_profile,
                session=mock_session,
            )

        assert exc_info.value.status_code == 409
