"""Direct unit tests for profiles router functions.

These tests call router functions directly with mock sessions to ensure
coverage is properly tracked (avoiding ASGI transport coverage issues).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from mlx_manager.models import ServerProfile, ServerProfileCreate, ServerProfileUpdate
from mlx_manager.routers.profiles import (
    create_profile,
    duplicate_profile,
    get_next_port,
    list_profiles,
    update_profile,
)


class TestListProfilesDirect:
    """Direct tests for list_profiles function."""

    @pytest.mark.asyncio
    async def test_list_profiles_returns_all(self):
        """Test list_profiles returns all profiles from database."""
        # Create mock profiles
        mock_profiles = [
            MagicMock(spec=ServerProfile, id=1, name="Profile 1"),
            MagicMock(spec=ServerProfile, id=2, name="Profile 2"),
        ]

        # Setup mock session
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_profiles
        mock_session.execute.return_value = mock_result

        # Call function directly
        result = await list_profiles(session=mock_session)

        assert result == mock_profiles
        mock_session.execute.assert_called_once()


class TestGetNextPortDirect:
    """Direct tests for get_next_port function."""

    @pytest.mark.asyncio
    async def test_get_next_port_with_existing_ports(self):
        """Test get_next_port calculates next port correctly."""
        # Setup mock session with existing ports
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [10240, 10241, 10243]
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.profiles.settings") as mock_settings:
            mock_settings.default_port_start = 10240

            result = await get_next_port(session=mock_session)

        # Should return 10244 (after 10243)
        assert result == {"port": 10244}

    @pytest.mark.asyncio
    async def test_get_next_port_empty_db(self):
        """Test get_next_port returns default when no profiles exist."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        with patch("mlx_manager.routers.profiles.settings") as mock_settings:
            mock_settings.default_port_start = 10240

            result = await get_next_port(session=mock_session)

        assert result == {"port": 10240}


class TestCreateProfileDirect:
    """Direct tests for create_profile function."""

    @pytest.mark.asyncio
    async def test_create_profile_success(self):
        """Test create_profile creates and returns new profile."""
        profile_data = ServerProfileCreate(
            name="Test Profile",
            model_path="mlx-community/test-model",
            model_type="lm",
            port=10240,
        )

        mock_session = AsyncMock()
        # First query for name - no conflict
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = None
        # Second query for port - no conflict
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = None
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        result = await create_profile(profile_data=profile_data, session=mock_session)

        assert result.name == "Test Profile"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_profile_name_conflict(self):
        """Test create_profile raises 409 on name conflict."""
        profile_data = ServerProfileCreate(
            name="Existing",
            model_path="mlx-community/test-model",
            model_type="lm",
            port=10240,
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()  # Existing profile
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await create_profile(profile_data=profile_data, session=mock_session)

        assert exc_info.value.status_code == 409
        assert "name already exists" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_create_profile_port_conflict(self):
        """Test create_profile raises 409 on port conflict."""
        profile_data = ServerProfileCreate(
            name="New Profile",
            model_path="mlx-community/test-model",
            model_type="lm",
            port=10240,
        )

        mock_session = AsyncMock()
        # First query for name - no conflict
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = None
        # Second query for port - conflict!
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = MagicMock()
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        with pytest.raises(HTTPException) as exc_info:
            await create_profile(profile_data=profile_data, session=mock_session)

        assert exc_info.value.status_code == 409
        assert "Port already in use" in exc_info.value.detail


class TestUpdateProfileDirect:
    """Direct tests for update_profile function."""

    @pytest.mark.asyncio
    async def test_update_profile_success(self):
        """Test update_profile updates and returns profile."""
        existing_profile = ServerProfile(
            id=1,
            name="Old Name",
            model_path="mlx-community/test-model",
            model_type="lm",
            port=10240,
        )

        profile_data = ServerProfileUpdate(name="New Name")

        mock_session = AsyncMock()
        # First query - find existing profile
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = existing_profile
        # Second query - check name conflict (none)
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = None
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        result = await update_profile(profile_id=1, profile_data=profile_data, session=mock_session)

        assert result.name == "New Name"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_profile_not_found(self):
        """Test update_profile raises 404 when profile not found."""
        profile_data = ServerProfileUpdate(name="New Name")

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await update_profile(profile_id=999, profile_data=profile_data, session=mock_session)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_update_profile_name_conflict(self):
        """Test update_profile raises 409 on name conflict."""
        existing_profile = ServerProfile(
            id=1,
            name="Old Name",
            model_path="mlx-community/test-model",
            model_type="lm",
            port=10240,
        )

        profile_data = ServerProfileUpdate(name="Taken Name")

        mock_session = AsyncMock()
        # First query - find existing profile
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = existing_profile
        # Second query - name conflict!
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = MagicMock()
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        with pytest.raises(HTTPException) as exc_info:
            await update_profile(profile_id=1, profile_data=profile_data, session=mock_session)

        assert exc_info.value.status_code == 409
        assert "name already exists" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_update_profile_port_conflict(self):
        """Test update_profile raises 409 on port conflict."""
        existing_profile = ServerProfile(
            id=1,
            name="Profile",
            model_path="mlx-community/test-model",
            model_type="lm",
            port=10240,
        )

        profile_data = ServerProfileUpdate(port=10241)

        mock_session = AsyncMock()
        # First query - find existing profile
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = existing_profile
        # Second query - port conflict!
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = MagicMock()
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        with pytest.raises(HTTPException) as exc_info:
            await update_profile(profile_id=1, profile_data=profile_data, session=mock_session)

        assert exc_info.value.status_code == 409
        assert "Port already in use" in exc_info.value.detail


class TestDuplicateProfileDirect:
    """Direct tests for duplicate_profile function."""

    @pytest.mark.asyncio
    async def test_duplicate_profile_success(self):
        """Test duplicate_profile creates copy with new name."""
        existing_profile = ServerProfile(
            id=1,
            name="Original",
            description="A test profile",
            model_path="mlx-community/test-model",
            model_type="lm",
            port=10240,
            host="127.0.0.1",
        )

        mock_session = AsyncMock()
        # First query - check name conflict (none)
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = None
        # Second query - get existing ports
        mock_result2 = MagicMock()
        mock_result2.scalars.return_value.all.return_value = [10240]
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        with patch("mlx_manager.routers.profiles.settings") as mock_settings:
            mock_settings.default_port_start = 10240

            result = await duplicate_profile(
                new_name="Copy", profile=existing_profile, session=mock_session
            )

        assert result.name == "Copy"
        assert result.port == 10241
        assert result.model_path == existing_profile.model_path
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_profile_name_conflict(self):
        """Test duplicate_profile raises 409 on name conflict."""
        existing_profile = ServerProfile(
            id=1,
            name="Original",
            model_path="mlx-community/test-model",
            model_type="lm",
            port=10240,
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()  # Name exists
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await duplicate_profile(
                new_name="Existing", profile=existing_profile, session=mock_session
            )

        assert exc_info.value.status_code == 409
