"""Tests for FastAPI dependencies."""

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

# Ensure test environment is set before importing app modules
os.environ["MLX_MANAGER_DATABASE_PATH"] = ":memory:"
os.environ["MLX_MANAGER_DEBUG"] = "false"

from mlx_manager.dependencies import get_profile_or_404
from mlx_manager.models import ServerProfile


class TestGetProfileOr404:
    """Tests for the get_profile_or_404 dependency."""

    @pytest.mark.asyncio
    async def test_returns_profile_when_exists(self, test_session, test_models):
        """Test returns profile when it exists in database."""
        # Create a profile
        profile = ServerProfile(
            name="Test Profile",
            model_id=test_models["model1"].id,
        )
        test_session.add(profile)
        await test_session.commit()
        await test_session.refresh(profile)

        # Get the profile
        result = await get_profile_or_404(profile.id, test_session)

        assert result is not None
        assert result.id == profile.id
        assert result.name == "Test Profile"
        assert result.model_id == test_models["model1"].id

    @pytest.mark.asyncio
    async def test_raises_404_when_not_exists(self, test_session):
        """Test raises HTTPException 404 when profile doesn't exist."""
        with pytest.raises(HTTPException) as exc_info:
            await get_profile_or_404(999, test_session)

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Profile not found"

    @pytest.mark.asyncio
    async def test_returns_correct_profile_among_many(self, test_session, test_models):
        """Test returns the correct profile when multiple exist."""
        # Create multiple profiles
        profiles = []
        for i in range(5):
            profile = ServerProfile(
                name=f"Profile {i}",
                model_id=test_models["model1"].id,
            )
            test_session.add(profile)
            profiles.append(profile)

        await test_session.commit()

        # Refresh to get IDs
        for profile in profiles:
            await test_session.refresh(profile)

        # Get a specific profile (the third one)
        target_profile = profiles[2]
        result = await get_profile_or_404(target_profile.id, test_session)

        assert result.id == target_profile.id
        assert result.name == "Profile 2"

    @pytest.mark.asyncio
    async def test_raises_404_for_zero_id(self, test_session):
        """Test raises 404 for profile ID 0."""
        with pytest.raises(HTTPException) as exc_info:
            await get_profile_or_404(0, test_session)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_raises_404_for_negative_id(self, test_session):
        """Test raises 404 for negative profile ID."""
        with pytest.raises(HTTPException) as exc_info:
            await get_profile_or_404(-1, test_session)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_all_profile_fields(self, test_session, test_models):
        """Test returns profile with all fields populated."""
        # Create a profile with all current fields
        profile = ServerProfile(
            name="Complete Profile",
            description="A complete test profile",
            model_id=test_models["model1"].id,
            context_length=4096,
            auto_start=True,
            system_prompt="You are a helpful assistant.",
            # Generation parameters
            temperature=0.8,
            max_tokens=8192,
            top_p=0.9,
        )
        test_session.add(profile)
        await test_session.commit()
        await test_session.refresh(profile)

        # Get the profile
        result = await get_profile_or_404(profile.id, test_session)

        # Verify all fields
        assert result.name == "Complete Profile"
        assert result.description == "A complete test profile"
        assert result.model_id == test_models["model1"].id
        assert result.context_length == 4096
        assert result.auto_start is True
        assert result.system_prompt == "You are a helpful assistant."
        # Generation parameters
        assert result.temperature == 0.8
        assert result.max_tokens == 8192
        assert result.top_p == 0.9

    @pytest.mark.asyncio
    async def test_profile_after_deletion_raises_404(self, test_session, test_models):
        """Test raises 404 after profile is deleted."""
        # Create and delete a profile
        profile = ServerProfile(
            name="Deleted Profile",
            model_id=test_models["model1"].id,
        )
        test_session.add(profile)
        await test_session.commit()
        await test_session.refresh(profile)
        profile_id = profile.id

        # Delete the profile
        await test_session.delete(profile)
        await test_session.commit()

        # Try to get the deleted profile
        with pytest.raises(HTTPException) as exc_info:
            await get_profile_or_404(profile_id, test_session)

        assert exc_info.value.status_code == 404


class TestGetCurrentUser:
    """Tests for the get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self, test_session):
        """Test that invalid token raises 401."""
        from mlx_manager.dependencies import get_current_user

        with patch("mlx_manager.dependencies.decode_token", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user("invalid_token", test_session)
            assert exc_info.value.status_code == 401
            assert "Could not validate credentials" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_token_without_email_raises_401(self, test_session):
        """Test that token without email raises 401."""
        from mlx_manager.dependencies import get_current_user

        # Token payload without 'sub' field
        with patch("mlx_manager.dependencies.decode_token", return_value={"other": "data"}):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user("token_without_email", test_session)
            assert exc_info.value.status_code == 401
            assert "Could not validate credentials" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_user_not_found_raises_401(self, test_session):
        """Test that user not found raises 401."""
        from mlx_manager.dependencies import get_current_user

        # Valid token but user doesn't exist
        token_payload = {"sub": "nonexistent@example.com"}
        with patch("mlx_manager.dependencies.decode_token", return_value=token_payload):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user("valid_token", test_session)
            assert exc_info.value.status_code == 401
            assert "Could not validate credentials" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_pending_user_raises_401(self, test_session):
        """Test that pending user raises 401."""
        from mlx_manager.dependencies import get_current_user
        from mlx_manager.models import User, UserStatus
        from mlx_manager.services.auth_service import hash_password

        # Create pending user
        user = User(
            email="pending@example.com",
            hashed_password=hash_password("password"),
            is_admin=False,
            status=UserStatus.PENDING,
        )
        test_session.add(user)
        await test_session.commit()

        # Try to authenticate
        token_payload = {"sub": "pending@example.com"}
        with patch("mlx_manager.dependencies.decode_token", return_value=token_payload):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user("valid_token", test_session)
            assert exc_info.value.status_code == 401
            assert "Could not validate credentials" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_disabled_user_raises_401(self, test_session):
        """Test that disabled user raises 401."""
        from mlx_manager.dependencies import get_current_user
        from mlx_manager.models import User, UserStatus
        from mlx_manager.services.auth_service import hash_password

        # Create disabled user
        user = User(
            email="disabled@example.com",
            hashed_password=hash_password("password"),
            is_admin=False,
            status=UserStatus.DISABLED,
        )
        test_session.add(user)
        await test_session.commit()

        # Try to authenticate
        token_payload = {"sub": "disabled@example.com"}
        with patch("mlx_manager.dependencies.decode_token", return_value=token_payload):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user("valid_token", test_session)
            assert exc_info.value.status_code == 401
            assert "Could not validate credentials" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_approved_user_succeeds(self, test_session):
        """Test that approved user succeeds."""
        from mlx_manager.dependencies import get_current_user
        from mlx_manager.models import User, UserStatus
        from mlx_manager.services.auth_service import hash_password

        # Create approved user
        user = User(
            email="approved@example.com",
            hashed_password=hash_password("password"),
            is_admin=False,
            status=UserStatus.APPROVED,
        )
        test_session.add(user)
        await test_session.commit()

        # Authenticate
        token_payload = {"sub": "approved@example.com"}
        with patch("mlx_manager.dependencies.decode_token", return_value=token_payload):
            result = await get_current_user("valid_token", test_session)
            assert result.email == "approved@example.com"
            assert result.status == UserStatus.APPROVED
