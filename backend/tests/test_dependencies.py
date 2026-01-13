"""Tests for FastAPI dependencies."""

import os

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
    async def test_returns_profile_when_exists(self, test_session):
        """Test returns profile when it exists in database."""
        # Create a profile
        profile = ServerProfile(
            name="Test Profile",
            model_path="mlx-community/test-model",
            model_type="lm",
            port=10240,
            host="127.0.0.1",
            max_concurrency=1,
            queue_timeout=300,
            queue_size=100,
        )
        test_session.add(profile)
        await test_session.commit()
        await test_session.refresh(profile)

        # Get the profile
        result = await get_profile_or_404(profile.id, test_session)

        assert result is not None
        assert result.id == profile.id
        assert result.name == "Test Profile"
        assert result.model_path == "mlx-community/test-model"

    @pytest.mark.asyncio
    async def test_raises_404_when_not_exists(self, test_session):
        """Test raises HTTPException 404 when profile doesn't exist."""
        with pytest.raises(HTTPException) as exc_info:
            await get_profile_or_404(999, test_session)

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Profile not found"

    @pytest.mark.asyncio
    async def test_returns_correct_profile_among_many(self, test_session):
        """Test returns the correct profile when multiple exist."""
        # Create multiple profiles
        profiles = []
        for i in range(5):
            profile = ServerProfile(
                name=f"Profile {i}",
                model_path=f"mlx-community/model-{i}",
                model_type="lm",
                port=10240 + i,
                host="127.0.0.1",
                max_concurrency=1,
                queue_timeout=300,
                queue_size=100,
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
        assert result.port == 10242

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
    async def test_returns_all_profile_fields(self, test_session):
        """Test returns profile with all fields populated."""
        # Create a profile with all fields
        profile = ServerProfile(
            name="Complete Profile",
            description="A complete test profile",
            model_path="mlx-community/complete-model",
            model_type="multimodal",
            port=11000,
            host="0.0.0.0",
            context_length=4096,
            max_concurrency=4,
            queue_timeout=600,
            queue_size=200,
            tool_call_parser="default",
            reasoning_parser="cot",
            enable_auto_tool_choice=True,
            trust_remote_code=False,
            chat_template_file="/path/to/template",
            log_level="DEBUG",
            log_file="/path/to/log",
            no_log_file=False,
            auto_start=True,
        )
        test_session.add(profile)
        await test_session.commit()
        await test_session.refresh(profile)

        # Get the profile
        result = await get_profile_or_404(profile.id, test_session)

        # Verify all fields
        assert result.name == "Complete Profile"
        assert result.description == "A complete test profile"
        assert result.model_path == "mlx-community/complete-model"
        assert result.model_type == "multimodal"
        assert result.port == 11000
        assert result.host == "0.0.0.0"
        assert result.context_length == 4096
        assert result.max_concurrency == 4
        assert result.queue_timeout == 600
        assert result.queue_size == 200
        assert result.tool_call_parser == "default"
        assert result.reasoning_parser == "cot"
        assert result.enable_auto_tool_choice is True
        assert result.trust_remote_code is False
        assert result.chat_template_file == "/path/to/template"
        assert result.log_level == "DEBUG"
        assert result.log_file == "/path/to/log"
        assert result.no_log_file is False
        assert result.auto_start is True

    @pytest.mark.asyncio
    async def test_profile_after_deletion_raises_404(self, test_session):
        """Test raises 404 after profile is deleted."""
        # Create and delete a profile
        profile = ServerProfile(
            name="Deleted Profile",
            model_path="mlx-community/deleted-model",
            model_type="lm",
            port=10240,
            host="127.0.0.1",
            max_concurrency=1,
            queue_timeout=300,
            queue_size=100,
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
