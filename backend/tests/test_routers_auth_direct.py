"""Direct unit tests for auth router functions.

These tests call router functions directly with mock sessions to ensure
coverage is properly tracked (avoiding ASGI transport coverage issues).
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from mlx_manager.models import PasswordReset, User, UserCreate, UserStatus, UserUpdate
from mlx_manager.routers.auth import (
    delete_user,
    get_me,
    get_pending_count,
    list_users,
    login,
    register,
    reset_password,
    update_user,
)


# Helper to create a mock user
def create_mock_user(
    id: int = 1,
    email: str = "test@example.com",
    is_admin: bool = False,
    status: UserStatus = UserStatus.APPROVED,
) -> User:
    """Create a mock user for testing."""
    return User(
        id=id,
        email=email,
        hashed_password="hashed",
        is_admin=is_admin,
        status=status,
    )


class TestRegisterDirect:
    """Direct tests for register function."""

    @pytest.mark.asyncio
    async def test_register_first_user_becomes_admin(self):
        """Test that first user becomes admin and is auto-approved."""
        user_data = UserCreate(email="admin@example.com", password="password123")

        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        # First query: check email doesn't exist
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = None

        # Second query: count users (0 = first user)
        mock_result2 = MagicMock()
        mock_result2.scalar.return_value = 0

        mock_session.execute.side_effect = [mock_result1, mock_result2]

        with patch("mlx_manager.routers.auth.hash_password", return_value="hashed"):
            result = await register(user_data=user_data, session=mock_session)

        assert result.email == "admin@example.com"
        assert result.is_admin is True
        assert result.status == UserStatus.APPROVED
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_subsequent_user_is_pending(self):
        """Test that subsequent users are created as pending."""
        user_data = UserCreate(email="user@example.com", password="password123")

        mock_session = AsyncMock()
        mock_session.add = MagicMock()

        # First query: check email doesn't exist
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = None

        # Second query: count users (1 = not first user)
        mock_result2 = MagicMock()
        mock_result2.scalar.return_value = 1

        mock_session.execute.side_effect = [mock_result1, mock_result2]

        with patch("mlx_manager.routers.auth.hash_password", return_value="hashed"):
            result = await register(user_data=user_data, session=mock_session)

        assert result.email == "user@example.com"
        assert result.is_admin is False
        assert result.status == UserStatus.PENDING

    @pytest.mark.asyncio
    async def test_register_duplicate_email_fails(self):
        """Test that registering with existing email fails."""
        user_data = UserCreate(email="existing@example.com", password="password123")

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = create_mock_user()  # Email exists
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await register(user_data=user_data, session=mock_session)

        assert exc_info.value.status_code == 409
        assert "already registered" in exc_info.value.detail


class TestLoginDirect:
    """Direct tests for login function."""

    @pytest.mark.asyncio
    async def test_login_approved_user_success(self):
        """Test successful login for approved user."""
        mock_user = create_mock_user(status=UserStatus.APPROVED)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        mock_form = MagicMock()
        mock_form.username = "test@example.com"
        mock_form.password = "password123"

        with (
            patch("mlx_manager.routers.auth.verify_password", return_value=True),
            patch("mlx_manager.routers.auth.create_access_token", return_value="jwt-token"),
        ):
            result = await login(form_data=mock_form, session=mock_session)

        assert result.access_token == "jwt-token"
        assert result.token_type == "bearer"

    @pytest.mark.asyncio
    async def test_login_user_not_found(self):
        """Test login with non-existent email."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        mock_form = MagicMock()
        mock_form.username = "nonexistent@example.com"
        mock_form.password = "password123"

        with pytest.raises(HTTPException) as exc_info:
            await login(form_data=mock_form, session=mock_session)

        assert exc_info.value.status_code == 401
        assert "Incorrect" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_login_wrong_password(self):
        """Test login with incorrect password."""
        mock_user = create_mock_user(status=UserStatus.APPROVED)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        mock_form = MagicMock()
        mock_form.username = "test@example.com"
        mock_form.password = "wrongpassword"

        with patch("mlx_manager.routers.auth.verify_password", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await login(form_data=mock_form, session=mock_session)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_login_pending_user_fails(self):
        """Test that pending users cannot log in."""
        mock_user = create_mock_user(status=UserStatus.PENDING)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        mock_form = MagicMock()
        mock_form.username = "test@example.com"
        mock_form.password = "password123"

        with patch("mlx_manager.routers.auth.verify_password", return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                await login(form_data=mock_form, session=mock_session)

        assert exc_info.value.status_code == 403
        assert "pending" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_login_disabled_user_fails(self):
        """Test that disabled users cannot log in."""
        mock_user = create_mock_user(status=UserStatus.DISABLED)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        mock_form = MagicMock()
        mock_form.username = "test@example.com"
        mock_form.password = "password123"

        with patch("mlx_manager.routers.auth.verify_password", return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                await login(form_data=mock_form, session=mock_session)

        assert exc_info.value.status_code == 403
        assert "disabled" in exc_info.value.detail


class TestGetMeDirect:
    """Direct tests for get_me function."""

    @pytest.mark.asyncio
    async def test_get_me_returns_current_user(self):
        """Test that get_me returns the current user."""
        mock_user = create_mock_user(email="me@example.com", is_admin=True)

        result = await get_me(current_user=mock_user)

        assert result == mock_user
        assert result.email == "me@example.com"


class TestListUsersDirect:
    """Direct tests for list_users function."""

    @pytest.mark.asyncio
    async def test_list_users_returns_all_users(self):
        """Test that list_users returns all users."""
        mock_users = [
            create_mock_user(id=1, email="user1@example.com"),
            create_mock_user(id=2, email="user2@example.com"),
        ]

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_users
        mock_session.execute.return_value = mock_result

        mock_admin = create_mock_user(is_admin=True)

        result = await list_users(_admin=mock_admin, session=mock_session)

        assert len(result) == 2
        assert result[0].email == "user1@example.com"


class TestGetPendingCountDirect:
    """Direct tests for get_pending_count function."""

    @pytest.mark.asyncio
    async def test_get_pending_count_returns_count(self):
        """Test that pending count is returned."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result

        mock_admin = create_mock_user(is_admin=True)

        result = await get_pending_count(_admin=mock_admin, session=mock_session)

        assert result == {"count": 5}

    @pytest.mark.asyncio
    async def test_get_pending_count_returns_zero_when_none(self):
        """Test that zero is returned when no pending users."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result

        mock_admin = create_mock_user(is_admin=True)

        result = await get_pending_count(_admin=mock_admin, session=mock_session)

        assert result == {"count": 0}


class TestUpdateUserDirect:
    """Direct tests for update_user function."""

    @pytest.mark.asyncio
    async def test_update_user_status(self):
        """Test updating user status."""
        mock_user = create_mock_user(id=2, status=UserStatus.PENDING)
        mock_admin = create_mock_user(id=1, is_admin=True)

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        user_data = UserUpdate(status=UserStatus.APPROVED)

        result = await update_user(
            user_id=2, user_data=user_data, admin=mock_admin, session=mock_session
        )

        assert result.status == UserStatus.APPROVED
        assert result.approved_at is not None
        assert result.approved_by == 1
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_not_found(self):
        """Test updating non-existent user."""
        mock_admin = create_mock_user(id=1, is_admin=True)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        user_data = UserUpdate(status=UserStatus.APPROVED)

        with pytest.raises(HTTPException) as exc_info:
            await update_user(
                user_id=999, user_data=user_data, admin=mock_admin, session=mock_session
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cannot_demote_last_admin(self):
        """Test that last admin cannot be demoted."""
        mock_admin = create_mock_user(id=1, is_admin=True)

        mock_session = AsyncMock()
        # First query: get user to update
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = mock_admin
        # Second query: count admins (only 1)
        mock_result2 = MagicMock()
        mock_result2.scalar.return_value = 1
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        user_data = UserUpdate(is_admin=False)

        with pytest.raises(HTTPException) as exc_info:
            await update_user(
                user_id=1, user_data=user_data, admin=mock_admin, session=mock_session
            )

        assert exc_info.value.status_code == 400
        assert "last admin" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_cannot_disable_last_active_admin(self):
        """Test that last active admin cannot disable themselves."""
        mock_admin = create_mock_user(id=1, is_admin=True, status=UserStatus.APPROVED)

        mock_session = AsyncMock()
        # First query: get user to update
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = mock_admin
        # Second query: count active admins (only 1)
        mock_result2 = MagicMock()
        mock_result2.scalar.return_value = 1
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        user_data = UserUpdate(status=UserStatus.DISABLED)

        with pytest.raises(HTTPException) as exc_info:
            await update_user(
                user_id=1, user_data=user_data, admin=mock_admin, session=mock_session
            )

        assert exc_info.value.status_code == 400
        assert "last active admin" in exc_info.value.detail


class TestDeleteUserDirect:
    """Direct tests for delete_user function."""

    @pytest.mark.asyncio
    async def test_delete_user_success(self):
        """Test deleting a user."""
        mock_user = create_mock_user(id=2, is_admin=False)
        mock_admin = create_mock_user(id=1, is_admin=True)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        await delete_user(user_id=2, admin=mock_admin, session=mock_session)

        mock_session.delete.assert_called_once_with(mock_user)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self):
        """Test deleting non-existent user."""
        mock_admin = create_mock_user(id=1, is_admin=True)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await delete_user(user_id=999, admin=mock_admin, session=mock_session)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cannot_delete_last_admin(self):
        """Test that last admin cannot delete themselves."""
        mock_admin = create_mock_user(id=1, is_admin=True)

        mock_session = AsyncMock()
        # First query: get user to delete
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = mock_admin
        # Second query: count admins (only 1)
        mock_result2 = MagicMock()
        mock_result2.scalar.return_value = 1
        mock_session.execute.side_effect = [mock_result1, mock_result2]

        with pytest.raises(HTTPException) as exc_info:
            await delete_user(user_id=1, admin=mock_admin, session=mock_session)

        assert exc_info.value.status_code == 400
        assert "last admin" in exc_info.value.detail


class TestResetPasswordDirect:
    """Direct tests for reset_password function."""

    @pytest.mark.asyncio
    async def test_reset_password_success(self):
        """Test resetting a user's password."""
        mock_user = create_mock_user(id=2)
        mock_admin = create_mock_user(id=1, is_admin=True)

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        password_data = PasswordReset(password="newpassword123")

        with patch("mlx_manager.routers.auth.hash_password", return_value="newhashed"):
            result = await reset_password(
                user_id=2, password_data=password_data, _admin=mock_admin, session=mock_session
            )

        assert result == {"message": "Password reset successfully"}
        assert mock_user.hashed_password == "newhashed"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_password_user_not_found(self):
        """Test resetting password for non-existent user."""
        mock_admin = create_mock_user(id=1, is_admin=True)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        password_data = PasswordReset(password="newpassword123")

        with pytest.raises(HTTPException) as exc_info:
            await reset_password(
                user_id=999, password_data=password_data, _admin=mock_admin, session=mock_session
            )

        assert exc_info.value.status_code == 404
