"""Tests for authentication API endpoints."""

import pytest

from mlx_manager.models import User, UserStatus
from mlx_manager.services.auth_service import hash_password


class TestAuthRegister:
    """Tests for POST /api/auth/register endpoint."""

    @pytest.mark.asyncio
    async def test_register_first_user_becomes_admin(self, client):
        """Test that the first registered user becomes admin and is auto-approved."""
        response = await client.post(
            "/api/auth/register",
            json={"email": "admin@example.com", "password": "password123"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "admin@example.com"
        assert data["is_admin"] is True
        assert data["status"] == "approved"

    @pytest.mark.asyncio
    async def test_register_second_user_is_pending(self, client, test_session):
        """Test that subsequent users are pending approval."""
        # Create first user (admin)
        first_user = User(
            email="admin@example.com",
            hashed_password=hash_password("password123"),
            is_admin=True,
            status=UserStatus.APPROVED,
        )
        test_session.add(first_user)
        await test_session.commit()

        # Register second user
        response = await client.post(
            "/api/auth/register",
            json={"email": "user@example.com", "password": "password123"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "user@example.com"
        assert data["is_admin"] is False
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_register_duplicate_email_fails(self, client, test_session):
        """Test that registering with existing email fails."""
        # Create existing user
        user = User(
            email="existing@example.com",
            hashed_password=hash_password("password123"),
            is_admin=True,
            status=UserStatus.APPROVED,
        )
        test_session.add(user)
        await test_session.commit()

        # Try to register with same email
        response = await client.post(
            "/api/auth/register",
            json={"email": "existing@example.com", "password": "newpassword"},
        )

        assert response.status_code == 409
        assert "already registered" in response.json()["detail"]


class TestAuthLogin:
    """Tests for POST /api/auth/login endpoint."""

    @pytest.mark.asyncio
    async def test_login_approved_user(self, client, test_session):
        """Test that approved users can log in."""
        # Create approved user
        user = User(
            email="user@example.com",
            hashed_password=hash_password("password123"),
            is_admin=False,
            status=UserStatus.APPROVED,
        )
        test_session.add(user)
        await test_session.commit()

        # Login
        response = await client.post(
            "/api/auth/login",
            data={"username": "user@example.com", "password": "password123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    @pytest.mark.asyncio
    async def test_login_pending_user_fails(self, client, test_session):
        """Test that pending users cannot log in."""
        # Create pending user
        user = User(
            email="pending@example.com",
            hashed_password=hash_password("password123"),
            is_admin=False,
            status=UserStatus.PENDING,
        )
        test_session.add(user)
        await test_session.commit()

        # Try to login
        response = await client.post(
            "/api/auth/login",
            data={"username": "pending@example.com", "password": "password123"},
        )

        assert response.status_code == 403
        assert "pending" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_disabled_user_fails(self, client, test_session):
        """Test that disabled users cannot log in."""
        # Create disabled user
        user = User(
            email="disabled@example.com",
            hashed_password=hash_password("password123"),
            is_admin=False,
            status=UserStatus.DISABLED,
        )
        test_session.add(user)
        await test_session.commit()

        # Try to login
        response = await client.post(
            "/api/auth/login",
            data={"username": "disabled@example.com", "password": "password123"},
        )

        assert response.status_code == 403
        assert "disabled" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_wrong_password_fails(self, client, test_session):
        """Test that wrong password fails."""
        # Create user
        user = User(
            email="user@example.com",
            hashed_password=hash_password("password123"),
            is_admin=False,
            status=UserStatus.APPROVED,
        )
        test_session.add(user)
        await test_session.commit()

        # Try to login with wrong password
        response = await client.post(
            "/api/auth/login",
            data={"username": "user@example.com", "password": "wrongpassword"},
        )

        assert response.status_code == 401
        assert "Incorrect" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_nonexistent_user_fails(self, client):
        """Test that nonexistent user cannot log in."""
        response = await client.post(
            "/api/auth/login",
            data={"username": "noone@example.com", "password": "password123"},
        )

        assert response.status_code == 401


class TestAuthMe:
    """Tests for GET /api/auth/me endpoint."""

    @pytest.mark.asyncio
    async def test_get_me_authenticated(self, auth_client):
        """Test getting current user info when authenticated."""
        response = await auth_client.get("/api/auth/me")

        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert "is_admin" in data
        assert "status" in data

    @pytest.mark.asyncio
    async def test_get_me_unauthenticated(self, client):
        """Test that unauthenticated request fails."""
        response = await client.get("/api/auth/me")

        assert response.status_code == 401


class TestAdminListUsers:
    """Tests for GET /api/auth/users endpoint."""

    @pytest.mark.asyncio
    async def test_list_users_as_admin(self, admin_client):
        """Test listing users as admin."""
        response = await admin_client.get("/api/auth/users")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # At least the admin user

    @pytest.mark.asyncio
    async def test_list_users_as_non_admin_fails(self, auth_client):
        """Test that non-admin cannot list users."""
        response = await auth_client.get("/api/auth/users")

        assert response.status_code == 403


class TestAdminPendingCount:
    """Tests for GET /api/auth/users/pending/count endpoint."""

    @pytest.mark.asyncio
    async def test_get_pending_count_as_admin(self, admin_client):
        """Test getting pending count as admin."""
        response = await admin_client.get("/api/auth/users/pending/count")

        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert isinstance(data["count"], int)

    @pytest.mark.asyncio
    async def test_get_pending_count_as_non_admin_fails(self, auth_client):
        """Test that non-admin cannot get pending count."""
        response = await auth_client.get("/api/auth/users/pending/count")

        assert response.status_code == 403


class TestAdminUpdateUser:
    """Tests for PUT /api/auth/users/{user_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_user_status(self, admin_client, test_session):
        """Test updating user status as admin."""
        # Create pending user
        user = User(
            email="pending@example.com",
            hashed_password=hash_password("password123"),
            is_admin=False,
            status=UserStatus.PENDING,
        )
        test_session.add(user)
        await test_session.commit()
        await test_session.refresh(user)

        # Update to approved
        response = await admin_client.put(
            f"/api/auth/users/{user.id}",
            json={"status": "approved"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"

    @pytest.mark.asyncio
    async def test_update_user_not_found(self, admin_client):
        """Test updating nonexistent user fails."""
        response = await admin_client.put(
            "/api/auth/users/99999",
            json={"status": "approved"},
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cannot_demote_last_admin(self, admin_client):
        """Test that the last admin cannot be demoted."""
        # Get current user (the admin) via /me
        me_response = await admin_client.get("/api/auth/me")
        admin_id = me_response.json()["id"]

        # Try to demote
        response = await admin_client.put(
            f"/api/auth/users/{admin_id}",
            json={"is_admin": False},
        )

        assert response.status_code == 400
        assert "last admin" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_cannot_disable_last_active_admin(self, admin_client):
        """Test that the last active admin cannot disable themselves."""
        # Get current user (the admin) via /me
        me_response = await admin_client.get("/api/auth/me")
        admin_id = me_response.json()["id"]

        # Try to disable self
        response = await admin_client.put(
            f"/api/auth/users/{admin_id}",
            json={"status": "disabled"},
        )

        assert response.status_code == 400
        assert "last active admin" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_update_user_as_non_admin_fails(self, auth_client):
        """Test that non-admin cannot update users."""
        response = await auth_client.put(
            "/api/auth/users/1",
            json={"status": "disabled"},
        )

        assert response.status_code == 403


class TestAdminDeleteUser:
    """Tests for DELETE /api/auth/users/{user_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_user(self, admin_client, test_session):
        """Test deleting a user as admin."""
        # Create user to delete
        user = User(
            email="todelete@example.com",
            hashed_password=hash_password("password123"),
            is_admin=False,
            status=UserStatus.APPROVED,
        )
        test_session.add(user)
        await test_session.commit()
        await test_session.refresh(user)

        # Delete
        response = await admin_client.delete(f"/api/auth/users/{user.id}")

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self, admin_client):
        """Test deleting nonexistent user fails."""
        response = await admin_client.delete("/api/auth/users/99999")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cannot_delete_last_admin(self, admin_client):
        """Test that the last admin cannot delete themselves."""
        # Get current user (the admin) via /me
        me_response = await admin_client.get("/api/auth/me")
        admin_id = me_response.json()["id"]

        # Try to delete self
        response = await admin_client.delete(f"/api/auth/users/{admin_id}")

        assert response.status_code == 400
        assert "last admin" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_user_as_non_admin_fails(self, auth_client):
        """Test that non-admin cannot delete users."""
        response = await auth_client.delete("/api/auth/users/1")

        assert response.status_code == 403


class TestAdminResetPassword:
    """Tests for POST /api/auth/users/{user_id}/reset-password endpoint."""

    @pytest.mark.asyncio
    async def test_reset_password(self, admin_client, test_session):
        """Test resetting a user's password as admin."""
        # Create user
        user = User(
            email="resetme@example.com",
            hashed_password=hash_password("oldpassword"),
            is_admin=False,
            status=UserStatus.APPROVED,
        )
        test_session.add(user)
        await test_session.commit()
        await test_session.refresh(user)

        # Reset password
        response = await admin_client.post(
            f"/api/auth/users/{user.id}/reset-password",
            json={"password": "newpassword123"},
        )

        assert response.status_code == 200
        assert "successfully" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_reset_password_user_not_found(self, admin_client):
        """Test resetting password for nonexistent user fails."""
        response = await admin_client.post(
            "/api/auth/users/99999/reset-password",
            json={"password": "newpassword123"},
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_reset_password_as_non_admin_fails(self, auth_client):
        """Test that non-admin cannot reset passwords."""
        response = await auth_client.post(
            "/api/auth/users/1/reset-password",
            json={"password": "newpassword123"},
        )

        assert response.status_code == 403
