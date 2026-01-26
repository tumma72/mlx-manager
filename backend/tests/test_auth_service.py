"""Tests for authentication service."""

import os
from datetime import timedelta

import jwt

# Ensure test environment is set before importing app modules
os.environ["MLX_MANAGER_DATABASE_PATH"] = ":memory:"
os.environ["MLX_MANAGER_DEBUG"] = "false"

from mlx_manager.config import settings
from mlx_manager.services.auth_service import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    """Tests for password hashing and verification."""

    def test_hash_password(self):
        """Test that password hashing produces a hash."""
        password = "testpassword123"
        hashed = hash_password(password)

        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$argon2")  # Argon2 hash format

    def test_verify_password_correct(self):
        """Test that correct password verification succeeds."""
        password = "testpassword123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test that incorrect password verification fails."""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        hashed = hash_password(password)

        assert verify_password(wrong_password, hashed) is False

    def test_same_password_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        password = "testpassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


class TestCreateAccessToken:
    """Tests for JWT token creation."""

    def test_create_token_with_default_expiry(self):
        """Test token creation with default expiry."""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)

        # Decode to verify
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])

        assert payload["sub"] == "test@example.com"
        assert "exp" in payload

    def test_create_token_with_custom_expiry(self):
        """Test token creation with custom expiry."""
        data = {"sub": "test@example.com"}
        custom_expiry = timedelta(hours=1)
        token = create_access_token(data, expires_delta=custom_expiry)

        # Decode to verify
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])

        assert payload["sub"] == "test@example.com"
        assert "exp" in payload

    def test_create_token_preserves_additional_data(self):
        """Test that additional data in payload is preserved."""
        data = {"sub": "test@example.com", "role": "admin", "custom": "value"}
        token = create_access_token(data)

        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])

        assert payload["sub"] == "test@example.com"
        assert payload["role"] == "admin"
        assert payload["custom"] == "value"


class TestDecodeToken:
    """Tests for JWT token decoding."""

    def test_decode_valid_token(self):
        """Test decoding a valid token."""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)

        payload = decode_token(token)

        assert payload is not None
        assert payload["sub"] == "test@example.com"

    def test_decode_invalid_token_returns_none(self):
        """Test that invalid token returns None."""
        invalid_token = "invalid.token.string"

        payload = decode_token(invalid_token)

        assert payload is None

    def test_decode_expired_token_returns_none(self):
        """Test that expired token returns None."""
        data = {"sub": "test@example.com"}
        # Create token that expires immediately
        token = create_access_token(data, expires_delta=timedelta(seconds=-1))

        payload = decode_token(token)

        assert payload is None

    def test_decode_token_wrong_secret_returns_none(self):
        """Test that token with wrong secret returns None."""
        data = {"sub": "test@example.com"}
        # Create token with different secret
        wrong_token = jwt.encode(
            {**data, "exp": 9999999999},
            "wrong_secret",
            algorithm=settings.jwt_algorithm,
        )

        payload = decode_token(wrong_token)

        assert payload is None

    def test_decode_token_wrong_algorithm_returns_none(self):
        """Test that token with wrong algorithm returns None."""
        data = {"sub": "test@example.com"}
        # Create token with different algorithm
        wrong_token = jwt.encode(
            {**data, "exp": 9999999999},
            settings.jwt_secret,
            algorithm="HS512",  # Different from HS256
        )

        payload = decode_token(wrong_token)

        assert payload is None

    def test_decode_token_handles_jwt_error(self):
        """Test that JWTError is caught and returns None."""
        # Malformed token that will raise JWTError
        malformed_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.malformed"

        payload = decode_token(malformed_token)

        assert payload is None

    def test_decode_token_handles_expired_signature_error(self):
        """Test that ExpiredSignatureError is caught and returns None."""
        data = {"sub": "test@example.com"}
        # Create an expired token
        expired_token = create_access_token(data, expires_delta=timedelta(seconds=-10))

        # Verify it's actually expired by trying to decode it
        payload = decode_token(expired_token)

        assert payload is None
