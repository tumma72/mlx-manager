"""Authentication service for password hashing and JWT token management.

Uses AuthLib jose for JWT encoding/decoding and pwdlib[argon2] for password hashing.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from authlib.jose import JoseError, jwt as authlib_jwt
from loguru import logger
from pwdlib import PasswordHash

from mlx_manager.config import settings

# Create module-level password hash instance (Argon2 by default)
password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a stored hash."""
    return password_hash.verify(plain_password, hashed_password)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token.

    Args:
        data: Payload data (typically {"sub": email})
        expires_delta: Optional custom expiry. Defaults to jwt_expire_days from settings.

    Returns:
        Encoded JWT token string.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(tz=UTC) + expires_delta
    else:
        expire = datetime.now(tz=UTC) + timedelta(days=settings.jwt_expire_days)
    to_encode["exp"] = expire
    header = {"alg": settings.jwt_algorithm}
    token: bytes = authlib_jwt.encode(header, to_encode, settings.jwt_secret)
    return token.decode("utf-8")


def decode_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT token.

    Args:
        token: JWT token string.

    Returns:
        Decoded payload dict, or None if invalid/expired.
    """
    try:
        payload = authlib_jwt.decode(token, settings.jwt_secret)
        payload.validate()
        return dict(payload)
    except JoseError as e:
        logger.debug(f"Invalid JWT token: {e}")
        return None
    except Exception as e:
        logger.debug(f"JWT decode error: {e}")
        return None
