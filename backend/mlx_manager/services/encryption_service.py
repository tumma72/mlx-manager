"""API key encryption service using Fernet symmetric encryption."""

import base64
import os
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from mlx_manager.config import ensure_data_dir, settings


def _get_salt_path() -> Path:
    """Get the path to the encryption salt file."""
    return settings.database_path.parent / ".encryption_salt"


def _get_or_create_salt() -> bytes:
    """Get or create persistent salt for key derivation.

    The salt is stored alongside the database in ~/.mlx-manager/.encryption_salt.
    If the salt file doesn't exist, a new 16-byte random salt is generated and persisted.
    """
    ensure_data_dir()
    salt_path = _get_salt_path()

    if salt_path.exists():
        return salt_path.read_bytes()

    # Generate new salt
    salt = os.urandom(16)
    salt_path.write_bytes(salt)
    return salt


@lru_cache(maxsize=1)
def _get_fernet_cached(jwt_secret: str, salt_hex: str) -> Fernet:
    """Get cached Fernet instance with derived key.

    Args:
        jwt_secret: The JWT secret used as password for key derivation.
        salt_hex: Hex-encoded salt (for cache key stability).

    Returns:
        Fernet instance for encryption/decryption.
    """
    salt = bytes.fromhex(salt_hex)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=1_200_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(jwt_secret.encode()))
    return Fernet(key)


def _get_fernet() -> Fernet:
    """Get Fernet instance with derived key from jwt_secret + salt."""
    salt = _get_or_create_salt()
    return _get_fernet_cached(settings.jwt_secret, salt.hex())


def encrypt_api_key(plain_key: str) -> str:
    """Encrypt an API key for secure storage.

    Args:
        plain_key: The plaintext API key to encrypt.

    Returns:
        Base64-encoded encrypted string suitable for database storage.
    """
    f = _get_fernet()
    return f.encrypt(plain_key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key from storage.

    Args:
        encrypted_key: The encrypted API key from database.

    Returns:
        The original plaintext API key.

    Raises:
        InvalidToken: If the encrypted key is invalid or tampered with.
    """
    f = _get_fernet()
    return f.decrypt(encrypted_key.encode()).decode()


def clear_cache() -> None:
    """Clear the cached Fernet instance.

    Useful for testing when jwt_secret or salt changes.
    """
    _get_fernet_cached.cache_clear()


# Re-export InvalidToken for consumers
__all__ = ["encrypt_api_key", "decrypt_api_key", "clear_cache", "InvalidToken"]
