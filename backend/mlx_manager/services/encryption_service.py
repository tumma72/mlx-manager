"""API key encryption service using AuthLib JWE (A256KW + A256GCM).

Replaces the previous Fernet-based implementation with AuthLib's JWE for
symmetric encryption. Uses the jwt_secret (SHA-256 hashed to 256 bits) as
the encryption key. No salt file needed — the key is deterministic from
the jwt_secret alone.
"""

import hashlib
from functools import lru_cache

from authlib.jose import JsonWebEncryption, OctKey
from loguru import logger

from mlx_manager.config import settings


class DecryptionError(Exception):
    """Raised when decryption fails (wrong key, tampered data, or invalid format)."""


@lru_cache(maxsize=1)
def _get_jwe_key_cached(jwt_secret: str) -> OctKey:
    """Get cached OctKey derived from jwt_secret via SHA-256.

    Args:
        jwt_secret: The JWT secret used to derive the encryption key.

    Returns:
        OctKey instance for JWE encryption/decryption.
    """
    key_bytes = hashlib.sha256(jwt_secret.encode()).digest()
    return OctKey.import_key(key_bytes)


def _get_jwe_key() -> OctKey:
    """Get OctKey instance derived from jwt_secret."""
    return _get_jwe_key_cached(settings.jwt_secret)


def encrypt_api_key(plain_key: str) -> str:
    """Encrypt an API key for secure storage using JWE (A256KW + A256GCM).

    Args:
        plain_key: The plaintext API key to encrypt.

    Returns:
        JWE compact serialization string suitable for database storage.
    """
    jwe = JsonWebEncryption()
    header = {"alg": "A256KW", "enc": "A256GCM"}
    key = _get_jwe_key()
    token: bytes = jwe.serialize_compact(header, plain_key.encode(), key)
    return token.decode("ascii")


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key from storage.

    Args:
        encrypted_key: The JWE compact serialization string from database.

    Returns:
        The original plaintext API key.

    Raises:
        DecryptionError: If the encrypted key is invalid, tampered with,
            or was encrypted with a different key.
    """
    try:
        jwe = JsonWebEncryption()
        key = _get_jwe_key()
        data = jwe.deserialize_compact(encrypted_key.encode("ascii"), key)
        payload: bytes = data["payload"]
        return payload.decode("utf-8")
    except Exception as e:
        logger.error(
            f"Failed to decrypt API key. If this was encrypted with a previous version, "
            f"please re-enter the API key in Settings. Error: {e}"
        )
        raise DecryptionError(str(e)) from e


def clear_cache() -> None:
    """Clear the cached JWE key instance.

    Useful for testing when jwt_secret changes.
    """
    _get_jwe_key_cached.cache_clear()


# Re-export DecryptionError as InvalidToken for backward compatibility with consumers
InvalidToken = DecryptionError

__all__ = ["encrypt_api_key", "decrypt_api_key", "clear_cache", "InvalidToken", "DecryptionError"]
