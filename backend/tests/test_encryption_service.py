"""Tests for the encryption service (AuthLib JWE)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_manager.services import encryption_service
from mlx_manager.services.encryption_service import (
    DecryptionError,
    clear_cache,
    decrypt_api_key,
    encrypt_api_key,
)


@pytest.fixture
def temp_data_dir(tmp_path: Path):
    """Create a temporary data directory and configure settings to use it."""
    data_dir = tmp_path / ".mlx-manager"
    data_dir.mkdir()
    db_path = data_dir / "test.db"

    # Clear any cached JWE key instance
    clear_cache()

    # Mock settings to use the temp directory
    with patch.object(encryption_service.settings, "database_path", db_path):
        with patch.object(encryption_service.settings, "jwt_secret", "test-secret-key-12345"):
            yield data_dir

    # Cleanup cache after test
    clear_cache()


class TestEncryptDecryptRoundtrip:
    """Test encrypt/decrypt roundtrip functionality."""

    def test_basic_roundtrip(self, temp_data_dir: Path):
        """Encrypting and decrypting returns the original value."""
        original = "sk-1234567890abcdef"
        encrypted = encrypt_api_key(original)
        decrypted = decrypt_api_key(encrypted)

        assert decrypted == original

    def test_roundtrip_with_special_characters(self, temp_data_dir: Path):
        """Handles API keys with special characters."""
        original = "sk-test!@#$%^&*()_+-=[]{}|;':\",./<>?"
        encrypted = encrypt_api_key(original)
        decrypted = decrypt_api_key(encrypted)

        assert decrypted == original

    def test_roundtrip_with_long_key(self, temp_data_dir: Path):
        """Handles long API keys."""
        original = "sk-" + "a" * 500
        encrypted = encrypt_api_key(original)
        decrypted = decrypt_api_key(encrypted)

        assert decrypted == original

    def test_roundtrip_with_unicode(self, temp_data_dir: Path):
        """Handles API keys with unicode characters."""
        original = "sk-test-unicode-\u00e9\u00e8\u00ea"
        encrypted = encrypt_api_key(original)
        decrypted = decrypt_api_key(encrypted)

        assert decrypted == original

    def test_empty_key_roundtrip(self, temp_data_dir: Path):
        """Handles empty API keys (edge case)."""
        original = ""
        encrypted = encrypt_api_key(original)
        decrypted = decrypt_api_key(encrypted)

        assert decrypted == original


class TestEncryptionUniqueness:
    """Test that encryption produces unique ciphertext."""

    def test_different_keys_produce_different_ciphertext(self, temp_data_dir: Path):
        """Different plaintext produces different ciphertext."""
        key1 = "sk-key-one-12345"
        key2 = "sk-key-two-67890"

        encrypted1 = encrypt_api_key(key1)
        encrypted2 = encrypt_api_key(key2)

        assert encrypted1 != encrypted2

    def test_same_key_produces_different_ciphertext_each_time(self, temp_data_dir: Path):
        """Same plaintext produces different ciphertext due to random IV in A256GCM."""
        original = "sk-same-key-12345"

        encrypted1 = encrypt_api_key(original)
        encrypted2 = encrypt_api_key(original)

        # JWE A256GCM uses random IV, so ciphertext differs
        assert encrypted1 != encrypted2

        # But both decrypt to the same value
        assert decrypt_api_key(encrypted1) == original
        assert decrypt_api_key(encrypted2) == original


class TestDecryptionFailures:
    """Test decryption failure cases."""

    def test_decrypt_with_wrong_secret_fails(self, temp_data_dir: Path):
        """Decryption fails when jwt_secret changes."""
        original = "sk-test-key-12345"
        encrypted = encrypt_api_key(original)

        # Clear cache and change secret
        clear_cache()

        with patch.object(encryption_service.settings, "jwt_secret", "different-secret"):
            with pytest.raises(DecryptionError):
                decrypt_api_key(encrypted)

    def test_decrypt_invalid_token_fails(self, temp_data_dir: Path):
        """Decryption fails for invalid ciphertext."""
        with pytest.raises(DecryptionError):
            decrypt_api_key("not-a-valid-encrypted-token")

    def test_decrypt_tampered_token_fails(self, temp_data_dir: Path):
        """Decryption fails for tampered ciphertext."""
        original = "sk-test-key-12345"
        encrypted = encrypt_api_key(original)

        # Tamper with the ciphertext
        tampered = encrypted[:-5] + "XXXXX"

        with pytest.raises(DecryptionError):
            decrypt_api_key(tampered)


class TestJweFormat:
    """Test JWE token format."""

    def test_encrypted_key_is_jwe_compact_serialization(self, temp_data_dir: Path):
        """Encrypted output is JWE compact serialization (5 dot-separated parts)."""
        original = "sk-test-key-12345"
        encrypted = encrypt_api_key(original)

        # JWE compact serialization has 5 parts: header.key.iv.ciphertext.tag
        parts = encrypted.split(".")
        assert len(parts) == 5

    def test_encrypted_key_is_ascii_safe(self, temp_data_dir: Path):
        """Encrypted output is ASCII-safe for database storage."""
        original = "sk-test-key-12345"
        encrypted = encrypt_api_key(original)

        # Should be valid ASCII (base64url encoding)
        encrypted.encode("ascii")  # Should not raise


class TestCacheManagement:
    """Test JWE key caching."""

    def test_cache_is_used_for_repeated_calls(self, temp_data_dir: Path):
        """JWE key instance is cached and reused."""
        # Multiple calls should use cached instance
        for _ in range(10):
            encrypt_api_key("test-key")

        # Cache should have exactly 1 entry
        cache_info = encryption_service._get_jwe_key_cached.cache_info()
        assert cache_info.hits >= 9  # At least 9 cache hits

    def test_clear_cache_resets_key(self, temp_data_dir: Path):
        """Clearing cache allows new secret to take effect."""
        original = "sk-test-key-12345"

        # Encrypt with first secret
        encrypted1 = encrypt_api_key(original)

        # Change secret and clear cache
        clear_cache()
        with patch.object(encryption_service.settings, "jwt_secret", "new-secret"):
            # This should use new secret
            encrypted2 = encrypt_api_key(original)

            # New encryption should differ (different key derivation)
            assert encrypted1 != encrypted2

            # Can decrypt with current (new) secret
            assert decrypt_api_key(encrypted2) == original


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_multiple_api_keys(self, temp_data_dir: Path):
        """Can encrypt and decrypt multiple different API keys."""
        keys = {
            "openai": "sk-proj-abcdef123456",
            "anthropic": "sk-ant-xyz789012345",
            "azure": "azure-key-with-hyphens-00000",
        }

        encrypted = {name: encrypt_api_key(key) for name, key in keys.items()}
        decrypted = {name: decrypt_api_key(enc) for name, enc in encrypted.items()}

        assert decrypted == keys

    def test_backward_compat_invalid_token_alias(self, temp_data_dir: Path):
        """InvalidToken alias works for backward compatibility."""
        from mlx_manager.services.encryption_service import InvalidToken

        assert InvalidToken is DecryptionError

        with pytest.raises(InvalidToken):
            decrypt_api_key("not-a-valid-encrypted-token")
