"""Tests for the encryption service."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.fernet import InvalidToken

from mlx_manager.services import encryption_service
from mlx_manager.services.encryption_service import (
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

    # Clear any cached Fernet instance
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
        """Same plaintext produces different ciphertext due to random IV."""
        original = "sk-same-key-12345"

        encrypted1 = encrypt_api_key(original)
        encrypted2 = encrypt_api_key(original)

        # Fernet uses random IV, so ciphertext differs
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
            with pytest.raises(InvalidToken):
                decrypt_api_key(encrypted)

    def test_decrypt_invalid_token_fails(self, temp_data_dir: Path):
        """Decryption fails for invalid ciphertext."""
        with pytest.raises(InvalidToken):
            decrypt_api_key("not-a-valid-encrypted-token")

    def test_decrypt_tampered_token_fails(self, temp_data_dir: Path):
        """Decryption fails for tampered ciphertext."""
        original = "sk-test-key-12345"
        encrypted = encrypt_api_key(original)

        # Tamper with the ciphertext
        tampered = encrypted[:-5] + "XXXXX"

        with pytest.raises(InvalidToken):
            decrypt_api_key(tampered)


class TestSaltPersistence:
    """Test salt file persistence."""

    def test_salt_file_created_on_first_use(self, temp_data_dir: Path):
        """Salt file is created when encryption is first used."""
        salt_path = temp_data_dir / ".encryption_salt"
        assert not salt_path.exists()

        # Trigger salt creation
        encrypt_api_key("test-key")

        assert salt_path.exists()
        assert len(salt_path.read_bytes()) == 16  # 16-byte salt

    def test_salt_file_persists_across_calls(self, temp_data_dir: Path):
        """Same salt is used across multiple calls."""
        salt_path = temp_data_dir / ".encryption_salt"

        # First encryption creates salt
        encrypt_api_key("test-key")
        salt1 = salt_path.read_bytes()

        # Clear cache to force re-read
        clear_cache()

        # Second encryption uses same salt
        encrypt_api_key("test-key")
        salt2 = salt_path.read_bytes()

        assert salt1 == salt2

    def test_existing_salt_file_is_used(self, temp_data_dir: Path):
        """Pre-existing salt file is used instead of generating new one."""
        salt_path = temp_data_dir / ".encryption_salt"

        # Create salt file manually
        known_salt = os.urandom(16)
        salt_path.write_bytes(known_salt)

        # Encrypt should use existing salt
        encrypt_api_key("test-key")

        # Verify salt wasn't overwritten
        assert salt_path.read_bytes() == known_salt


class TestCacheManagement:
    """Test Fernet instance caching."""

    def test_cache_is_used_for_repeated_calls(self, temp_data_dir: Path):
        """Fernet instance is cached and reused."""
        # Multiple calls should use cached instance
        for _ in range(10):
            encrypt_api_key("test-key")

        # Cache should have exactly 1 entry
        cache_info = encryption_service._get_fernet_cached.cache_info()
        assert cache_info.hits >= 9  # At least 9 cache hits

    def test_clear_cache_resets_fernet(self, temp_data_dir: Path):
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

    def test_encrypted_key_is_base64_safe(self, temp_data_dir: Path):
        """Encrypted output is safe for database storage (base64)."""
        original = "sk-test-key-12345"
        encrypted = encrypt_api_key(original)

        # Should be valid base64
        import base64

        # Fernet tokens are URL-safe base64
        decoded = base64.urlsafe_b64decode(encrypted)
        assert len(decoded) > len(original)  # Encryption adds overhead
