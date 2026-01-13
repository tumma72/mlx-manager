"""Tests for the security utilities."""

from unittest.mock import patch

from mlx_manager.utils.security import validate_model_path


class TestValidateModelPath:
    """Tests for the validate_model_path function."""

    def test_valid_huggingface_cache_path(self, tmp_path):
        """Test validates paths within HuggingFace cache directory."""
        cache_path = tmp_path / ".cache" / "huggingface" / "model"
        cache_path.mkdir(parents=True)

        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = [str(tmp_path / ".cache" / "huggingface")]

            result = validate_model_path(str(cache_path))

        assert result is True

    def test_valid_models_directory_path(self, tmp_path):
        """Test validates paths within models directory."""
        model_path = tmp_path / "models" / "my-model"
        model_path.mkdir(parents=True)

        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = [str(tmp_path / "models")]

            result = validate_model_path(str(model_path))

        assert result is True

    def test_valid_huggingface_model_id(self):
        """Test validates HuggingFace model IDs (org/model-name format)."""
        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = []

            # HuggingFace model IDs should be valid
            assert validate_model_path("mlx-community/test-model") is True
            assert validate_model_path("organization/model-name-4bit") is True
            assert validate_model_path("user/some-model") is True

    def test_invalid_absolute_path_outside_allowed_dirs(self, tmp_path):
        """Test rejects absolute paths outside allowed directories."""
        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = [str(tmp_path / "safe")]

            # Path outside allowed directories
            result = validate_model_path("/etc/passwd")

        assert result is False

    def test_invalid_path_traversal_attempt(self, tmp_path):
        """Test rejects path traversal attempts."""
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()

        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = [str(safe_dir)]

            # Path traversal attempt
            result = validate_model_path(str(safe_dir / ".." / "unsafe"))

        assert result is False

    def test_invalid_root_path(self):
        """Test rejects root path."""
        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = ["/home/user/models"]

            result = validate_model_path("/")

        assert result is False

    def test_invalid_empty_relative_path(self):
        """Test rejects empty path that doesn't look like HF model ID."""
        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = []

            # Empty path should resolve to cwd and likely fail
            result = validate_model_path("")

        # Result depends on whether cwd is in allowed dirs, but empty isn't an HF ID
        assert result is False  # Empty string doesn't contain "/" so not treated as HF ID

    def test_valid_nested_allowed_path(self, tmp_path):
        """Test validates deeply nested paths within allowed directories."""
        allowed_dir = tmp_path / "allowed"
        nested_path = allowed_dir / "level1" / "level2" / "level3" / "model"
        nested_path.mkdir(parents=True)

        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = [str(allowed_dir)]

            result = validate_model_path(str(nested_path))

        assert result is True

    def test_invalid_partial_match(self, tmp_path):
        """Test rejects paths that only partially match allowed directories."""
        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = [str(tmp_path / "allowed")]

            # Path that starts similarly but isn't under allowed dir
            result = validate_model_path(str(tmp_path / "allowed-other" / "model"))

        assert result is False

    def test_multiple_allowed_directories(self, tmp_path):
        """Test validates paths in any of multiple allowed directories."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir3 = tmp_path / "dir3"

        dir1.mkdir()
        dir2.mkdir()
        dir3.mkdir()

        model_in_dir2 = dir2 / "model"
        model_in_dir2.mkdir()

        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = [str(dir1), str(dir2), str(dir3)]

            result = validate_model_path(str(model_in_dir2))

        assert result is True

    def test_symlink_resolution(self, tmp_path):
        """Test properly resolves symlinks."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        model = real_dir / "model"
        model.mkdir()

        symlink = tmp_path / "symlink"
        symlink.symlink_to(real_dir)

        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = [str(real_dir)]

            # Path through symlink should resolve to real path
            result = validate_model_path(str(symlink / "model"))

        assert result is True

    def test_relative_path_starting_with_slash_is_not_hf_id(self):
        """Test that paths starting with / are not treated as HF model IDs."""
        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = []

            # Starts with / so not an HF model ID
            result = validate_model_path("/organization/model")

        # This should check the actual filesystem and likely fail
        assert result is False

    def test_hf_model_id_with_special_characters(self):
        """Test HuggingFace model IDs with valid special characters."""
        with patch("mlx_manager.utils.security.settings") as mock_settings:
            mock_settings.allowed_model_dirs = []

            # Valid HF model IDs with hyphens and numbers
            assert validate_model_path("mlx-community/Llama-3.2-1B-Instruct-4bit") is True
            assert validate_model_path("org-name/model_name_123") is True
