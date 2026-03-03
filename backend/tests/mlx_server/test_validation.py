"""Tests for input validation utilities.

Covers:
- validate_model_available: model resolution and availability checks
- validate_base64_image: MIME type and size validation for data URLs
- validate_image_url: routing between regular URLs and data URL validation
"""

import base64
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from mlx_manager.mlx_server.utils.request_helpers import validate_model_available
from mlx_manager.mlx_server.utils.validation import (
    VALID_IMAGE_TYPES,
    validate_base64_image,
    validate_image_url,
)

# The functions under test use lazy imports of get_settings from
# mlx_manager.mlx_server.config.  We patch at the config module level.
_CONFIG_SETTINGS = "mlx_manager.mlx_server.config.get_settings"


def _mock_settings(
    available_models: list[str] | None = None,
    default_model: str | None = None,
    max_image_size_mb: int = 20,
) -> Mock:
    """Create a mock MLXServerSettings object."""
    settings = Mock()
    settings.available_models = available_models or [
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Qwen3-0.6B-4bit-DWQ",
    ]
    settings.default_model = default_model
    settings.max_image_size_mb = max_image_size_mb
    return settings


# ---------------------------------------------------------------------------
# validate_model_available
# ---------------------------------------------------------------------------


class TestValidateModelAvailable:
    """Tests for model availability validation."""

    def test_returns_model_when_available(self) -> None:
        """Model in available_models list is returned unchanged."""
        with patch(_CONFIG_SETTINGS, return_value=_mock_settings()):
            result = validate_model_available("mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert result == "mlx-community/Llama-3.2-3B-Instruct-4bit"

    def test_raises_404_when_not_available(self) -> None:
        """Model not in available_models raises 404."""
        with patch(_CONFIG_SETTINGS, return_value=_mock_settings()):
            with pytest.raises(HTTPException) as exc_info:
                validate_model_available("mlx-community/nonexistent-model")
        assert exc_info.value.status_code == 404
        assert "not available" in exc_info.value.detail
        assert "nonexistent-model" in exc_info.value.detail

    def test_resolves_default_model(self) -> None:
        """When model is None and default_model is configured, returns default."""
        settings = _mock_settings(
            default_model="mlx-community/Qwen3-0.6B-4bit-DWQ",
        )
        with patch(_CONFIG_SETTINGS, return_value=settings):
            result = validate_model_available(None)
        assert result == "mlx-community/Qwen3-0.6B-4bit-DWQ"

    def test_resolves_empty_string_to_default(self) -> None:
        """Empty string model falls back to default_model."""
        settings = _mock_settings(
            default_model="mlx-community/Qwen3-0.6B-4bit-DWQ",
        )
        with patch(_CONFIG_SETTINGS, return_value=settings):
            result = validate_model_available("")
        assert result == "mlx-community/Qwen3-0.6B-4bit-DWQ"

    def test_raises_400_when_no_model_and_no_default(self) -> None:
        """When model is None and no default_model, raises 400."""
        settings = _mock_settings(default_model=None)
        with patch(_CONFIG_SETTINGS, return_value=settings):
            with pytest.raises(HTTPException) as exc_info:
                validate_model_available(None)
        assert exc_info.value.status_code == 400
        assert "No model specified" in exc_info.value.detail

    def test_raises_404_when_default_not_in_available(self) -> None:
        """Default model that is not in available_models raises 404."""
        settings = _mock_settings(
            available_models=["mlx-community/Llama-3.2-3B-Instruct-4bit"],
            default_model="mlx-community/missing-default",
        )
        with patch(_CONFIG_SETTINGS, return_value=settings):
            with pytest.raises(HTTPException) as exc_info:
                validate_model_available(None)
        assert exc_info.value.status_code == 404

    def test_error_lists_available_models(self) -> None:
        """404 error detail includes the list of available models."""
        settings = _mock_settings(available_models=["model-a", "model-b"])
        with patch(_CONFIG_SETTINGS, return_value=settings):
            with pytest.raises(HTTPException) as exc_info:
                validate_model_available("model-x")
        assert "model-a" in exc_info.value.detail
        assert "model-b" in exc_info.value.detail


# ---------------------------------------------------------------------------
# validate_base64_image
# ---------------------------------------------------------------------------


def _make_data_url(mime: str = "image/png", size_bytes: int = 100) -> str:
    """Build a data URL with the given MIME type and approximate decoded size."""
    raw = b"\x00" * size_bytes
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


class TestValidateBase64Image:
    """Tests for base64 image validation."""

    def test_passes_valid_png(self) -> None:
        """Valid PNG data URL passes without error."""
        url = _make_data_url("image/png", 1024)
        validate_base64_image(url)  # should not raise

    def test_passes_valid_jpeg(self) -> None:
        """Valid JPEG data URL passes without error."""
        url = _make_data_url("image/jpeg", 512)
        validate_base64_image(url)  # should not raise

    def test_passes_all_valid_mime_types(self) -> None:
        """All VALID_IMAGE_TYPES are accepted."""
        for mime in VALID_IMAGE_TYPES:
            url = _make_data_url(mime, 64)
            validate_base64_image(url)  # should not raise

    def test_rejects_unsupported_mime_type(self) -> None:
        """Unsupported MIME type raises 422."""
        url = _make_data_url("image/svg+xml", 100)
        with pytest.raises(HTTPException) as exc_info:
            validate_base64_image(url)
        assert exc_info.value.status_code == 422
        assert "Unsupported image type" in exc_info.value.detail
        assert "image/svg+xml" in exc_info.value.detail

    def test_rejects_oversized_image(self) -> None:
        """Image exceeding max_size_mb raises 422."""
        # Create data URL with ~2 MB of decoded data, limit to 1 MB
        url = _make_data_url("image/png", 2 * 1024 * 1024)
        with pytest.raises(HTTPException) as exc_info:
            validate_base64_image(url, max_size_mb=1)
        assert exc_info.value.status_code == 422
        assert "Image too large" in exc_info.value.detail
        assert "1MB" in exc_info.value.detail

    def test_passes_image_under_limit(self) -> None:
        """Image well under the limit passes."""
        # 500 KB of data, limit is 1 MB
        url = _make_data_url("image/png", 500 * 1024)
        validate_base64_image(url, max_size_mb=1)  # should not raise

    def test_skips_non_data_url(self) -> None:
        """Non-data URLs are skipped (no validation)."""
        validate_base64_image("https://example.com/image.png")  # should not raise

    def test_skips_non_image_data_url(self) -> None:
        """data URLs that are not image/* are skipped (no regex match)."""
        validate_base64_image("data:text/plain;base64,SGVsbG8=")  # should not raise

    def test_custom_max_size(self) -> None:
        """Custom max_size_mb is respected."""
        # ~5 MB data, max 50 MB -- should pass
        url = _make_data_url("image/jpeg", 5 * 1024 * 1024)
        validate_base64_image(url, max_size_mb=50)  # should not raise

        # Same data but with 1 MB limit -- should fail
        with pytest.raises(HTTPException) as exc_info:
            validate_base64_image(url, max_size_mb=1)
        assert exc_info.value.status_code == 422

    def test_size_estimation_accuracy(self) -> None:
        """Base64 size estimation is reasonably accurate."""
        # 1 MB of random-ish data
        raw = b"\x42" * (1024 * 1024)
        b64 = base64.b64encode(raw).decode()
        # Estimated decoded size: len(b64) * 3 / 4
        estimated = len(b64) * 3 / 4
        actual = len(raw)
        # Should be within 1% (base64 padding adds at most 2 bytes)
        assert abs(estimated - actual) / actual < 0.01


# ---------------------------------------------------------------------------
# validate_image_url
# ---------------------------------------------------------------------------


class TestValidateImageUrl:
    """Tests for the image URL validator."""

    def test_passes_regular_http_url(self) -> None:
        """Regular HTTP URLs are passed through without validation."""
        validate_image_url("https://example.com/photo.jpg")  # should not raise

    def test_passes_http_url_no_extension(self) -> None:
        """HTTP URLs without file extension pass."""
        validate_image_url("https://example.com/image?id=123")  # should not raise

    def test_validates_valid_data_url(self) -> None:
        """Valid data URLs pass validation."""
        url = _make_data_url("image/png", 1024)
        with patch(_CONFIG_SETTINGS, return_value=_mock_settings(max_image_size_mb=20)):
            validate_image_url(url)  # should not raise

    def test_rejects_invalid_data_url(self) -> None:
        """Data URL with unsupported type raises."""
        url = _make_data_url("image/svg+xml", 100)
        with patch(_CONFIG_SETTINGS, return_value=_mock_settings(max_image_size_mb=20)):
            with pytest.raises(HTTPException) as exc_info:
                validate_image_url(url)
        assert exc_info.value.status_code == 422

    def test_rejects_oversized_data_url(self) -> None:
        """Data URL exceeding configured max_image_size_mb raises."""
        url = _make_data_url("image/png", 5 * 1024 * 1024)
        with patch(_CONFIG_SETTINGS, return_value=_mock_settings(max_image_size_mb=1)):
            with pytest.raises(HTTPException) as exc_info:
                validate_image_url(url)
        assert exc_info.value.status_code == 422
        assert "Image too large" in exc_info.value.detail

    def test_local_file_path_passes(self) -> None:
        """Local file paths (not data: or http:) pass through."""
        validate_image_url("/tmp/test.png")  # should not raise

    def test_uses_config_max_size(self) -> None:
        """validate_image_url reads max_image_size_mb from settings."""
        url = _make_data_url("image/png", 512)
        mock = _mock_settings(max_image_size_mb=50)
        with patch(_CONFIG_SETTINGS, return_value=mock) as mock_fn:
            validate_image_url(url)  # should not raise
            mock_fn.assert_called_once()
