"""Input validation utilities for MLX Server.

Provides validation for base64 images and image URLs used in
vision/multimodal requests.
"""

import re

from fastapi import HTTPException

# Known image MIME types accepted for vision requests
VALID_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


def validate_base64_image(data_url: str, max_size_mb: int = 20) -> None:
    """Validate a base64-encoded image data URL.

    Checks:
    1. Valid data URL format with an image MIME type
    2. Decoded size is within the configured limit

    Args:
        data_url: The full data URL string (e.g. "data:image/png;base64,...").
        max_size_mb: Maximum decoded image size in megabytes.

    Raises:
        HTTPException: 422 if MIME type is unsupported or image exceeds size limit.
    """
    # Check data URL format
    match = re.match(r"data:(image/[\w+.-]+);base64,(.+)", data_url, re.DOTALL)
    if not match:
        # Not a recognized data URL -- could be a regular URL, skip validation
        return

    mime_type = match.group(1)
    if mime_type not in VALID_IMAGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported image type: {mime_type}. "
            f"Supported: {', '.join(sorted(VALID_IMAGE_TYPES))}",
        )

    # Estimate decoded size (base64 encodes 3 bytes as 4 chars)
    b64_data = match.group(2)
    estimated_bytes = len(b64_data) * 3 / 4
    max_bytes = max_size_mb * 1024 * 1024

    if estimated_bytes > max_bytes:
        size_mb = estimated_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=422,
            detail=f"Image too large: {size_mb:.1f}MB exceeds maximum of {max_size_mb}MB",
        )


def validate_image_url(url: str) -> None:
    """Validate an image URL for vision requests.

    For regular HTTP(S) URLs, no validation is performed (the image
    processor handles fetch errors).  For data URLs, validates the MIME
    type and decoded size against the server configuration.

    Args:
        url: The image URL or data URL to validate.

    Raises:
        HTTPException: 422 if the data URL has an unsupported MIME type or
            exceeds the configured size limit.
    """
    if url.startswith("data:"):
        from mlx_manager.mlx_server.config import get_settings

        validate_base64_image(url, max_size_mb=get_settings().max_image_size_mb)
