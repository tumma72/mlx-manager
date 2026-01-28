"""Image preprocessing service for vision models.

Handles:
- Base64 data URI decoding
- URL fetching with retry
- Image resizing to max dimension
- Batch image processing
"""

import asyncio
import base64
import logging
from io import BytesIO

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_DIMENSION = 2048  # Max width or height in pixels
URL_FETCH_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
MAX_URL_RETRIES = 3


async def preprocess_image(
    image_input: str,
    client: httpx.AsyncClient | None = None,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> Image.Image:
    """Fetch, decode, and resize an image.

    Args:
        image_input: One of:
            - Base64 data URI: "data:image/png;base64,<data>"
            - HTTP(S) URL: "https://example.com/image.jpg"
            - Local file path: "/path/to/image.jpg"
        client: Optional httpx.AsyncClient for URL fetching (created if not provided)
        max_dimension: Maximum width or height (default 2048px)

    Returns:
        PIL Image object, resized if necessary

    Raises:
        ValueError: If image cannot be decoded or fetched
    """
    if image_input.startswith("data:"):
        # Base64 data URI: "data:image/png;base64,<data>"
        try:
            # Extract base64 data after the comma
            _, data = image_input.split(",", 1)
            img_bytes = base64.b64decode(data)
            img = Image.open(BytesIO(img_bytes))
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}") from e

    elif image_input.startswith(("http://", "https://")):
        # URL - fetch with retry
        img = await _fetch_image_from_url(image_input, client)

    else:
        # Assume local file path
        try:
            img = Image.open(image_input)
        except Exception as e:
            raise ValueError(f"Failed to open image file {image_input}: {e}") from e

    # Convert to RGB if needed (removes alpha channel, handles palette images)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Auto-resize if exceeds max dimension
    max_dim = max(img.size)
    if max_dim > max_dimension:
        scale = max_dimension / max_dim
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        original_size = img.size
        img = img.resize(new_size, Image.LANCZOS)
        logger.warning(
            f"Image resized from {original_size[0]}x{original_size[1]} to "
            f"{new_size[0]}x{new_size[1]} (max dimension: {max_dimension}px)"
        )

    return img


async def _fetch_image_from_url(
    url: str,
    client: httpx.AsyncClient | None = None,
) -> Image.Image:
    """Fetch image from URL with retry and timeout.

    Args:
        url: HTTP(S) URL to fetch
        client: Optional httpx.AsyncClient (created if not provided)

    Returns:
        PIL Image object

    Raises:
        ValueError: If image cannot be fetched after retries
    """
    should_close = False
    if client is None:
        client = httpx.AsyncClient()
        should_close = True

    try:
        last_error: Exception | None = None
        for attempt in range(MAX_URL_RETRIES):
            try:
                response = await client.get(url, timeout=URL_FETCH_TIMEOUT, follow_redirects=True)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < MAX_URL_RETRIES - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.warning(
                        f"Failed to fetch {url} (attempt {attempt + 1}/{MAX_URL_RETRIES}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)

        raise ValueError(
            f"Failed to fetch image from {url} after {MAX_URL_RETRIES} attempts: {last_error}"
        )
    finally:
        if should_close:
            await client.aclose()


async def preprocess_images(
    image_inputs: list[str],
    client: httpx.AsyncClient | None = None,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> list[Image.Image]:
    """Process multiple images concurrently.

    Args:
        image_inputs: List of image sources (base64, URLs, or paths)
        client: Optional shared httpx.AsyncClient
        max_dimension: Maximum width or height

    Returns:
        List of PIL Image objects in same order as inputs
    """
    should_close = False
    if client is None:
        client = httpx.AsyncClient()
        should_close = True

    try:
        tasks = [
            preprocess_image(img_input, client, max_dimension) for img_input in image_inputs
        ]
        return await asyncio.gather(*tasks)
    finally:
        if should_close:
            await client.aclose()
