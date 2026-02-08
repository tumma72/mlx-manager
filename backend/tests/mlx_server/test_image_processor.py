"""Tests for image preprocessing service (base64, URL fetch, resize, batch)."""

import base64
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from PIL import Image

from mlx_manager.mlx_server.services.image_processor import (
    MAX_IMAGE_DIMENSION,
    MAX_URL_RETRIES,
    _fetch_image_from_url,
    preprocess_image,
    preprocess_images,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"


def _make_png_base64(width: int = 50, height: int = 50, mode: str = "RGB") -> str:
    """Create a base64-encoded PNG data URI."""
    img = Image.new(mode, (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _make_png_bytes(width: int = 50, height: int = 50, mode: str = "RGB") -> bytes:
    """Create raw PNG bytes for mock HTTP responses."""
    img = Image.new(mode, (width, height), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestPreprocessImageBase64:
    """Tests for base64 data URI decoding in preprocess_image."""

    @pytest.mark.asyncio
    async def test_valid_base64_rgb(self):
        """Decode a valid RGB base64 PNG data URI."""
        data_uri = _make_png_base64(100, 100, "RGB")
        img = await preprocess_image(data_uri)
        assert isinstance(img, Image.Image)
        assert img.size == (100, 100)
        assert img.mode == "RGB"

    @pytest.mark.asyncio
    async def test_valid_base64_rgba_converts_to_rgb(self):
        """RGBA images are converted to RGB."""
        data_uri = _make_png_base64(50, 50, "RGBA")
        img = await preprocess_image(data_uri)
        assert img.mode == "RGB"

    @pytest.mark.asyncio
    async def test_valid_base64_palette_converts_to_rgb(self):
        """Palette (P) mode images are converted to RGB."""
        # Create a palette image
        palette_img = Image.new("P", (30, 30))
        buf = io.BytesIO()
        palette_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        img = await preprocess_image(data_uri)
        assert img.mode == "RGB"

    @pytest.mark.asyncio
    async def test_grayscale_image_preserved(self):
        """Grayscale (L) mode images are kept as-is (L is in allowed modes)."""
        gray_img = Image.new("L", (30, 30), color=128)
        buf = io.BytesIO()
        gray_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        img = await preprocess_image(data_uri)
        assert img.mode == "L"

    @pytest.mark.asyncio
    async def test_invalid_base64_raises_value_error(self):
        """Invalid base64 data raises ValueError."""
        with pytest.raises(ValueError, match="Failed to decode base64"):
            await preprocess_image("data:image/png;base64,not-valid-base64!!!")

    @pytest.mark.asyncio
    async def test_base64_missing_comma_raises_value_error(self):
        """Base64 URI without comma separator raises ValueError."""
        with pytest.raises(ValueError, match="Failed to decode base64"):
            await preprocess_image("data:image/png;base64")


class TestPreprocessImageResize:
    """Tests for auto-resizing in preprocess_image."""

    @pytest.mark.asyncio
    async def test_small_image_not_resized(self):
        """Images within max dimension are not resized."""
        data_uri = _make_png_base64(100, 100)
        img = await preprocess_image(data_uri, max_dimension=200)
        assert img.size == (100, 100)

    @pytest.mark.asyncio
    async def test_large_image_resized(self):
        """Images exceeding max dimension are scaled down proportionally."""
        data_uri = _make_png_base64(4000, 2000)
        img = await preprocess_image(data_uri, max_dimension=2048)
        # 4000 is max dim, scale = 2048/4000 = 0.512
        assert img.size[0] == int(4000 * (2048 / 4000))
        assert img.size[1] == int(2000 * (2048 / 4000))
        assert max(img.size) <= 2048

    @pytest.mark.asyncio
    async def test_exact_max_dimension_not_resized(self):
        """Image exactly at max dimension is not resized."""
        data_uri = _make_png_base64(2048, 1024)
        img = await preprocess_image(data_uri, max_dimension=2048)
        assert img.size == (2048, 1024)

    @pytest.mark.asyncio
    async def test_custom_max_dimension(self):
        """Custom max_dimension is respected."""
        data_uri = _make_png_base64(500, 300)
        img = await preprocess_image(data_uri, max_dimension=200)
        assert max(img.size) <= 200

    @pytest.mark.asyncio
    async def test_default_max_dimension(self):
        """Default MAX_IMAGE_DIMENSION is 2048."""
        assert MAX_IMAGE_DIMENSION == 2048


class TestPreprocessImageLocalFile:
    """Tests for local file path handling in preprocess_image."""

    @pytest.mark.asyncio
    async def test_load_local_png(self):
        """Load a real local PNG file from test fixtures."""
        img_path = str(FIXTURES_DIR / "red_square.png")
        img = await preprocess_image(img_path)
        assert isinstance(img, Image.Image)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file_raises(self):
        """Non-existent file raises ValueError."""
        with pytest.raises(ValueError, match="Failed to open image file"):
            await preprocess_image("/nonexistent/path/image.png")


class TestPreprocessImageURL:
    """Tests for URL-based image fetching in preprocess_image."""

    @pytest.mark.asyncio
    async def test_url_fetches_image(self):
        """URL input fetches and returns PIL Image."""
        png_bytes = _make_png_bytes(80, 80)

        mock_response = MagicMock()
        mock_response.content = png_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        img = await preprocess_image(
            "https://example.com/image.png",
            client=mock_client,
        )

        assert isinstance(img, Image.Image)
        assert img.size == (80, 80)
        mock_client.get.assert_called_once()


class TestFetchImageFromUrl:
    """Tests for _fetch_image_from_url with retries."""

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Successful fetch returns PIL Image."""
        png_bytes = _make_png_bytes(64, 64)

        mock_response = MagicMock()
        mock_response.content = png_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        img = await _fetch_image_from_url("https://example.com/img.png", client=mock_client)
        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)

    @pytest.mark.asyncio
    async def test_retry_on_request_error(self):
        """Retries on httpx.RequestError and succeeds on later attempt."""
        png_bytes = _make_png_bytes(32, 32)

        mock_response = MagicMock()
        mock_response.content = png_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            side_effect=[
                httpx.ConnectError("Connection refused"),
                mock_response,
            ]
        )
        mock_client.aclose = AsyncMock()

        with patch(
            "mlx_manager.mlx_server.services.image_processor.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            img = await _fetch_image_from_url("https://example.com/img.png", client=mock_client)

        assert isinstance(img, Image.Image)
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_http_status_error(self):
        """Retries on httpx.HTTPStatusError."""
        png_bytes = _make_png_bytes(32, 32)

        mock_good_response = MagicMock()
        mock_good_response.content = png_bytes
        mock_good_response.raise_for_status = MagicMock()

        mock_bad_response = MagicMock()
        mock_bad_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=[mock_bad_response, mock_good_response])
        mock_client.aclose = AsyncMock()

        with patch(
            "mlx_manager.mlx_server.services.image_processor.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            img = await _fetch_image_from_url("https://example.com/img.png", client=mock_client)

        assert isinstance(img, Image.Image)

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises(self):
        """After MAX_URL_RETRIES failures, raises ValueError."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.aclose = AsyncMock()

        with patch(
            "mlx_manager.mlx_server.services.image_processor.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            with pytest.raises(ValueError, match=f"after {MAX_URL_RETRIES} attempts"):
                await _fetch_image_from_url("https://example.com/img.png", client=mock_client)

        assert mock_client.get.call_count == MAX_URL_RETRIES

    @pytest.mark.asyncio
    async def test_creates_client_if_none_provided(self):
        """Creates and closes httpx client when none is provided."""
        png_bytes = _make_png_bytes(32, 32)

        mock_response = MagicMock()
        mock_response.content = png_bytes
        mock_response.raise_for_status = MagicMock()

        mock_instance = AsyncMock()
        mock_instance.get = AsyncMock(return_value=mock_response)
        mock_instance.aclose = AsyncMock()

        with patch(
            "mlx_manager.mlx_server.services.image_processor.httpx.AsyncClient",
            return_value=mock_instance,
        ) as mock_cls:
            await _fetch_image_from_url("https://example.com/img.png")

            mock_cls.assert_called_once()
            mock_instance.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_close_provided_client(self):
        """Does not close client when one is provided externally."""
        png_bytes = _make_png_bytes(32, 32)

        mock_response = MagicMock()
        mock_response.content = png_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        await _fetch_image_from_url("https://example.com/img.png", client=mock_client)

        mock_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_exponential_backoff_sleep_times(self):
        """Verify exponential backoff: 1s, 2s between retries."""
        png_bytes = _make_png_bytes(32, 32)

        mock_good_response = MagicMock()
        mock_good_response.content = png_bytes
        mock_good_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            side_effect=[
                httpx.ConnectError("fail1"),
                httpx.ConnectError("fail2"),
                mock_good_response,
            ]
        )
        mock_client.aclose = AsyncMock()

        sleep_times = []

        async def capture_sleep(seconds):
            sleep_times.append(seconds)

        with patch(
            "mlx_manager.mlx_server.services.image_processor.asyncio.sleep",
            side_effect=capture_sleep,
        ):
            await _fetch_image_from_url("https://example.com/img.png", client=mock_client)

        # 2^0 = 1, 2^1 = 2
        assert sleep_times == [1, 2]


class TestPreprocessImages:
    """Tests for batch image preprocessing."""

    @pytest.mark.asyncio
    async def test_batch_multiple_base64_images(self):
        """Process multiple base64 images concurrently."""
        uris = [_make_png_base64(50, 50) for _ in range(3)]
        images = await preprocess_images(uris)
        assert len(images) == 3
        for img in images:
            assert isinstance(img, Image.Image)
            assert img.size == (50, 50)

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        """Empty input list returns empty output list."""
        images = await preprocess_images([])
        assert images == []

    @pytest.mark.asyncio
    async def test_batch_single_image(self):
        """Single image in batch works correctly."""
        uris = [_make_png_base64(30, 30)]
        images = await preprocess_images(uris)
        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_batch_creates_client_if_none(self):
        """Creates and closes httpx client for batch when none provided."""
        uris = [_make_png_base64(20, 20)]

        mock_instance = AsyncMock()
        mock_instance.aclose = AsyncMock()

        with patch(
            "mlx_manager.mlx_server.services.image_processor.httpx.AsyncClient",
            return_value=mock_instance,
        ) as mock_cls:
            await preprocess_images(uris)

            mock_cls.assert_called_once()
            mock_instance.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_does_not_close_provided_client(self):
        """Does not close client when one is provided externally."""
        uris = [_make_png_base64(20, 20)]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        await preprocess_images(uris, client=mock_client)

        mock_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_with_mixed_types(self):
        """Mix of base64 and local file paths works correctly."""
        b64_uri = _make_png_base64(40, 40)
        local_path = str(FIXTURES_DIR / "red_square.png")

        images = await preprocess_images([b64_uri, local_path])
        assert len(images) == 2
        for img in images:
            assert isinstance(img, Image.Image)

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self):
        """Output images are in same order as input."""
        # Create images of different sizes
        uri_small = _make_png_base64(10, 10)
        uri_big = _make_png_base64(200, 200)
        uri_medium = _make_png_base64(100, 100)

        images = await preprocess_images([uri_small, uri_big, uri_medium])
        assert images[0].size == (10, 10)
        assert images[1].size == (200, 200)
        assert images[2].size == (100, 100)
