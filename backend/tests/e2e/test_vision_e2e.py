"""End-to-end tests for vision/multimodal model inference.

Tests validate the complete pipeline:
1. HTTP request with base64 image content
2. Model type detection (VISION)
3. Image preprocessing (PIL conversion, resize)
4. mlx-vlm inference (generate_vision_completion)
5. Response generation (ChatCompletionResponse)

Tiered approach:
- Quick tier (@e2e_vision_quick): Qwen2-VL-2B / Qwen3-VL-8B / Gemma-3-12b (first available)
- Full tier (@e2e_vision_full): Gemma-3-27b (~15GB, thorough)

Run:
  pytest -m e2e_vision_quick   # Quick tests only
  pytest -m e2e_vision_full    # Full tests only
  pytest -m e2e_vision         # All vision E2E
"""

import base64
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden" / "vision"
IMAGES_DIR = FIXTURES_DIR / "images"


def load_image_base64(filename: str) -> str:
    """Load a test image as base64 data URI."""
    path = IMAGES_DIR / filename
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def load_golden_prompt(name: str) -> str:
    """Load a golden prompt fixture."""
    return (GOLDEN_DIR / f"{name}.txt").read_text().strip()


def build_vision_request(
    model: str,
    prompt: str,
    image_urls: list[str],
    max_tokens: int = 256,
    stream: bool = False,
) -> dict:
    """Build an OpenAI-compatible chat completion request with images."""
    content: list[dict] = []
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": prompt})

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful vision assistant. Be concise."},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.1,  # Low temperature for deterministic outputs
    }


# ──────────────────────────────────────────────
# Quick tier: first available vision model (Qwen2-VL-2B / Qwen3-VL-8B / Gemma-3-12b)
# ──────────────────────────────────────────────


@pytest.mark.e2e
@pytest.mark.e2e_vision
@pytest.mark.e2e_vision_quick
class TestVisionQuickDescribe:
    """Test single-image description with quick model."""

    async def test_describe_red_square(self, app_client, vision_model_quick):
        """Vision model should describe a red square image."""
        prompt = load_golden_prompt("describe_image")
        image = load_image_base64("red_square.png")

        request = build_vision_request(vision_model_quick, prompt, [image])
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"

        content = data["choices"][0]["message"]["content"].lower()
        # Model should mention red and/or square
        assert "red" in content or "square" in content, (
            f"Expected 'red' or 'square' in response, got: {content}"
        )

    async def test_describe_blue_circle(self, app_client, vision_model_quick):
        """Vision model should describe a blue circle image."""
        prompt = load_golden_prompt("describe_image")
        image = load_image_base64("blue_circle.png")

        request = build_vision_request(vision_model_quick, prompt, [image])
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"].lower()
        assert "blue" in content or "circle" in content, (
            f"Expected 'blue' or 'circle' in response, got: {content}"
        )


@pytest.mark.e2e
@pytest.mark.e2e_vision
@pytest.mark.e2e_vision_quick
class TestVisionQuickMultiImage:
    """Test multi-image comparison with quick model."""

    async def test_compare_two_images(self, app_client, vision_model_quick):
        """Vision model should compare two different images."""
        prompt = load_golden_prompt("compare_images")
        img1 = load_image_base64("red_square.png")
        img2 = load_image_base64("blue_circle.png")

        request = build_vision_request(vision_model_quick, prompt, [img1, img2])
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"].lower()
        # Should mention both shapes or both colors
        has_colors = "red" in content and "blue" in content
        has_shapes = "square" in content and "circle" in content
        assert has_colors or has_shapes, (
            f"Expected comparison mentioning both images, got: {content}"
        )


@pytest.mark.e2e
@pytest.mark.e2e_vision
@pytest.mark.e2e_vision_quick
class TestVisionQuickOCR:
    """Test OCR/text reading with quick model."""

    async def test_read_text_from_image(self, app_client, vision_model_quick):
        """Vision model should read text from an image."""
        prompt = load_golden_prompt("ocr_text")
        image = load_image_base64("text_sample.png")

        request = build_vision_request(vision_model_quick, prompt, [image])
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        # Should contain "Hello" or "MLX" (case-insensitive)
        assert "hello" in content.lower() or "mlx" in content.lower(), (
            f"Expected 'Hello' or 'MLX' in OCR result, got: {content}"
        )


@pytest.mark.e2e
@pytest.mark.e2e_vision
@pytest.mark.e2e_vision_quick
class TestVisionQuickStreaming:
    """Test streaming vision responses with quick model."""

    async def test_streaming_vision_response(self, app_client, vision_model_quick):
        """Vision model should stream response chunks via SSE."""
        prompt = load_golden_prompt("describe_image")
        image = load_image_base64("red_square.png")

        request = build_vision_request(vision_model_quick, prompt, [image], stream=True)
        async with app_client.stream("POST", "/v1/chat/completions", json=request) as response:
            assert response.status_code == 200

            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(line[6:])

            # Should have received at least one chunk
            assert len(chunks) > 0, "Expected streaming chunks but got none"


@pytest.mark.e2e
@pytest.mark.e2e_vision
@pytest.mark.e2e_vision_quick
class TestVisionQuickErrorHandling:
    """Test error handling for vision requests."""

    async def test_text_model_rejects_images(self, app_client):
        """Sending images to a text-only model should return 400."""
        image = load_image_base64("red_square.png")
        request = build_vision_request(
            "mlx-community/Qwen3-0.6B-4bit-DWQ",  # Text-only model
            "Describe this image",
            [image],
        )
        response = await app_client.post("/v1/chat/completions", json=request)
        assert response.status_code == 400


# ──────────────────────────────────────────────
# Full tier: Gemma-3-27b-it-4bit-DWQ
# ──────────────────────────────────────────────


@pytest.mark.e2e
@pytest.mark.e2e_vision
@pytest.mark.e2e_vision_full
class TestVisionFullDescribe:
    """Test single-image description with full model (more accurate)."""

    async def test_describe_red_square(self, app_client, vision_model_full):
        """Full model should accurately describe red square."""
        prompt = load_golden_prompt("describe_image")
        image = load_image_base64("red_square.png")

        request = build_vision_request(vision_model_full, prompt, [image])
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"].lower()
        # Full model should identify BOTH color and shape
        assert "red" in content, f"Expected 'red' in response, got: {content}"
        assert "square" in content or "rectangle" in content, (
            f"Expected 'square' or 'rectangle' in response, got: {content}"
        )

    async def test_describe_blue_circle(self, app_client, vision_model_full):
        """Full model should accurately describe blue circle."""
        prompt = load_golden_prompt("describe_image")
        image = load_image_base64("blue_circle.png")

        request = build_vision_request(vision_model_full, prompt, [image])
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"].lower()
        assert "blue" in content, f"Expected 'blue' in response, got: {content}"
        assert "circle" in content or "round" in content, (
            f"Expected 'circle' or 'round' in response, got: {content}"
        )


@pytest.mark.e2e
@pytest.mark.e2e_vision
@pytest.mark.e2e_vision_full
class TestVisionFullOCR:
    """Test OCR with full model (higher accuracy expected)."""

    async def test_read_text_accurately(self, app_client, vision_model_full):
        """Full model should read 'Hello MLX' text accurately."""
        prompt = load_golden_prompt("ocr_text")
        image = load_image_base64("text_sample.png")

        request = build_vision_request(vision_model_full, prompt, [image])
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        # Full model should get the complete text
        assert "Hello" in content and "MLX" in content, (
            f"Expected 'Hello MLX' in OCR result, got: {content}"
        )


@pytest.mark.e2e
@pytest.mark.e2e_vision
@pytest.mark.e2e_vision_full
class TestVisionFullMultiImage:
    """Test multi-image comparison with full model."""

    async def test_detailed_comparison(self, app_client, vision_model_full):
        """Full model should provide detailed comparison of two images."""
        prompt = load_golden_prompt("compare_images")
        img1 = load_image_base64("red_square.png")
        img2 = load_image_base64("blue_circle.png")

        request = build_vision_request(vision_model_full, prompt, [img1, img2])
        response = await app_client.post("/v1/chat/completions", json=request)

        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"].lower()
        # Full model should mention all 4 attributes
        assert "red" in content, "Should mention red color"
        assert "blue" in content, "Should mention blue color"
        assert "square" in content or "rectangle" in content, "Should mention square shape"
        assert "circle" in content or "round" in content, "Should mention circle shape"
