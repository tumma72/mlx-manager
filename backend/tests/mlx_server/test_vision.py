"""Tests for vision model inference."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from mlx_manager.mlx_server.services.vision import (
    generate_vision_completion,
)


def create_test_image(width: int = 100, height: int = 100) -> Image.Image:
    """Create a test PIL Image."""
    return Image.new("RGB", (width, height), color="blue")


class TestVisionService:
    """Tests for vision generation service."""

    @pytest.mark.asyncio
    async def test_generate_vision_completion_calls_pool(self):
        """Verify vision completion fetches model from pool."""
        # Setup mock
        mock_pool = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.tokenizer = MagicMock()  # processor
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        images = [create_test_image()]

        # Mock the pool module import and the actual generation
        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with patch(
                "mlx_manager.mlx_server.services.vision._generate_vision_complete"
            ) as mock_gen:
                mock_gen.return_value = {
                    "id": "test",
                    "choices": [{"message": {"content": "Test response"}}],
                }

                await generate_vision_completion(
                    model_id="test-vision-model",
                    text_prompt="What is in this image?",
                    images=images,
                    stream=False,
                )

                mock_pool.get_model.assert_called_once_with("test-vision-model")

    @pytest.mark.asyncio
    async def test_generate_vision_completion_stream_mode(self):
        """Verify streaming returns async generator."""
        # Setup mock
        mock_pool = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.tokenizer = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        images = [create_test_image()]

        # Mock the pool module import
        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            # Mock the streaming generation
            async def mock_stream(*args, **kwargs):
                yield {"id": "test", "choices": [{"delta": {"content": "Hello"}}]}

            with patch(
                "mlx_manager.mlx_server.services.vision._stream_vision_generate",
                side_effect=mock_stream,
            ):
                result = await generate_vision_completion(
                    model_id="test-vision-model",
                    text_prompt="What is in this image?",
                    images=images,
                    stream=True,
                )

                # Result should be an async generator
                import inspect

                assert inspect.isasyncgen(result), "Stream mode should return async generator"


class TestChatVisionIntegration:
    """Tests for chat endpoint vision handling."""

    @pytest.mark.asyncio
    async def test_text_model_with_images_returns_400(self):
        """Verify 400 error when sending images to text-only model."""
        from fastapi import HTTPException

        from mlx_manager.mlx_server.api.v1.chat import create_chat_completion
        from mlx_manager.mlx_server.schemas.openai import (
            ChatCompletionRequest,
            ChatMessage,
        )

        # Request with images to a text model
        request = ChatCompletionRequest(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",  # Text model
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc123"},
                        },
                    ],
                )
            ],
        )

        # Mock pool and detection
        with patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type") as mock_detect:
            from mlx_manager.mlx_server.models.types import ModelType

            mock_detect.return_value = ModelType.TEXT_GEN

            with pytest.raises(HTTPException) as exc_info:
                await create_chat_completion(request)

            assert exc_info.value.status_code == 400
            assert "vision model" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_text_only_request_uses_inference_service(self):
        """Verify text-only requests use generate_chat_completion."""
        from mlx_manager.mlx_server.api.v1.chat import create_chat_completion
        from mlx_manager.mlx_server.schemas.openai import (
            ChatCompletionRequest,
            ChatMessage,
        )

        request = ChatCompletionRequest(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            messages=[ChatMessage(role="user", content="Hello, how are you?")],
        )

        with patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion") as mock_gen:
            mock_gen.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "I'm fine!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

            result = await create_chat_completion(request)

            mock_gen.assert_called_once()
            # Verify vision service was NOT called
            assert "I'm fine!" in str(result)

    @pytest.mark.asyncio
    async def test_vision_request_routes_to_vision_service(self):
        """Verify vision requests use generate_vision_completion."""
        from mlx_manager.mlx_server.api.v1.chat import create_chat_completion
        from mlx_manager.mlx_server.models.types import ModelType
        from mlx_manager.mlx_server.schemas.openai import (
            ChatCompletionRequest,
            ChatMessage,
        )

        # Request with images to a vision model
        request = ChatCompletionRequest(
            model="mlx-community/Qwen2-VL-2B-Instruct-4bit",  # Vision model
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc123"},
                        },
                    ],
                )
            ],
        )

        with patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type") as mock_detect:
            mock_detect.return_value = ModelType.VISION

            with patch("mlx_manager.mlx_server.api.v1.chat.preprocess_images") as mock_preprocess:
                mock_preprocess.return_value = [create_test_image()]

                with patch(
                    "mlx_manager.mlx_server.api.v1.chat.generate_vision_completion"
                ) as mock_vision:
                    mock_vision.return_value = {
                        "id": "test-vision",
                        "object": "chat.completion",
                        "created": 1234567890,
                        "model": "test",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "This is a blue image.",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 266,
                            "completion_tokens": 5,
                            "total_tokens": 271,
                        },
                    }

                    result = await create_chat_completion(request)

                    # Verify vision service was called
                    mock_vision.assert_called_once()
                    assert "blue image" in str(result)

    @pytest.mark.asyncio
    async def test_multiple_images_in_request(self):
        """Verify multiple images are extracted and passed to vision service."""
        from mlx_manager.mlx_server.api.v1.chat import create_chat_completion
        from mlx_manager.mlx_server.models.types import ModelType
        from mlx_manager.mlx_server.schemas.openai import (
            ChatCompletionRequest,
            ChatMessage,
        )

        # Request with multiple images
        request = ChatCompletionRequest(
            model="mlx-community/Qwen2-VL-2B-Instruct-4bit",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Compare these images"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,img2"}},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,img3"}},
                    ],
                )
            ],
        )

        with patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type") as mock_detect:
            mock_detect.return_value = ModelType.VISION

            with patch("mlx_manager.mlx_server.api.v1.chat.preprocess_images") as mock_preprocess:
                # Return 3 images
                mock_preprocess.return_value = [
                    create_test_image(),
                    create_test_image(),
                    create_test_image(),
                ]

                with patch(
                    "mlx_manager.mlx_server.api.v1.chat.generate_vision_completion"
                ) as mock_vision:
                    mock_vision.return_value = {
                        "id": "test",
                        "object": "chat.completion",
                        "created": 1234567890,
                        "model": "test",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "All are blue."},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 780,
                            "completion_tokens": 4,
                            "total_tokens": 784,
                        },
                    }

                    await create_chat_completion(request)

                    # Verify preprocess was called with 3 URLs
                    mock_preprocess.assert_called_once()
                    urls = mock_preprocess.call_args[0][0]
                    assert len(urls) == 3

                    # Verify vision service was called with 3 images
                    mock_vision.assert_called_once()
                    images_arg = mock_vision.call_args[1]["images"]
                    assert len(images_arg) == 3

    @pytest.mark.asyncio
    async def test_embeddings_model_with_images_returns_400(self):
        """Verify 400 error when sending images to embeddings model."""
        from fastapi import HTTPException

        from mlx_manager.mlx_server.api.v1.chat import create_chat_completion
        from mlx_manager.mlx_server.models.types import ModelType
        from mlx_manager.mlx_server.schemas.openai import (
            ChatCompletionRequest,
            ChatMessage,
        )

        request = ChatCompletionRequest(
            model="mlx-community/bge-large",  # Embeddings model
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Embed this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                )
            ],
        )

        with patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type") as mock_detect:
            mock_detect.return_value = ModelType.EMBEDDINGS

            with pytest.raises(HTTPException) as exc_info:
                await create_chat_completion(request)

            assert exc_info.value.status_code == 400
            assert "vision model" in exc_info.value.detail.lower()
