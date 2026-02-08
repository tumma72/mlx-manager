"""Tests for vision model inference."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from mlx_manager.mlx_server.services.vision import (
    _generate_vision_complete,
    _stream_vision_generate,
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
        from mlx_manager.mlx_server.models.types import ModelType

        # Setup mock
        mock_pool = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.tokenizer = MagicMock()  # processor
        mock_loaded.model_type = ModelType.VISION.value  # Set model type for type check
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
        from mlx_manager.mlx_server.models.types import ModelType

        # Setup mock
        mock_pool = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.tokenizer = MagicMock()
        mock_loaded.model_type = ModelType.VISION.value  # Set model type for type check
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


class TestGenerateVisionCompletionOrchestrator:
    """Tests for the generate_vision_completion orchestrator."""

    @pytest.mark.asyncio
    async def test_non_vision_model_type_raises(self):
        """Verify RuntimeError when model is not a vision type."""
        from mlx_manager.mlx_server.models.types import ModelType

        mock_pool = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model_type = ModelType.TEXT_GEN.value
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with pytest.raises(RuntimeError, match="was loaded as"):
                await generate_vision_completion(
                    model_id="text-model",
                    text_prompt="What is this?",
                    images=[create_test_image()],
                    stream=False,
                )

    @pytest.mark.asyncio
    async def test_non_streaming_calls_generate_vision_complete(self):
        """Non-streaming mode calls _generate_vision_complete."""
        from mlx_manager.mlx_server.models.types import ModelType

        mock_pool = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.tokenizer = MagicMock()
        mock_loaded.model_type = ModelType.VISION.value
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with patch(
                "mlx_manager.mlx_server.services.vision._generate_vision_complete",
                new_callable=AsyncMock,
            ) as mock_complete:
                mock_complete.return_value = {
                    "id": "test",
                    "choices": [{"message": {"content": "test"}}],
                }

                result = await generate_vision_completion(
                    model_id="vision-model",
                    text_prompt="Describe this",
                    images=[create_test_image()],
                    stream=False,
                    max_tokens=100,
                    temperature=0.5,
                )

                mock_complete.assert_called_once()
                assert result["id"] == "test"

    @pytest.mark.asyncio
    async def test_streaming_calls_stream_vision_generate(self):
        """Streaming mode calls _stream_vision_generate."""
        from mlx_manager.mlx_server.models.types import ModelType

        mock_pool = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.tokenizer = MagicMock()
        mock_loaded.model_type = ModelType.VISION.value
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def mock_stream(*args, **kwargs):
            yield {"id": "chunk1"}

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with patch(
                "mlx_manager.mlx_server.services.vision._stream_vision_generate",
                side_effect=mock_stream,
            ):
                result = await generate_vision_completion(
                    model_id="vision-model",
                    text_prompt="Describe this",
                    images=[create_test_image()],
                    stream=True,
                )

                import inspect

                assert inspect.isasyncgen(result)


class TestStreamVisionGenerate:
    """Tests for _stream_vision_generate async generator."""

    @pytest.mark.asyncio
    async def test_stream_yields_three_chunks(self):
        """Streaming yields role chunk, content chunk, and finish chunk."""
        mock_model = MagicMock()
        mock_processor = MagicMock()

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx_vlm.generate") as mock_vlm_gen,
            patch("mlx_vlm.prompt_utils.apply_chat_template", return_value="<formatted>"),
            patch("mlx_vlm.utils.load_config", return_value={}),
        ):
            mock_response = MagicMock()
            mock_response.text = "This is a blue square."
            mock_vlm_gen.return_value = mock_response

            chunks = []
            async for chunk in _stream_vision_generate(
                model=mock_model,
                processor=mock_processor,
                text_prompt="What is this?",
                images=[create_test_image()],
                max_tokens=100,
                temperature=0.7,
                completion_id="chatcmpl-test123",
                created=1234567890,
                model_id="test-vision-model",
            ):
                chunks.append(chunk)

            assert len(chunks) == 3

            # First chunk: role
            assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
            assert chunks[0]["choices"][0]["delta"]["content"] == ""
            assert chunks[0]["choices"][0]["finish_reason"] is None

            # Second chunk: content
            assert chunks[1]["choices"][0]["delta"]["content"] == "This is a blue square."
            assert chunks[1]["choices"][0]["finish_reason"] is None

            # Third chunk: finish
            assert chunks[2]["choices"][0]["delta"] == {}
            assert chunks[2]["choices"][0]["finish_reason"] == "stop"

            # All chunks share same id and model
            for c in chunks:
                assert c["id"] == "chatcmpl-test123"
                assert c["model"] == "test-vision-model"
                assert c["object"] == "chat.completion.chunk"
                assert c["created"] == 1234567890

    @pytest.mark.asyncio
    async def test_stream_calls_vlm_generate_correctly(self):
        """Verify vlm.generate is called with correct params."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        test_images = [create_test_image(), create_test_image()]

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx_vlm.generate") as mock_vlm_gen,
            patch("mlx_vlm.prompt_utils.apply_chat_template", return_value="<prompt>"),
            patch("mlx_vlm.utils.load_config", return_value={"model_type": "qwen2_vl"}),
        ):
            mock_response = MagicMock()
            mock_response.text = "Result"
            mock_vlm_gen.return_value = mock_response

            async for _ in _stream_vision_generate(
                model=mock_model,
                processor=mock_processor,
                text_prompt="Compare these",
                images=test_images,
                max_tokens=200,
                temperature=0.5,
                completion_id="id",
                created=0,
                model_id="test-model",
            ):
                pass

            mock_vlm_gen.assert_called_once_with(
                mock_model,
                mock_processor,
                "<prompt>",
                test_images,
                max_tokens=200,
                temp=0.5,
                verbose=False,
            )

    @pytest.mark.asyncio
    async def test_stream_loads_config_for_chat_template(self):
        """Verify load_config is called with model_id."""
        mock_model = MagicMock()
        mock_processor = MagicMock()

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx_vlm.generate") as mock_vlm_gen,
            patch("mlx_vlm.prompt_utils.apply_chat_template") as mock_template,
            patch("mlx_vlm.utils.load_config") as mock_load_config,
        ):
            mock_config = {"model_type": "test"}
            mock_load_config.return_value = mock_config
            mock_template.return_value = "<formatted>"

            mock_response = MagicMock()
            mock_response.text = "OK"
            mock_vlm_gen.return_value = mock_response

            async for _ in _stream_vision_generate(
                model=mock_model,
                processor=mock_processor,
                text_prompt="test prompt",
                images=[create_test_image()],
                max_tokens=50,
                temperature=0.3,
                completion_id="id",
                created=0,
                model_id="my-vision/model",
            ):
                pass

            mock_load_config.assert_called_once_with("my-vision/model")
            mock_template.assert_called_once_with(
                mock_processor, mock_config, "test prompt", num_images=1
            )


class TestGenerateVisionComplete:
    """Tests for _generate_vision_complete non-streaming function."""

    @pytest.mark.asyncio
    async def test_complete_returns_full_response(self):
        """Non-streaming returns a full chat.completion response."""
        mock_model = MagicMock()
        mock_processor = MagicMock()

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx_vlm.generate") as mock_vlm_gen,
            patch("mlx_vlm.prompt_utils.apply_chat_template", return_value="<prompt>"),
            patch("mlx_vlm.utils.load_config", return_value={}),
        ):
            mock_response = MagicMock()
            mock_response.text = "This is a cat sitting on a mat."
            mock_vlm_gen.return_value = mock_response

            result = await _generate_vision_complete(
                model=mock_model,
                processor=mock_processor,
                text_prompt="What is in this image?",
                images=[create_test_image()],
                max_tokens=100,
                temperature=0.7,
                completion_id="chatcmpl-abc123",
                created=1700000000,
                model_id="test-vision-model",
            )

            assert result["id"] == "chatcmpl-abc123"
            assert result["object"] == "chat.completion"
            assert result["created"] == 1700000000
            assert result["model"] == "test-vision-model"
            assert result["choices"][0]["index"] == 0
            assert result["choices"][0]["message"]["role"] == "assistant"
            assert result["choices"][0]["message"]["content"] == "This is a cat sitting on a mat."
            assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_complete_usage_token_estimation(self):
        """Non-streaming estimates token counts in usage."""
        mock_model = MagicMock()
        mock_processor = MagicMock()

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx_vlm.generate") as mock_vlm_gen,
            patch("mlx_vlm.prompt_utils.apply_chat_template", return_value="<prompt>"),
            patch("mlx_vlm.utils.load_config", return_value={}),
        ):
            mock_response = MagicMock()
            mock_response.text = "A blue square on white background"  # 6 words
            mock_vlm_gen.return_value = mock_response

            result = await _generate_vision_complete(
                model=mock_model,
                processor=mock_processor,
                text_prompt="What is this?",  # 3 words
                images=[create_test_image(), create_test_image()],  # 2 images * 256 = 512
                max_tokens=100,
                temperature=0.7,
                completion_id="test",
                created=0,
                model_id="test",
            )

            usage = result["usage"]
            assert usage["completion_tokens"] == 6  # 6 words
            # prompt_tokens = 3 words + 2 images * 256
            assert usage["prompt_tokens"] == 3 + 2 * 256
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.asyncio
    async def test_complete_calls_vlm_generate_correctly(self):
        """Verify non-streaming calls vlm.generate with correct params."""
        mock_model = MagicMock()
        mock_processor = MagicMock()
        test_images = [create_test_image()]

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx_vlm.generate") as mock_vlm_gen,
            patch("mlx_vlm.prompt_utils.apply_chat_template", return_value="<prompt>"),
            patch("mlx_vlm.utils.load_config", return_value={}),
        ):
            mock_response = MagicMock()
            mock_response.text = "Result"
            mock_vlm_gen.return_value = mock_response

            await _generate_vision_complete(
                model=mock_model,
                processor=mock_processor,
                text_prompt="Describe this",
                images=test_images,
                max_tokens=512,
                temperature=0.3,
                completion_id="test",
                created=0,
                model_id="test-model",
            )

            mock_vlm_gen.assert_called_once_with(
                mock_model,
                mock_processor,
                "<prompt>",
                test_images,
                max_tokens=512,
                temp=0.3,
                verbose=False,
            )

    @pytest.mark.asyncio
    async def test_complete_metal_thread_error_propagates(self):
        """Errors from run_on_metal_thread propagate."""

        async def raise_error(fn, **kwargs):
            raise RuntimeError("Vision generation failed: timeout")

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=raise_error,
        ):
            with pytest.raises(RuntimeError, match="Vision generation failed"):
                await _generate_vision_complete(
                    model=MagicMock(),
                    processor=MagicMock(),
                    text_prompt="test",
                    images=[create_test_image()],
                    max_tokens=100,
                    temperature=0.7,
                    completion_id="test",
                    created=0,
                    model_id="test",
                )
