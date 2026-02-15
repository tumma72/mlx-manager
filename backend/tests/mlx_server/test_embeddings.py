"""Tests for embeddings endpoint and service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.api.v1.embeddings import create_embeddings
from mlx_manager.mlx_server.models.ir import EmbeddingResult
from mlx_manager.mlx_server.schemas.openai import (
    EmbeddingRequest,
    EmbeddingResponse,
)


class TestEmbeddingsSchemas:
    """Tests for embeddings schemas."""

    def test_embedding_request_single_input(self):
        """Test EmbeddingRequest with single string."""
        req = EmbeddingRequest(input="Hello world", model="test-model")
        assert req.input == "Hello world"

    def test_embedding_request_batch_input(self):
        """Test EmbeddingRequest with list of strings."""
        req = EmbeddingRequest(
            input=["Hello", "World"],
            model="test-model",
        )
        assert isinstance(req.input, list)
        assert len(req.input) == 2

    def test_embedding_response_structure(self):
        """Test EmbeddingResponse structure."""
        from mlx_manager.mlx_server.schemas.openai import (
            EmbeddingData,
            EmbeddingUsage,
        )

        resp = EmbeddingResponse(
            data=[
                EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
                EmbeddingData(embedding=[0.4, 0.5, 0.6], index=1),
            ],
            model="test-model",
            usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
        )

        assert resp.object == "list"
        assert len(resp.data) == 2
        assert resp.data[0].object == "embedding"
        assert len(resp.data[0].embedding) == 3


class TestEmbeddingsEndpoint:
    """Tests for /v1/embeddings endpoint."""

    @pytest.mark.asyncio
    async def test_non_embedding_model_returns_400(self):
        """Verify 400 error when using non-embedding model."""
        from fastapi import HTTPException

        request = EmbeddingRequest(
            input="Hello world",
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",  # Text model
        )

        with patch("mlx_manager.mlx_server.api.v1.embeddings.detect_model_type") as mock_detect:
            from mlx_manager.mlx_server.models.types import ModelType

            mock_detect.return_value = ModelType.TEXT_GEN

            with pytest.raises(HTTPException) as exc_info:
                await create_embeddings(request)

            assert exc_info.value.status_code == 400
            assert "embedding" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_empty_input_returns_400(self):
        """Verify 400 error when input is empty list."""
        from fastapi import HTTPException

        request = EmbeddingRequest(
            input=[],
            model="mlx-community/all-MiniLM-L6-v2-4bit",
        )

        with patch("mlx_manager.mlx_server.api.v1.embeddings.detect_model_type") as mock_detect:
            from mlx_manager.mlx_server.models.types import ModelType

            mock_detect.return_value = ModelType.EMBEDDINGS

            with pytest.raises(HTTPException) as exc_info:
                await create_embeddings(request)

            assert exc_info.value.status_code == 400
            assert "empty" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_successful_embedding_generation(self):
        """Test successful embedding generation."""
        request = EmbeddingRequest(
            input=["Hello", "World"],
            model="mlx-community/all-MiniLM-L6-v2-4bit",
        )

        with patch("mlx_manager.mlx_server.api.v1.embeddings.detect_model_type") as mock_detect:
            from mlx_manager.mlx_server.models.types import ModelType

            mock_detect.return_value = ModelType.EMBEDDINGS

            with patch("mlx_manager.mlx_server.api.v1.embeddings.generate_embeddings") as mock_gen:
                # Return mock embeddings
                mock_gen.return_value = EmbeddingResult(
                    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    dimensions=3,
                    total_tokens=10,
                    finish_reason="stop",
                )

                response = await create_embeddings(request)

                assert isinstance(response, EmbeddingResponse)
                assert len(response.data) == 2
                assert response.data[0].index == 0
                assert response.data[1].index == 1
                assert response.usage.total_tokens == 10


class TestEmbeddingsService:
    """Tests for embeddings service."""

    def test_generate_embeddings_function_signature(self):
        """Verify generate_embeddings has correct function signature."""
        import inspect

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        sig = inspect.signature(generate_embeddings)
        params = list(sig.parameters.keys())
        assert "model_id" in params
        assert "texts" in params

    def test_generate_embeddings_is_async(self):
        """Verify generate_embeddings is an async function."""
        import inspect

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        assert inspect.iscoroutinefunction(generate_embeddings)

    def test_service_delegates_to_adapter(self):
        """Verify service delegates to adapter (not direct mlx calls)."""
        import inspect

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        source = inspect.getsource(generate_embeddings)
        assert "get_model_pool" in source
        assert "loaded.adapter.generate_embeddings" in source


class TestGenerateEmbeddingsService:
    """Tests for generate_embeddings service with mocked adapter."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_basic(self):
        """Generate embeddings returns result from adapter."""
        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        expected_result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dimensions=3,
            total_tokens=7,
            finish_reason="stop",
        )

        mock_adapter = MagicMock()
        mock_adapter.generate_embeddings = AsyncMock(return_value=expected_result)

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            result = await generate_embeddings(
                model_id="test-embed-model",
                texts=["Hello", "World"],
            )

            assert result.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            assert result.total_tokens == 7
            assert result.dimensions == 3
            assert result.finish_reason == "stop"
            mock_pool.get_model.assert_called_once_with("test-embed-model")
            mock_adapter.generate_embeddings.assert_called_once_with(
                mock_loaded.model, ["Hello", "World"]
            )

    @pytest.mark.asyncio
    async def test_generate_embeddings_single_text(self):
        """Generate embeddings for a single text input."""
        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        expected_result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            dimensions=4,
            total_tokens=5,
            finish_reason="stop",
        )

        mock_adapter = MagicMock()
        mock_adapter.generate_embeddings = AsyncMock(return_value=expected_result)

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            result = await generate_embeddings(
                model_id="test-embed",
                texts=["Hello world test"],
            )

            assert len(result.embeddings) == 1
            assert result.total_tokens == 5
            assert result.dimensions == 4
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_embeddings_passes_model_and_texts(self):
        """Verify adapter is called with the loaded model and texts."""
        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        expected_result = EmbeddingResult(
            embeddings=[[1.0, 2.0]],
            dimensions=2,
            total_tokens=2,
            finish_reason="stop",
        )

        mock_adapter = MagicMock()
        mock_adapter.generate_embeddings = AsyncMock(return_value=expected_result)

        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            await generate_embeddings(
                model_id="test-embed",
                texts=["Test"],
            )

            # Verify adapter was called with the correct model and texts
            mock_adapter.generate_embeddings.assert_called_once_with(mock_model, ["Test"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_error_propagates(self):
        """Errors from adapter propagate through the service."""
        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        mock_adapter = MagicMock()
        mock_adapter.generate_embeddings = AsyncMock(
            side_effect=RuntimeError("Embeddings generation failed: Metal error")
        )

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with pytest.raises(RuntimeError, match="Embeddings generation failed"):
                await generate_embeddings(
                    model_id="test-embed",
                    texts=["Hello"],
                )

    @pytest.mark.asyncio
    async def test_generate_embeddings_no_adapter_raises(self):
        """AssertionError when loaded model has no adapter."""
        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = None

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            with pytest.raises(AssertionError, match="No adapter"):
                await generate_embeddings(
                    model_id="test-embed",
                    texts=["Hello"],
                )

    @pytest.mark.asyncio
    async def test_endpoint_timeout_returns_408(self):
        """Timeout during embeddings generation raises TimeoutHTTPException."""
        from mlx_manager.mlx_server.errors import TimeoutHTTPException

        request = EmbeddingRequest(
            input="Hello",
            model="mlx-community/all-MiniLM-L6-v2-4bit",
        )

        with patch("mlx_manager.mlx_server.api.v1.embeddings.detect_model_type") as mock_detect:
            from mlx_manager.mlx_server.models.types import ModelType

            mock_detect.return_value = ModelType.EMBEDDINGS

            # Mock generate_embeddings to return an AsyncMock that raises TimeoutError
            async def timeout_coro(*args, **kwargs):
                raise TimeoutError("timed out")

            with patch(
                "mlx_manager.mlx_server.api.v1.embeddings.generate_embeddings",
                new=AsyncMock(side_effect=timeout_coro),
            ):
                with pytest.raises(TimeoutHTTPException):
                    await create_embeddings(request)

    @pytest.mark.asyncio
    async def test_endpoint_generic_exception_returns_500(self):
        """Generic exception during embeddings generation returns 500."""
        from fastapi import HTTPException

        request = EmbeddingRequest(
            input="Hello",
            model="mlx-community/all-MiniLM-L6-v2-4bit",
        )

        with patch("mlx_manager.mlx_server.api.v1.embeddings.detect_model_type") as mock_detect:
            from mlx_manager.mlx_server.models.types import ModelType

            mock_detect.return_value = ModelType.EMBEDDINGS

            # Mock generate_embeddings to raise RuntimeError
            async def error_coro(*args, **kwargs):
                raise RuntimeError("GPU error")

            with patch(
                "mlx_manager.mlx_server.api.v1.embeddings.generate_embeddings",
                new=AsyncMock(side_effect=error_coro),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await create_embeddings(request)

                assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_generate_embeddings_multiple_texts(self):
        """Multiple texts are passed through to adapter correctly."""
        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        expected_result = EmbeddingResult(
            embeddings=[[0.1], [0.2], [0.3]],
            dimensions=1,
            total_tokens=9,
            finish_reason="stop",
        )

        mock_adapter = MagicMock()
        mock_adapter.generate_embeddings = AsyncMock(return_value=expected_result)

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.adapter = mock_adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            result = await generate_embeddings(
                model_id="test-embed",
                texts=["text1", "text2", "text3"],
            )

            assert result.total_tokens == 9
            assert len(result.embeddings) == 3
            mock_adapter.generate_embeddings.assert_called_once_with(
                mock_loaded.model, ["text1", "text2", "text3"]
            )
