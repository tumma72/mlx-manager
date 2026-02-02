"""Tests for embeddings endpoint and service."""

from unittest.mock import patch

import pytest

from mlx_manager.mlx_server.api.v1.embeddings import create_embeddings
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
                mock_gen.return_value = (
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # embeddings
                    10,  # total_tokens
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

    def test_service_imports_pool(self):
        """Verify service can import model pool."""
        import inspect

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        source = inspect.getsource(generate_embeddings)
        assert "get_model_pool" in source
        assert "text_embeds" in source
