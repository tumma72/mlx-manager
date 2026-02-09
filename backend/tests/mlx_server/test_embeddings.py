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


class TestGenerateEmbeddingsService:
    """Tests for generate_embeddings service with mocked Metal thread."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_basic(self):
        """Generate embeddings returns vectors and token count."""
        from unittest.mock import AsyncMock, MagicMock

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        # Mock tokenizer with inner _tokenizer for batch encoding
        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        mock_tokenizer.encode.side_effect = [
            [1, 2, 3],  # 3 tokens for "Hello"
            [4, 5, 6, 7],  # 4 tokens for "World"
        ]

        # Mock model output
        mock_output = MagicMock()
        mock_text_embeds = MagicMock()
        mock_text_embeds.tolist.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_output.text_embeds = mock_text_embeds

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.tokenizer = mock_tokenizer

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.array", side_effect=lambda v: v),
            patch("mlx.core.eval"),
        ):
            embeddings_list, total_tokens = await generate_embeddings(
                model_id="test-embed-model",
                texts=["Hello", "World"],
            )

            assert embeddings_list == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            assert total_tokens == 7  # 3 + 4
            mock_pool.get_model.assert_called_once_with("test-embed-model")

    @pytest.mark.asyncio
    async def test_generate_embeddings_single_text(self):
        """Generate embeddings for a single text input."""
        from unittest.mock import AsyncMock, MagicMock

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "attention_mask": [[1, 1, 1, 1, 1]],
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        mock_output = MagicMock()
        mock_text_embeds = MagicMock()
        mock_text_embeds.tolist.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_output.text_embeds = mock_text_embeds

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.tokenizer = mock_tokenizer

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.array", side_effect=lambda v: v),
            patch("mlx.core.eval"),
        ):
            embeddings_list, total_tokens = await generate_embeddings(
                model_id="test-embed",
                texts=["Hello world test"],
            )

            assert len(embeddings_list) == 1
            assert total_tokens == 5

    @pytest.mark.asyncio
    async def test_generate_embeddings_tokenizer_without_inner(self):
        """Tokenizer without _tokenizer attr uses itself for batch encoding."""
        from unittest.mock import AsyncMock, MagicMock

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        # Tokenizer without _tokenizer attribute; getattr fallback returns self
        mock_tokenizer = MagicMock(spec=["encode", "__call__"])
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
        }
        mock_tokenizer.encode.return_value = [1, 2]

        mock_output = MagicMock()
        mock_text_embeds = MagicMock()
        mock_text_embeds.tolist.return_value = [[0.5, 0.6]]
        mock_output.text_embeds = mock_text_embeds

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.tokenizer = mock_tokenizer

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.array", side_effect=lambda v: v),
            patch("mlx.core.eval"),
        ):
            embeddings_list, total_tokens = await generate_embeddings(
                model_id="test-embed",
                texts=["Hi"],
            )

            assert embeddings_list == [[0.5, 0.6]]
            assert total_tokens == 2
            # Since no _tokenizer, getattr falls back to tokenizer itself
            mock_tokenizer.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_model_forward_pass(self):
        """Verify model is called with input_ids and attention_mask from tokenizer."""
        from unittest.mock import AsyncMock, MagicMock

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": [[10, 20]],
            "attention_mask": [[1, 1]],
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        mock_tokenizer.encode.return_value = [10, 20]

        mock_output = MagicMock()
        mock_text_embeds = MagicMock()
        mock_text_embeds.tolist.return_value = [[1.0, 2.0]]
        mock_output.text_embeds = mock_text_embeds

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.tokenizer = mock_tokenizer

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.array", side_effect=lambda v: v),
            patch("mlx.core.eval"),
        ):
            await generate_embeddings(
                model_id="test-embed",
                texts=["Test"],
            )

            # Verify model was called with input_ids and attention_mask
            mock_model.assert_called_once()
            call_args = mock_model.call_args
            assert call_args[0][0] == [[10, 20]]  # input_ids (batched)
            assert call_args[1]["attention_mask"] == [[1, 1]]  # attention_mask

    @pytest.mark.asyncio
    async def test_generate_embeddings_tokenizer_called_correctly(self):
        """Verify inner tokenizer is called with correct batch params."""
        from unittest.mock import AsyncMock, MagicMock

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": [[1], [2]],
            "attention_mask": [[1], [1]],
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        mock_tokenizer.encode.return_value = [1]

        mock_output = MagicMock()
        mock_text_embeds = MagicMock()
        mock_text_embeds.tolist.return_value = [[0.1], [0.2]]
        mock_output.text_embeds = mock_text_embeds

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.tokenizer = mock_tokenizer

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.array", side_effect=lambda v: v),
            patch("mlx.core.eval"),
        ):
            await generate_embeddings(
                model_id="test-embed",
                texts=["A", "B"],
            )

            mock_inner_tokenizer.assert_called_once_with(
                ["A", "B"],
                return_tensors=None,
                padding=True,
                truncation=True,
                max_length=512,
            )

    @pytest.mark.asyncio
    async def test_generate_embeddings_error_propagates(self):
        """Errors from run_on_metal_thread propagate."""
        from unittest.mock import AsyncMock, MagicMock

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        mock_loaded = MagicMock()
        mock_loaded.model = MagicMock()
        mock_loaded.tokenizer = MagicMock()

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def raise_error(fn, **kwargs):
            raise RuntimeError("Embeddings generation failed: Metal error")

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=raise_error,
            ),
        ):
            with pytest.raises(RuntimeError, match="Embeddings generation failed"):
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
            from unittest.mock import AsyncMock

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
        from unittest.mock import AsyncMock

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
    async def test_generate_embeddings_multiple_texts_token_count(self):
        """Total tokens is sum of individual text tokens, not padded batch."""
        from unittest.mock import AsyncMock, MagicMock

        from mlx_manager.mlx_server.services.embeddings import generate_embeddings

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]],
            "attention_mask": [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_inner_tokenizer
        mock_tokenizer.encode.side_effect = [
            [1, 2, 3],  # 3 tokens
            [4, 5],  # 2 tokens
            [6, 7, 8, 9],  # 4 tokens
        ]

        mock_output = MagicMock()
        mock_text_embeds = MagicMock()
        mock_text_embeds.tolist.return_value = [[0.1], [0.2], [0.3]]
        mock_output.text_embeds = mock_text_embeds

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        mock_loaded = MagicMock()
        mock_loaded.model = mock_model
        mock_loaded.tokenizer = mock_tokenizer

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=mock_loaded)

        async def run_fn_directly(fn, **kwargs):
            return fn()

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=run_fn_directly,
            ),
            patch("mlx.core.array", side_effect=lambda v: v),
            patch("mlx.core.eval"),
        ):
            _, total_tokens = await generate_embeddings(
                model_id="test-embed",
                texts=["text1", "text2", "text3"],
            )

            assert total_tokens == 9  # 3 + 2 + 4
