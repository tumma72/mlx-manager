"""Tests for OpenAI cloud backend client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mlx_manager.mlx_server.services.cloud.openai import (
    OPENAI_API_URL,
    OpenAICloudBackend,
    create_openai_backend,
)


class TestOpenAICloudBackendInitialization:
    """Tests for OpenAI backend initialization."""

    def test_default_base_url_is_openai_api(self) -> None:
        """Default base_url is OPENAI_API_URL."""
        backend = OpenAICloudBackend(api_key="test-key")
        assert backend.base_url == OPENAI_API_URL
        assert backend.base_url == "https://api.openai.com"

    def test_custom_base_url_accepted(self) -> None:
        """Custom base_url is accepted."""
        backend = OpenAICloudBackend(
            api_key="test-key",
            base_url="https://api.azure.openai.com/v1",
        )
        assert backend.base_url == "https://api.azure.openai.com/v1"

    def test_api_key_stored(self) -> None:
        """API key is stored privately."""
        backend = OpenAICloudBackend(api_key="sk-test-key-123")
        assert backend._api_key == "sk-test-key-123"

    def test_circuit_breaker_starts_closed(self) -> None:
        """Circuit breaker starts in closed state."""
        backend = OpenAICloudBackend(api_key="test-key")
        assert backend.is_circuit_open is False


class TestBuildHeaders:
    """Tests for _build_headers method."""

    def test_headers_include_bearer_authorization(self) -> None:
        """Headers include Authorization with Bearer token."""
        backend = OpenAICloudBackend(api_key="sk-test-key")
        headers = backend._build_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test-key"

    def test_headers_include_content_type(self) -> None:
        """Headers include Content-Type application/json."""
        backend = OpenAICloudBackend(api_key="test-key")
        headers = backend._build_headers()

        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    def test_headers_applied_to_client(self) -> None:
        """Headers are applied to the underlying HTTP client."""
        backend = OpenAICloudBackend(api_key="sk-my-api-key")

        assert backend._client.headers["Authorization"] == "Bearer sk-my-api-key"
        assert backend._client.headers["Content-Type"] == "application/json"


class TestChatCompletionNonStreaming:
    """Tests for non-streaming chat completion."""

    @pytest.fixture
    def backend(self) -> OpenAICloudBackend:
        """Create a test backend."""
        return OpenAICloudBackend(api_key="test-key")

    async def test_non_streaming_returns_dict(self, backend: OpenAICloudBackend) -> None:
        """Non-streaming chat_completion returns a dict."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"content": "Hello!"}}],
        }

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await backend.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4",
                max_tokens=100,
            )

            assert isinstance(result, dict)
            assert result["id"] == "chatcmpl-123"
            assert result["choices"][0]["message"]["content"] == "Hello!"

    async def test_non_streaming_passes_request_data(
        self, backend: OpenAICloudBackend
    ) -> None:
        """Non-streaming passes correct request data to API."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": []}

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await backend.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo",
                max_tokens=200,
                temperature=0.7,
            )

            # Verify endpoint
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/v1/chat/completions"

            # Verify request body
            json_data = call_args[1]["json"]
            assert json_data["model"] == "gpt-3.5-turbo"
            assert json_data["messages"] == [{"role": "user", "content": "Hello"}]
            assert json_data["max_tokens"] == 200
            assert json_data["temperature"] == 0.7
            assert json_data["stream"] is False

    async def test_non_streaming_passes_additional_kwargs(
        self, backend: OpenAICloudBackend
    ) -> None:
        """Non-streaming passes additional kwargs to request."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": []}

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await backend.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4",
                max_tokens=100,
                top_p=0.9,
                presence_penalty=0.5,
            )

            json_data = mock_post.call_args[1]["json"]
            assert json_data["top_p"] == 0.9
            assert json_data["presence_penalty"] == 0.5

    async def test_non_streaming_propagates_http_error(
        self, backend: OpenAICloudBackend
    ) -> None:
        """Non-streaming propagates HTTP errors."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        http_error = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )
        mock_response.raise_for_status.side_effect = http_error

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(httpx.HTTPStatusError):
                await backend.chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="gpt-4",
                    max_tokens=100,
                )


class TestChatCompletionStreaming:
    """Tests for streaming chat completion."""

    @pytest.fixture
    def backend(self) -> OpenAICloudBackend:
        """Create a test backend."""
        return OpenAICloudBackend(api_key="test-key")

    async def test_streaming_returns_async_generator(
        self, backend: OpenAICloudBackend
    ) -> None:
        """Streaming chat_completion returns an async generator."""
        # Mock the stream context manager
        async def mock_aiter_lines():
            yield 'data: {"id": "chunk-1", "choices": [{"delta": {"content": "Hi"}}]}'
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm):
            result = await backend.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4",
                max_tokens=100,
                stream=True,
            )

            # Should be an async generator
            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0]["id"] == "chunk-1"

    async def test_streaming_passes_stream_true(
        self, backend: OpenAICloudBackend
    ) -> None:
        """Streaming sets stream=True in request data."""
        async def mock_aiter_lines():
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm) as mock_stream:
            result = await backend.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4",
                max_tokens=100,
                stream=True,
            )

            # Consume the generator
            async for _ in result:
                pass

            # Verify stream=True in request
            json_data = mock_stream.call_args[1]["json"]
            assert json_data["stream"] is True


class TestStreamChatCompletionSSEParsing:
    """Tests for SSE parsing in _stream_chat_completion."""

    @pytest.fixture
    def backend(self) -> OpenAICloudBackend:
        """Create a test backend."""
        return OpenAICloudBackend(api_key="test-key")

    async def test_parses_sse_data_lines(self, backend: OpenAICloudBackend) -> None:
        """Parses data: prefixed SSE lines."""
        chunk1 = {"id": "1", "choices": [{"delta": {"content": "Hello"}}]}
        chunk2 = {"id": "2", "choices": [{"delta": {"content": " world"}}]}

        async def mock_aiter_lines():
            yield f"data: {json.dumps(chunk1)}"
            yield f"data: {json.dumps(chunk2)}"
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm):
            result = await backend.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4",
                max_tokens=100,
                stream=True,
            )

            chunks = [chunk async for chunk in result]

            assert len(chunks) == 2
            assert chunks[0] == chunk1
            assert chunks[1] == chunk2

    async def test_skips_empty_lines(self, backend: OpenAICloudBackend) -> None:
        """Skips empty lines in SSE stream."""
        chunk = {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}

        async def mock_aiter_lines():
            yield ""  # Empty line
            yield f"data: {json.dumps(chunk)}"
            yield "   "  # Whitespace only
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm):
            result = await backend.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4",
                max_tokens=100,
                stream=True,
            )

            chunks = [chunk async for chunk in result]

            # Only 1 chunk, empty lines skipped
            assert len(chunks) == 1
            assert chunks[0] == chunk

    async def test_handles_done_marker(self, backend: OpenAICloudBackend) -> None:
        """Stops iteration at [DONE] marker."""
        chunk = {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}

        async def mock_aiter_lines():
            yield f"data: {json.dumps(chunk)}"
            yield "data: [DONE]"
            yield 'data: {"should": "not appear"}'  # After DONE

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm):
            result = await backend.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4",
                max_tokens=100,
                stream=True,
            )

            chunks = [chunk async for chunk in result]

            # Only 1 chunk before [DONE]
            assert len(chunks) == 1

    async def test_handles_malformed_json_gracefully(
        self, backend: OpenAICloudBackend
    ) -> None:
        """Skips malformed JSON without crashing."""
        valid_chunk = {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}

        async def mock_aiter_lines():
            yield "data: {not valid json"
            yield f"data: {json.dumps(valid_chunk)}"
            yield "data: also invalid"
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm):
            result = await backend.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4",
                max_tokens=100,
                stream=True,
            )

            chunks = [chunk async for chunk in result]

            # Only valid chunk captured
            assert len(chunks) == 1
            assert chunks[0] == valid_chunk

    async def test_ignores_non_data_lines(self, backend: OpenAICloudBackend) -> None:
        """Ignores lines that don't start with 'data:'."""
        chunk = {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}

        async def mock_aiter_lines():
            yield ": comment line"  # SSE comment
            yield "event: message"  # SSE event type
            yield f"data: {json.dumps(chunk)}"
            yield "id: 123"  # SSE id
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm):
            result = await backend.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4",
                max_tokens=100,
                stream=True,
            )

            chunks = [chunk async for chunk in result]

            # Only 1 data chunk
            assert len(chunks) == 1


class TestCreateOpenaiBackend:
    """Tests for factory function."""

    def test_creates_backend_with_defaults(self) -> None:
        """Factory creates backend with default base_url."""
        backend = create_openai_backend(api_key="sk-test")

        assert isinstance(backend, OpenAICloudBackend)
        assert backend.base_url == OPENAI_API_URL
        assert backend._api_key == "sk-test"

    def test_creates_backend_with_custom_base_url(self) -> None:
        """Factory creates backend with custom base_url."""
        backend = create_openai_backend(
            api_key="sk-test",
            base_url="https://api.azure.openai.com",
        )

        assert backend.base_url == "https://api.azure.openai.com"

    def test_passes_kwargs_to_backend(self) -> None:
        """Factory passes kwargs to backend constructor."""
        backend = create_openai_backend(
            api_key="sk-test",
            timeout=120.0,
            circuit_breaker_fail_max=10,
        )

        assert backend._client.timeout.read == 120.0
        assert backend._circuit_breaker.fail_max == 10

    def test_none_base_url_uses_default(self) -> None:
        """Factory uses default when base_url is None."""
        backend = create_openai_backend(api_key="sk-test", base_url=None)

        assert backend.base_url == OPENAI_API_URL
