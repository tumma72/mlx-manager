"""Tests for AnthropicCloudBackend."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mlx_manager.mlx_server.services.cloud.anthropic import (
    ANTHROPIC_API_URL,
    ANTHROPIC_VERSION,
    AnthropicCloudBackend,
    create_anthropic_backend,
)


class TestAnthropicCloudBackendInitialization:
    """Tests for client initialization."""

    def test_client_created_with_default_url(self) -> None:
        """Client uses default Anthropic API URL."""
        client = AnthropicCloudBackend(api_key="test-key")
        assert client.base_url == ANTHROPIC_API_URL

    def test_client_accepts_custom_base_url(self) -> None:
        """Client accepts custom base_url."""
        custom_url = "https://custom.anthropic.com"
        client = AnthropicCloudBackend(api_key="test-key", base_url=custom_url)
        assert client.base_url == custom_url

    def test_headers_include_api_key(self) -> None:
        """Headers include x-api-key."""
        client = AnthropicCloudBackend(api_key="test-key-123")
        headers = client._build_headers()
        assert headers["x-api-key"] == "test-key-123"

    def test_headers_include_anthropic_version(self) -> None:
        """Headers include anthropic-version."""
        client = AnthropicCloudBackend(api_key="test-key")
        headers = client._build_headers()
        assert headers["anthropic-version"] == ANTHROPIC_VERSION

    def test_headers_include_content_type(self) -> None:
        """Headers include Content-Type."""
        client = AnthropicCloudBackend(api_key="test-key")
        headers = client._build_headers()
        assert headers["Content-Type"] == "application/json"

    def test_custom_anthropic_version(self) -> None:
        """Client accepts custom anthropic_version."""
        client = AnthropicCloudBackend(
            api_key="test-key", anthropic_version="2024-01-01"
        )
        headers = client._build_headers()
        assert headers["anthropic-version"] == "2024-01-01"


class TestTranslateRequest:
    """Tests for _translate_request method."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    def test_extracts_system_message_to_separate_field(
        self, client: AnthropicCloudBackend
    ) -> None:
        """System message extracted to separate 'system' field."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = client._translate_request(
            messages, "claude-3-opus", max_tokens=100, temperature=0.7, stream=False
        )

        assert result["system"] == "You are helpful"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_preserves_user_assistant_messages(
        self, client: AnthropicCloudBackend
    ) -> None:
        """User and assistant messages preserved in messages array."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        result = client._translate_request(
            messages, "claude-3-opus", max_tokens=100, temperature=0.7, stream=False
        )

        assert "system" not in result
        assert len(result["messages"]) == 3
        assert result["messages"][0] == {"role": "user", "content": "Hello"}
        assert result["messages"][1] == {"role": "assistant", "content": "Hi there"}
        assert result["messages"][2] == {"role": "user", "content": "How are you?"}

    def test_maps_temperature_and_max_tokens(
        self, client: AnthropicCloudBackend
    ) -> None:
        """Temperature and max_tokens mapped correctly."""
        messages = [{"role": "user", "content": "Hello"}]
        result = client._translate_request(
            messages, "claude-3-opus", max_tokens=256, temperature=0.5, stream=False
        )

        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 256

    def test_maps_model_and_stream(self, client: AnthropicCloudBackend) -> None:
        """Model and stream parameters mapped correctly."""
        messages = [{"role": "user", "content": "Hello"}]
        result = client._translate_request(
            messages, "claude-3-opus", max_tokens=100, temperature=0.7, stream=True
        )

        assert result["model"] == "claude-3-opus"
        assert result["stream"] is True

    def test_translates_stop_string_to_stop_sequences(
        self, client: AnthropicCloudBackend
    ) -> None:
        """Single stop string translated to stop_sequences list."""
        messages = [{"role": "user", "content": "Hello"}]
        result = client._translate_request(
            messages,
            "claude-3-opus",
            max_tokens=100,
            temperature=0.7,
            stream=False,
            stop="END",
        )

        assert result["stop_sequences"] == ["END"]

    def test_translates_stop_list_to_stop_sequences(
        self, client: AnthropicCloudBackend
    ) -> None:
        """Stop list translated to stop_sequences."""
        messages = [{"role": "user", "content": "Hello"}]
        result = client._translate_request(
            messages,
            "claude-3-opus",
            max_tokens=100,
            temperature=0.7,
            stream=False,
            stop=["END", "STOP"],
        )

        assert result["stop_sequences"] == ["END", "STOP"]

    def test_no_stop_sequences_when_stop_not_provided(
        self, client: AnthropicCloudBackend
    ) -> None:
        """No stop_sequences when stop not provided."""
        messages = [{"role": "user", "content": "Hello"}]
        result = client._translate_request(
            messages, "claude-3-opus", max_tokens=100, temperature=0.7, stream=False
        )

        assert "stop_sequences" not in result


class TestTranslateResponse:
    """Tests for _translate_response method."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    def test_concatenates_content_blocks_to_string(
        self, client: AnthropicCloudBackend
    ) -> None:
        """Content blocks concatenated to single string."""
        anthropic_response = {
            "id": "msg_123",
            "model": "claude-3-opus",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._translate_response(anthropic_response)

        assert result["choices"][0]["message"]["content"] == "Hello World"

    def test_translates_stop_reason_end_turn(
        self, client: AnthropicCloudBackend
    ) -> None:
        """end_turn translated to stop."""
        anthropic_response = {
            "id": "msg_123",
            "model": "claude-3-opus",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._translate_response(anthropic_response)

        assert result["choices"][0]["finish_reason"] == "stop"

    def test_translates_stop_reason_max_tokens(
        self, client: AnthropicCloudBackend
    ) -> None:
        """max_tokens translated to length."""
        anthropic_response = {
            "id": "msg_123",
            "model": "claude-3-opus",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 100},
        }
        result = client._translate_response(anthropic_response)

        assert result["choices"][0]["finish_reason"] == "length"

    def test_translates_stop_reason_stop_sequence(
        self, client: AnthropicCloudBackend
    ) -> None:
        """stop_sequence translated to stop."""
        anthropic_response = {
            "id": "msg_123",
            "model": "claude-3-opus",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "stop_sequence",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._translate_response(anthropic_response)

        assert result["choices"][0]["finish_reason"] == "stop"

    def test_maps_usage_correctly(self, client: AnthropicCloudBackend) -> None:
        """Usage mapped from input/output to prompt/completion tokens."""
        anthropic_response = {
            "id": "msg_123",
            "model": "claude-3-opus",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        result = client._translate_response(anthropic_response)

        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["completion_tokens"] == 50
        assert result["usage"]["total_tokens"] == 150

    def test_changes_id_prefix(self, client: AnthropicCloudBackend) -> None:
        """ID prefix changed from msg_ to chatcmpl-."""
        anthropic_response = {
            "id": "msg_abc123",
            "model": "claude-3-opus",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._translate_response(anthropic_response)

        assert result["id"] == "chatcmpl-abc123"

    def test_sets_object_type(self, client: AnthropicCloudBackend) -> None:
        """Object type set to chat.completion."""
        anthropic_response = {
            "id": "msg_123",
            "model": "claude-3-opus",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._translate_response(anthropic_response)

        assert result["object"] == "chat.completion"

    def test_preserves_model(self, client: AnthropicCloudBackend) -> None:
        """Model name preserved."""
        anthropic_response = {
            "id": "msg_123",
            "model": "claude-3-opus-20240229",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._translate_response(anthropic_response)

        assert result["model"] == "claude-3-opus-20240229"

    def test_ignores_non_text_content_blocks(
        self, client: AnthropicCloudBackend
    ) -> None:
        """Non-text content blocks ignored."""
        anthropic_response = {
            "id": "msg_123",
            "model": "claude-3-opus",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "tool_1"},
                {"type": "text", "text": "World"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._translate_response(anthropic_response)

        assert result["choices"][0]["message"]["content"] == "Hello World"


class TestStreamWithTranslation:
    """Tests for _stream_with_translation method."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    async def test_content_block_delta_yields_openai_chunk(
        self, client: AnthropicCloudBackend
    ) -> None:
        """content_block_delta events yield OpenAI format chunks."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'

        with patch.object(
            client, "_stream_with_circuit_breaker", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in client._stream_with_translation(
                {"model": "claude-3-opus"}
            ):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0]["object"] == "chat.completion.chunk"
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
            assert chunks[0]["choices"][0]["finish_reason"] is None

    async def test_message_delta_yields_finish_reason(
        self, client: AnthropicCloudBackend
    ) -> None:
        """message_delta with stop_reason yields finish chunk."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield 'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}'

        with patch.object(
            client, "_stream_with_circuit_breaker", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in client._stream_with_translation(
                {"model": "claude-3-opus"}
            ):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0]["choices"][0]["finish_reason"] == "stop"
            assert chunks[0]["choices"][0]["delta"] == {}

    async def test_message_stop_ends_stream(
        self, client: AnthropicCloudBackend
    ) -> None:
        """message_stop ends stream."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'
            yield 'data: {"type": "message_stop"}'
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Should not appear"}}'

        with patch.object(
            client, "_stream_with_circuit_breaker", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in client._stream_with_translation(
                {"model": "claude-3-opus"}
            ):
                chunks.append(chunk)

            # Only one chunk before message_stop
            assert len(chunks) == 1
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"

    async def test_empty_lines_ignored(self, client: AnthropicCloudBackend) -> None:
        """Empty lines are ignored."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield ""
            yield "   "
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'

        with patch.object(
            client, "_stream_with_circuit_breaker", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in client._stream_with_translation(
                {"model": "claude-3-opus"}
            ):
                chunks.append(chunk)

            assert len(chunks) == 1

    async def test_malformed_json_ignored(self, client: AnthropicCloudBackend) -> None:
        """Malformed JSON is ignored."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "data: not valid json"
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'

        with patch.object(
            client, "_stream_with_circuit_breaker", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in client._stream_with_translation(
                {"model": "claude-3-opus"}
            ):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"

    async def test_event_lines_ignored(self, client: AnthropicCloudBackend) -> None:
        """Lines starting with 'event:' are ignored."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "event: message_start"
            yield "event: content_block_start"
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'

        with patch.object(
            client, "_stream_with_circuit_breaker", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in client._stream_with_translation(
                {"model": "claude-3-opus"}
            ):
                chunks.append(chunk)

            assert len(chunks) == 1

    async def test_model_included_in_chunks(
        self, client: AnthropicCloudBackend
    ) -> None:
        """Model from request included in chunks."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield 'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}'

        with patch.object(
            client, "_stream_with_circuit_breaker", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in client._stream_with_translation(
                {"model": "claude-3-opus-20240229"}
            ):
                chunks.append(chunk)

            assert chunks[0]["model"] == "claude-3-opus-20240229"


class TestChatCompletionIntegration:
    """Tests for chat_completion method integration."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    async def test_non_streaming_returns_translated_dict(
        self, client: AnthropicCloudBackend
    ) -> None:
        """Non-streaming chat_completion returns translated dict."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "msg_123",
            "model": "claude-3-opus",
            "content": [{"type": "text", "text": "Hello, how can I help?"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            result = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-opus",
                max_tokens=100,
                stream=False,
            )

            assert isinstance(result, dict)
            assert result["object"] == "chat.completion"
            assert result["choices"][0]["message"]["content"] == "Hello, how can I help?"

    async def test_streaming_returns_generator(
        self, client: AnthropicCloudBackend
    ) -> None:
        """Streaming chat_completion returns async generator."""
        result = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-opus",
            max_tokens=100,
            stream=True,
        )

        # Should return an async generator
        assert hasattr(result, "__anext__")


class TestFactoryFunction:
    """Tests for create_anthropic_backend factory function."""

    def test_creates_backend_with_api_key(self) -> None:
        """Factory creates backend with API key."""
        backend = create_anthropic_backend(api_key="test-key")
        assert isinstance(backend, AnthropicCloudBackend)
        assert backend._api_key == "test-key"

    def test_creates_backend_with_default_url(self) -> None:
        """Factory creates backend with default URL."""
        backend = create_anthropic_backend(api_key="test-key")
        assert backend.base_url == ANTHROPIC_API_URL

    def test_creates_backend_with_custom_url(self) -> None:
        """Factory creates backend with custom URL."""
        backend = create_anthropic_backend(
            api_key="test-key", base_url="https://custom.api.com"
        )
        assert backend.base_url == "https://custom.api.com"

    def test_passes_kwargs_to_client(self) -> None:
        """Factory passes kwargs to client."""
        backend = create_anthropic_backend(
            api_key="test-key",
            timeout=120.0,
            circuit_breaker_fail_max=10,
        )
        assert backend._client.timeout.read == 120.0
        assert backend._circuit_breaker.fail_max == 10
