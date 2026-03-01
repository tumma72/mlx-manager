"""Tests for AnthropicCloudBackend."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mlx_manager.mlx_server.models.ir import (
    InferenceResult,
    InternalRequest,
    StreamEvent,
    TextResult,
)
from mlx_manager.mlx_server.services.cloud.anthropic import (
    ANTHROPIC_API_URL,
    ANTHROPIC_VERSION,
    AnthropicCloudBackend,
    create_anthropic_backend,
)
from mlx_manager.models.enums import ApiType
from mlx_manager.models.value_objects import InferenceParams


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
        client = AnthropicCloudBackend(api_key="test-key", anthropic_version="2024-01-01")
        headers = client._build_headers()
        assert headers["anthropic-version"] == "2024-01-01"


class TestTranslateRequest:
    """Tests for _translate_request method."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    def test_extracts_system_message_to_separate_field(self, client: AnthropicCloudBackend) -> None:
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

    def test_preserves_user_assistant_messages(self, client: AnthropicCloudBackend) -> None:
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

    def test_maps_temperature_and_max_tokens(self, client: AnthropicCloudBackend) -> None:
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

    def test_translates_stop_string_to_stop_sequences(self, client: AnthropicCloudBackend) -> None:
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

    def test_translates_stop_list_to_stop_sequences(self, client: AnthropicCloudBackend) -> None:
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

    def test_no_stop_sequences_when_stop_not_provided(self, client: AnthropicCloudBackend) -> None:
        """No stop_sequences when stop not provided."""
        messages = [{"role": "user", "content": "Hello"}]
        result = client._translate_request(
            messages, "claude-3-opus", max_tokens=100, temperature=0.7, stream=False
        )

        assert "stop_sequences" not in result


def _make_ir(
    *,
    model: str = "claude-3-opus",
    messages: list[dict[str, Any]] | None = None,
    stream: bool = False,
    original_protocol: ApiType | None = ApiType.ANTHROPIC,
    original_request: Any = None,
    max_tokens: int = 100,
    temperature: float = 1.0,
) -> InternalRequest:
    """Helper to create InternalRequest for tests."""
    return InternalRequest(
        model=model,
        messages=messages or [{"role": "user", "content": "Hi"}],
        params=InferenceParams(max_tokens=max_tokens, temperature=temperature),
        stream=stream,
        original_protocol=original_protocol,
        original_request=original_request,
    )


class TestParseResponseToIR:
    """Tests for _parse_response_to_ir method."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    def test_parses_text_content_blocks(self, client: AnthropicCloudBackend) -> None:
        """Concatenates text content blocks."""
        anthropic_response = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._parse_response_to_ir(anthropic_response)

        assert isinstance(result, InferenceResult)
        assert result.result.content == "Hello World"

    def test_translates_stop_reason_end_turn(self, client: AnthropicCloudBackend) -> None:
        """end_turn translated to stop."""
        anthropic_response = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._parse_response_to_ir(anthropic_response)

        assert result.result.finish_reason == "stop"

    def test_translates_stop_reason_max_tokens(self, client: AnthropicCloudBackend) -> None:
        """max_tokens translated to length."""
        anthropic_response = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 100},
        }
        result = client._parse_response_to_ir(anthropic_response)

        assert result.result.finish_reason == "length"

    def test_translates_stop_reason_stop_sequence(self, client: AnthropicCloudBackend) -> None:
        """stop_sequence translated to stop."""
        anthropic_response = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "stop_sequence",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._parse_response_to_ir(anthropic_response)

        assert result.result.finish_reason == "stop"

    def test_maps_usage_correctly(self, client: AnthropicCloudBackend) -> None:
        """Usage mapped from input/output to prompt/completion tokens."""
        anthropic_response = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        result = client._parse_response_to_ir(anthropic_response)

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50

    def test_ignores_non_text_content_blocks(self, client: AnthropicCloudBackend) -> None:
        """Non-text content blocks ignored."""
        anthropic_response = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "tool_1"},
                {"type": "text", "text": "World"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._parse_response_to_ir(anthropic_response)

        assert result.result.content == "Hello World"

    def test_handles_missing_usage(self, client: AnthropicCloudBackend) -> None:
        """Handles response without usage field."""
        anthropic_response = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
        }
        result = client._parse_response_to_ir(anthropic_response)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


class TestStreamAnthropicNative:
    """Tests for _stream_anthropic_native method."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    async def test_yields_event_dicts_for_sse(self, client: AnthropicCloudBackend) -> None:
        """Yields dicts with event and data keys for EventSourceResponse."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "event: content_block_delta"
            yield 'data: {"type": "content_block_delta", "delta": {"text": "Hello"}}'

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_anthropic_native({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 1
            assert events[0]["event"] == "content_block_delta"
            assert "data" in events[0]

    async def test_pairs_event_type_with_data(self, client: AnthropicCloudBackend) -> None:
        """Pairs event: type with subsequent data: line."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "event: message_start"
            yield 'data: {"type": "message_start", "message": {}}'
            yield "event: content_block_delta"
            yield 'data: {"type": "content_block_delta", "delta": {"text": "Hi"}}'

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_anthropic_native({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 2
            assert events[0]["event"] == "message_start"
            assert events[1]["event"] == "content_block_delta"

    async def test_skips_empty_lines(self, client: AnthropicCloudBackend) -> None:
        """Skips empty lines."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield ""
            yield "   "
            yield "event: content_block_delta"
            yield 'data: {"type": "content_block_delta", "delta": {"text": "Hi"}}'

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_anthropic_native({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 1


class TestStreamToIR:
    """Tests for _stream_to_ir method."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    async def test_content_block_delta_yields_stream_event(
        self, client: AnthropicCloudBackend
    ) -> None:
        """content_block_delta events yield StreamEvent."""
        delta = (
            'data: {"type": "content_block_delta", '
            '"delta": {"type": "text_delta", "text": "Hello"}}'
        )

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield delta

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_to_ir({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 1
            assert isinstance(events[0], StreamEvent)
            assert events[0].content == "Hello"

    async def test_message_delta_yields_text_result(self, client: AnthropicCloudBackend) -> None:
        """message_delta with stop_reason yields TextResult."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield 'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}'

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_to_ir({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 1
            assert isinstance(events[0], TextResult)
            assert events[0].finish_reason == "stop"

    async def test_message_stop_ends_stream(self, client: AnthropicCloudBackend) -> None:
        """message_stop ends the stream."""
        delta_hello = (
            'data: {"type": "content_block_delta", '
            '"delta": {"type": "text_delta", "text": "Hello"}}'
        )
        delta_not_shown = (
            'data: {"type": "content_block_delta", '
            '"delta": {"type": "text_delta", "text": "Should not appear"}}'
        )

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield delta_hello
            yield 'data: {"type": "message_stop"}'
            yield delta_not_shown

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_to_ir({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 1
            assert isinstance(events[0], StreamEvent)
            assert events[0].content == "Hello"

    async def test_empty_lines_ignored(self, client: AnthropicCloudBackend) -> None:
        """Empty lines are ignored."""
        delta_hello = (
            'data: {"type": "content_block_delta", '
            '"delta": {"type": "text_delta", "text": "Hello"}}'
        )

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield ""
            yield "   "
            yield delta_hello

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_to_ir({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 1

    async def test_malformed_json_ignored(self, client: AnthropicCloudBackend) -> None:
        """Malformed JSON is ignored."""
        delta_hello = (
            'data: {"type": "content_block_delta", '
            '"delta": {"type": "text_delta", "text": "Hello"}}'
        )

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "data: not valid json"
            yield delta_hello

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_to_ir({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 1
            assert isinstance(events[0], StreamEvent)
            assert events[0].content == "Hello"

    async def test_event_lines_ignored(self, client: AnthropicCloudBackend) -> None:
        """Lines starting with 'event:' are ignored (not data: prefix)."""
        delta_hello = (
            'data: {"type": "content_block_delta", '
            '"delta": {"type": "text_delta", "text": "Hello"}}'
        )

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "event: message_start"
            yield "event: content_block_start"
            yield delta_hello

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_to_ir({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 1

    async def test_max_tokens_stop_reason_mapped(self, client: AnthropicCloudBackend) -> None:
        """max_tokens stop_reason mapped to 'length' finish_reason."""

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield 'data: {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}'

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            events = []
            async for event in client._stream_to_ir({"model": "claude-3-opus"}):
                events.append(event)

            assert len(events) == 1
            assert isinstance(events[0], TextResult)
            assert events[0].finish_reason == "length"


class TestForwardRequestSameProtocol:
    """Tests for forward_request with same-protocol (Anthropic->Anthropic) passthrough."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    async def test_non_streaming_returns_raw_response(self, client: AnthropicCloudBackend) -> None:
        """Same-protocol non-streaming returns raw_response."""
        original = MagicMock()
        original.model_dump.return_value = {
            "model": "claude-3-opus-original",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 200,
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            ir = _make_ir(
                model="claude-3-opus-routed",
                original_protocol=ApiType.ANTHROPIC,
                original_request=original,
            )
            outcome = await client.forward_request(ir)

            assert outcome.raw_response is not None
            assert outcome.ir_result is None
            assert outcome.raw_response["id"] == "msg_123"

    async def test_streaming_returns_raw_stream(self, client: AnthropicCloudBackend) -> None:
        """Same-protocol streaming returns raw_stream."""
        original = MagicMock()
        original.model_dump.return_value = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 200,
        }

        async def mock_stream_lines() -> AsyncGenerator[str, None]:
            yield "event: content_block_delta"
            yield 'data: {"type": "content_block_delta", "delta": {"text": "Hi"}}'

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream_lines()):
            ir = _make_ir(
                stream=True,
                original_protocol=ApiType.ANTHROPIC,
                original_request=original,
            )
            outcome = await client.forward_request(ir)

            assert outcome.raw_stream is not None
            assert outcome.ir_stream is None

    async def test_passthrough_overrides_model(self, client: AnthropicCloudBackend) -> None:
        """Same-protocol passthrough overrides model from IR."""
        original = MagicMock()
        original.model_dump.return_value = {
            "model": "claude-3-opus-original",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 200,
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {},
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            ir = _make_ir(
                model="claude-3-opus-routed",
                original_protocol=ApiType.ANTHROPIC,
                original_request=original,
            )
            await client.forward_request(ir)

            json_data = mock_post.call_args[1]["json"]
            assert json_data["model"] == "claude-3-opus-routed"


class TestForwardRequestCrossProtocol:
    """Tests for forward_request with cross-protocol (OpenAI->Anthropic) conversion."""

    @pytest.fixture
    def client(self) -> AnthropicCloudBackend:
        """Create a test client."""
        return AnthropicCloudBackend(api_key="test-key")

    async def test_non_streaming_returns_ir_result(self, client: AnthropicCloudBackend) -> None:
        """Cross-protocol non-streaming returns ir_result."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello, how can I help?"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            ir = _make_ir(original_protocol=ApiType.OPENAI)
            outcome = await client.forward_request(ir)

            assert outcome.ir_result is not None
            assert outcome.raw_response is None
            assert isinstance(outcome.ir_result, InferenceResult)
            assert outcome.ir_result.result.content == "Hello, how can I help?"
            assert outcome.ir_result.result.finish_reason == "stop"
            assert outcome.ir_result.prompt_tokens == 10
            assert outcome.ir_result.completion_tokens == 8

    async def test_streaming_returns_ir_stream(self, client: AnthropicCloudBackend) -> None:
        """Cross-protocol streaming returns ir_stream."""
        delta_hello = (
            'data: {"type": "content_block_delta", '
            '"delta": {"type": "text_delta", "text": "Hello"}}'
        )

        async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield delta_hello
            yield 'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}'

        with patch.object(client, "_stream_with_circuit_breaker", return_value=mock_stream()):
            ir = _make_ir(stream=True, original_protocol=ApiType.OPENAI)
            outcome = await client.forward_request(ir)

            assert outcome.ir_stream is not None
            assert outcome.raw_stream is None

            events = [event async for event in outcome.ir_stream]
            assert len(events) == 2
            assert isinstance(events[0], StreamEvent)
            assert events[0].content == "Hello"
            assert isinstance(events[1], TextResult)
            assert events[1].finish_reason == "stop"

    async def test_cross_protocol_translates_request(self, client: AnthropicCloudBackend) -> None:
        """Cross-protocol translates IR to Anthropic format."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "usage": {},
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            ir = _make_ir(
                model="claude-3-opus",
                messages=[
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                original_protocol=ApiType.OPENAI,
                max_tokens=200,
                temperature=0.7,
            )
            await client.forward_request(ir)

            json_data = mock_post.call_args[1]["json"]
            assert json_data["model"] == "claude-3-opus"
            assert json_data["max_tokens"] == 200
            assert json_data["temperature"] == 0.7
            assert json_data["system"] == "Be helpful"
            # System message extracted, only user message in messages
            assert len(json_data["messages"]) == 1
            assert json_data["messages"][0]["role"] == "user"

    async def test_none_protocol_treated_as_cross_protocol(
        self, client: AnthropicCloudBackend
    ) -> None:
        """None original_protocol treated as cross-protocol."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "usage": {},
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            ir = _make_ir(original_protocol=None)
            outcome = await client.forward_request(ir)

            assert outcome.ir_result is not None
            assert outcome.raw_response is None


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
        backend = create_anthropic_backend(api_key="test-key", base_url="https://custom.api.com")
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
