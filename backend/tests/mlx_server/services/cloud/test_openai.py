"""Tests for OpenAI cloud backend client."""

import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mlx_manager.mlx_server.models.ir import (
    InferenceResult,
    InternalRequest,
    RoutingOutcome,
    StreamEvent,
    TextResult,
)
from mlx_manager.mlx_server.services.cloud.openai import (
    OPENAI_API_URL,
    OpenAICloudBackend,
    create_openai_backend,
)
from mlx_manager.models.enums import ApiType
from mlx_manager.models.value_objects import InferenceParams


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


def _make_ir(
    *,
    model: str = "gpt-4",
    messages: list[dict[str, Any]] | None = None,
    stream: bool = False,
    original_protocol: ApiType | None = ApiType.OPENAI,
    original_request: Any = None,
    stop: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 100,
    temperature: float = 1.0,
) -> InternalRequest:
    """Helper to create InternalRequest for tests."""
    return InternalRequest(
        model=model,
        messages=messages or [{"role": "user", "content": "Hi"}],
        params=InferenceParams(max_tokens=max_tokens, temperature=temperature),
        stream=stream,
        stop=stop,
        tools=tools,
        original_protocol=original_protocol,
        original_request=original_request,
    )


class TestForwardRequestSameProtocol:
    """Tests for forward_request with same-protocol (OpenAI->OpenAI) passthrough."""

    @pytest.fixture
    def backend(self) -> OpenAICloudBackend:
        """Create a test backend."""
        return OpenAICloudBackend(api_key="test-key")

    async def test_non_streaming_returns_raw_response(self, backend: OpenAICloudBackend) -> None:
        """Same-protocol non-streaming returns RoutingOutcome with raw_response."""
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

            ir = _make_ir(original_protocol=ApiType.OPENAI)
            outcome = await backend.forward_request(ir)

            assert isinstance(outcome, RoutingOutcome)
            assert outcome.raw_response is not None
            assert outcome.raw_response["id"] == "chatcmpl-123"
            assert outcome.ir_result is None

    async def test_streaming_returns_raw_stream(self, backend: OpenAICloudBackend) -> None:
        """Same-protocol streaming returns RoutingOutcome with raw_stream."""

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
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
            ir = _make_ir(stream=True, original_protocol=ApiType.OPENAI)
            outcome = await backend.forward_request(ir)

            assert outcome.raw_stream is not None
            assert outcome.ir_stream is None

            # Consume stream
            chunks = [chunk async for chunk in outcome.raw_stream]
            assert len(chunks) == 1
            assert chunks[0]["id"] == "chunk-1"

    async def test_passthrough_uses_original_request_when_available(
        self, backend: OpenAICloudBackend
    ) -> None:
        """Same-protocol passthrough uses original_request.model_dump() when set."""
        original = MagicMock()
        original.model_dump.return_value = {
            "model": "gpt-4-original",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 200,
            "temperature": 0.5,
            "top_p": 0.9,
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi"}}]}

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            ir = _make_ir(
                model="gpt-4-routed",
                original_protocol=ApiType.OPENAI,
                original_request=original,
            )
            await backend.forward_request(ir)

            # Verify model is overridden
            json_data = mock_post.call_args[1]["json"]
            assert json_data["model"] == "gpt-4-routed"
            # Verify original request fields preserved
            assert json_data["top_p"] == 0.9
            assert json_data["stream"] is False


class TestForwardRequestCrossProtocol:
    """Tests for forward_request with cross-protocol (Anthropic->OpenAI) conversion."""

    @pytest.fixture
    def backend(self) -> OpenAICloudBackend:
        """Create a test backend."""
        return OpenAICloudBackend(api_key="test-key")

    async def test_non_streaming_returns_ir_result(self, backend: OpenAICloudBackend) -> None:
        """Cross-protocol non-streaming returns RoutingOutcome with ir_result."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"content": "Hello!", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            ir = _make_ir(original_protocol=ApiType.ANTHROPIC)
            outcome = await backend.forward_request(ir)

            assert outcome.ir_result is not None
            assert outcome.raw_response is None
            assert isinstance(outcome.ir_result, InferenceResult)
            assert outcome.ir_result.result.content == "Hello!"
            assert outcome.ir_result.result.finish_reason == "stop"
            assert outcome.ir_result.prompt_tokens == 10
            assert outcome.ir_result.completion_tokens == 5

    async def test_streaming_returns_ir_stream(self, backend: OpenAICloudBackend) -> None:
        """Cross-protocol streaming returns RoutingOutcome with ir_stream."""

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
            yield 'data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]}'
            yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}'
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm):
            ir = _make_ir(stream=True, original_protocol=ApiType.ANTHROPIC)
            outcome = await backend.forward_request(ir)

            assert outcome.ir_stream is not None
            assert outcome.raw_stream is None

            events = [event async for event in outcome.ir_stream]
            assert len(events) == 2
            # First event: content
            assert isinstance(events[0], StreamEvent)
            assert events[0].content == "Hi"
            # Second event: finish
            assert isinstance(events[1], TextResult)
            assert events[1].finish_reason == "stop"

    async def test_cross_protocol_builds_from_ir_fields(self, backend: OpenAICloudBackend) -> None:
        """Cross-protocol builds request from IR fields, not original_request."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            ir = _make_ir(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                original_protocol=ApiType.ANTHROPIC,
                max_tokens=200,
                temperature=0.7,
                stop=["END"],
                tools=[{"type": "function", "function": {"name": "test"}}],
            )
            await backend.forward_request(ir)

            json_data = mock_post.call_args[1]["json"]
            assert json_data["model"] == "gpt-4"
            assert json_data["messages"] == [{"role": "user", "content": "Hello"}]
            assert json_data["max_tokens"] == 200
            assert json_data["temperature"] == 0.7
            assert json_data["stop"] == ["END"]
            assert json_data["tools"] == [{"type": "function", "function": {"name": "test"}}]
            assert json_data["stream"] is False

    async def test_none_protocol_treated_as_cross_protocol(
        self, backend: OpenAICloudBackend
    ) -> None:
        """None original_protocol treated as cross-protocol."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {},
        }

        with patch.object(backend._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            ir = _make_ir(original_protocol=None)
            outcome = await backend.forward_request(ir)

            assert outcome.ir_result is not None
            assert outcome.raw_response is None


class TestParseResponseToIR:
    """Tests for _parse_response_to_ir static method."""

    def test_parses_basic_response(self) -> None:
        """Parses basic OpenAI response to InferenceResult."""
        response = {
            "choices": [
                {
                    "message": {"content": "Hello world", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = OpenAICloudBackend._parse_response_to_ir(response)

        assert isinstance(result, InferenceResult)
        assert result.result.content == "Hello world"
        assert result.result.finish_reason == "stop"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

    def test_parses_tool_calls(self) -> None:
        """Parses response with tool_calls."""
        tool_calls = [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
        response = {
            "choices": [
                {
                    "message": {"content": "", "tool_calls": tool_calls},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = OpenAICloudBackend._parse_response_to_ir(response)

        assert result.result.tool_calls == tool_calls
        assert result.result.finish_reason == "tool_calls"

    def test_parses_reasoning_content(self) -> None:
        """Parses response with reasoning_content."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "reasoning_content": "thinking...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = OpenAICloudBackend._parse_response_to_ir(response)

        assert result.result.reasoning_content == "thinking..."

    def test_handles_missing_usage(self) -> None:
        """Handles response without usage field."""
        response = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
        }
        result = OpenAICloudBackend._parse_response_to_ir(response)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


class TestStreamToIR:
    """Tests for _stream_to_ir static method."""

    async def test_converts_content_chunks(self) -> None:
        """Converts OpenAI content delta chunks to StreamEvent."""

        async def mock_stream() -> AsyncGenerator[dict, None]:
            yield {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]}

        events = [event async for event in OpenAICloudBackend._stream_to_ir(mock_stream())]

        assert len(events) == 2
        assert all(isinstance(e, StreamEvent) for e in events)
        assert events[0].content == "Hello"
        assert events[1].content == " world"

    async def test_converts_finish_reason(self) -> None:
        """Converts finish_reason chunk to TextResult."""

        async def mock_stream() -> AsyncGenerator[dict, None]:
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        events = [event async for event in OpenAICloudBackend._stream_to_ir(mock_stream())]

        assert len(events) == 1
        assert isinstance(events[0], TextResult)
        assert events[0].finish_reason == "stop"

    async def test_converts_reasoning_content(self) -> None:
        """Converts reasoning_content delta to StreamEvent."""

        async def mock_stream() -> AsyncGenerator[dict, None]:
            yield {
                "choices": [{"delta": {"reasoning_content": "thinking..."}, "finish_reason": None}]
            }

        events = [event async for event in OpenAICloudBackend._stream_to_ir(mock_stream())]

        assert len(events) == 1
        assert isinstance(events[0], StreamEvent)
        assert events[0].reasoning_content == "thinking..."

    async def test_skips_empty_choices(self) -> None:
        """Skips chunks with empty choices."""

        async def mock_stream() -> AsyncGenerator[dict, None]:
            yield {"choices": []}
            yield {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}

        events = [event async for event in OpenAICloudBackend._stream_to_ir(mock_stream())]

        assert len(events) == 1
        assert events[0].content == "Hi"


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

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
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
            chunks = [chunk async for chunk in backend._stream_chat_completion({"stream": True})]

            assert len(chunks) == 2
            assert chunks[0] == chunk1
            assert chunks[1] == chunk2

    async def test_skips_empty_lines(self, backend: OpenAICloudBackend) -> None:
        """Skips empty lines in SSE stream."""
        chunk = {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
            yield ""
            yield f"data: {json.dumps(chunk)}"
            yield "   "
            yield "data: [DONE]"

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm):
            chunks = [c async for c in backend._stream_chat_completion({"stream": True})]
            assert len(chunks) == 1

    async def test_handles_done_marker(self, backend: OpenAICloudBackend) -> None:
        """Stops iteration at [DONE] marker."""
        chunk = {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps(chunk)}"
            yield "data: [DONE]"
            yield 'data: {"should": "not appear"}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        with patch.object(backend._client, "stream", return_value=mock_stream_cm):
            chunks = [c async for c in backend._stream_chat_completion({"stream": True})]
            assert len(chunks) == 1

    async def test_handles_malformed_json_gracefully(self, backend: OpenAICloudBackend) -> None:
        """Skips malformed JSON without crashing."""
        valid_chunk = {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}

        async def mock_aiter_lines() -> AsyncGenerator[str, None]:
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
            chunks = [c async for c in backend._stream_chat_completion({"stream": True})]
            assert len(chunks) == 1
            assert chunks[0] == valid_chunk


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
