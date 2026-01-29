"""Tests for /v1/messages Anthropic Messages API endpoint."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.api.v1.messages import (
    _handle_non_streaming,
    _handle_streaming,
    create_message,
)
from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    MessageParam,
    TextBlock,
    TextBlockParam,
)
from mlx_manager.mlx_server.services.protocol import InternalRequest, reset_translator


@pytest.fixture(autouse=True)
def reset_protocol_translator():
    """Reset the protocol translator singleton before each test."""
    reset_translator()


@pytest.fixture
def basic_request():
    """Create a basic Anthropic messages request."""
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=1000,
        messages=[MessageParam(role="user", content="Hello")],
    )


@pytest.fixture
def request_with_system():
    """Create a request with system message."""
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=1000,
        messages=[MessageParam(role="user", content="Hello")],
        system="You are a helpful assistant",
    )


@pytest.fixture
def streaming_request():
    """Create a streaming request."""
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=1000,
        messages=[MessageParam(role="user", content="Hello")],
        stream=True,
    )


@pytest.fixture
def mock_internal_request():
    """Create a mock internal request."""
    return InternalRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1000,
        temperature=1.0,
        top_p=None,
        stream=False,
        stop=None,
    )


class TestRequestValidation:
    """Tests for request validation."""

    def test_valid_request_with_max_tokens(self, basic_request):
        """Valid request with required max_tokens is accepted."""
        assert basic_request.max_tokens == 1000
        assert len(basic_request.messages) == 1

    def test_missing_max_tokens_raises_validation_error(self):
        """Missing max_tokens raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            AnthropicMessagesRequest(
                model="test-model",
                messages=[MessageParam(role="user", content="Hello")],
            )
        assert "max_tokens" in str(exc_info.value)

    def test_system_message_accepted(self, request_with_system):
        """System message is accepted in separate field."""
        assert request_with_system.system == "You are a helpful assistant"

    def test_system_message_as_list_of_blocks(self):
        """System message can be list of TextBlockParam."""
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[MessageParam(role="user", content="Hello")],
            system=[
                TextBlockParam(text="First."),
                TextBlockParam(text="Second."),
            ],
        )
        assert len(request.system) == 2

    def test_temperature_default_is_one(self, basic_request):
        """Temperature defaults to 1.0."""
        assert basic_request.temperature == 1.0

    def test_temperature_validated_range(self):
        """Temperature must be between 0.0 and 1.0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AnthropicMessagesRequest(
                model="test-model",
                max_tokens=1000,
                messages=[MessageParam(role="user", content="Hello")],
                temperature=1.5,  # Invalid - exceeds 1.0
            )


class TestNonStreamingResponse:
    """Tests for non-streaming /v1/messages response."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    async def test_response_has_anthropic_format(
        self, mock_generate, basic_request, mock_internal_request
    ):
        """Response has Anthropic format with msg_ prefix and content as list."""
        mock_generate.return_value = {
            "id": "chatcmpl-abc123",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        response = await _handle_non_streaming(basic_request, mock_internal_request)

        assert isinstance(response, AnthropicMessagesResponse)
        assert response.id.startswith("msg_")
        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert isinstance(response.content[0], TextBlock)
        assert response.content[0].text == "Hello there!"

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    async def test_stop_reason_translated_stop(
        self, mock_generate, basic_request, mock_internal_request
    ):
        """OpenAI 'stop' is translated to Anthropic 'end_turn'."""
        mock_generate.return_value = {
            "id": "chatcmpl-abc123",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        response = await _handle_non_streaming(basic_request, mock_internal_request)

        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    async def test_stop_reason_translated_length(
        self, mock_generate, basic_request, mock_internal_request
    ):
        """OpenAI 'length' is translated to Anthropic 'max_tokens'."""
        mock_generate.return_value = {
            "id": "chatcmpl-abc123",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        response = await _handle_non_streaming(basic_request, mock_internal_request)

        assert response.stop_reason == "max_tokens"

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    async def test_usage_included(
        self, mock_generate, basic_request, mock_internal_request
    ):
        """Usage statistics are included in response."""
        mock_generate.return_value = {
            "id": "chatcmpl-abc123",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        response = await _handle_non_streaming(basic_request, mock_internal_request)

        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50


class TestStreamingResponse:
    """Tests for streaming /v1/messages response."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    async def test_returns_event_source_response(
        self, mock_generate, streaming_request
    ):
        """Streaming returns EventSourceResponse."""
        # Create async generator mock
        async def mock_stream():
            yield {
                "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        mock_generate.return_value = mock_stream()

        internal = InternalRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1000,
            temperature=1.0,
            top_p=None,
            stream=True,
            stop=None,
        )

        response = await _handle_streaming(streaming_request, internal)

        assert isinstance(response, EventSourceResponse)

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    async def test_streaming_events_format(self, mock_generate, streaming_request):
        """Streaming events have correct Anthropic format."""
        # Create async generator mock
        async def mock_stream():
            yield {
                "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        mock_generate.return_value = mock_stream()

        internal = InternalRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1000,
            temperature=1.0,
            top_p=None,
            stream=True,
            stop=None,
        )

        response = await _handle_streaming(streaming_request, internal)

        # Collect events
        events = []
        async for event in response.body_iterator:
            events.append(event)

        # Should have at minimum: message_start, content_block_start,
        # content_block_delta (tokens), content_block_stop, message_delta, message_stop
        assert len(events) >= 5

        # Check first event is message_start
        first_event = events[0]
        assert first_event["event"] == "message_start"
        data = json.loads(first_event["data"])
        assert data["type"] == "message_start"
        assert data["message"]["id"].startswith("msg_")

        # Check second event is content_block_start
        second_event = events[1]
        assert second_event["event"] == "content_block_start"
        data = json.loads(second_event["data"])
        assert data["type"] == "content_block_start"
        assert data["index"] == 0

        # Find content_block_delta event
        delta_events = [
            e for e in events if e.get("event") == "content_block_delta"
        ]
        assert len(delta_events) >= 1
        delta_data = json.loads(delta_events[0]["data"])
        assert delta_data["type"] == "content_block_delta"
        assert delta_data["delta"]["type"] == "text_delta"

        # Check last events
        stop_events = [e for e in events if e.get("event") == "content_block_stop"]
        assert len(stop_events) == 1

        delta_final_events = [e for e in events if e.get("event") == "message_delta"]
        assert len(delta_final_events) == 1
        delta_final_data = json.loads(delta_final_events[0]["data"])
        assert "stop_reason" in delta_final_data["delta"]

        message_stop_events = [e for e in events if e.get("event") == "message_stop"]
        assert len(message_stop_events) == 1


class TestProtocolTranslationIntegration:
    """Tests for protocol translation integration."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.messages.get_translator")
    async def test_system_message_in_internal_messages(
        self, mock_get_translator, mock_generate, request_with_system
    ):
        """System message is placed in internal messages array."""
        # Set up translator mock
        mock_translator = MagicMock()
        internal = InternalRequest(
            model="test-model",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=1000,
            temperature=1.0,
            top_p=None,
            stream=False,
            stop=None,
        )
        mock_translator.anthropic_to_internal.return_value = internal
        mock_translator.openai_stop_to_anthropic.return_value = "end_turn"
        mock_get_translator.return_value = mock_translator

        mock_generate.return_value = {
            "id": "chatcmpl-abc123",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        await create_message(request_with_system)

        # Verify generate_chat_completion was called with system message in messages
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.messages.get_translator")
    async def test_temperature_passed_through(
        self, mock_get_translator, mock_generate, basic_request
    ):
        """Temperature is passed through to generate_chat_completion."""
        mock_translator = MagicMock()
        internal = InternalRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            stream=False,
            stop=None,
        )
        mock_translator.anthropic_to_internal.return_value = internal
        mock_translator.openai_stop_to_anthropic.return_value = "end_turn"
        mock_get_translator.return_value = mock_translator

        mock_generate.return_value = {
            "id": "chatcmpl-abc123",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        basic_request.temperature = 0.7
        basic_request.top_p = 0.9
        await create_message(basic_request)

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.messages.get_translator")
    async def test_generic_exception_returns_500(
        self, mock_get_translator, mock_generate, basic_request
    ):
        """Generic exception returns HTTP 500."""
        mock_translator = MagicMock()
        mock_translator.anthropic_to_internal.side_effect = RuntimeError("Test error")
        mock_get_translator.return_value = mock_translator

        with pytest.raises(HTTPException) as exc_info:
            await create_message(basic_request)

        assert exc_info.value.status_code == 500
        assert "Test error" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.messages.get_translator")
    async def test_http_exception_reraises(
        self, mock_get_translator, mock_generate, basic_request
    ):
        """HTTPException is re-raised without wrapping."""
        mock_translator = MagicMock()
        mock_translator.anthropic_to_internal.side_effect = HTTPException(
            status_code=400, detail="Bad request"
        )
        mock_get_translator.return_value = mock_translator

        with pytest.raises(HTTPException) as exc_info:
            await create_message(basic_request)

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Bad request"


class TestCreateMessageEndpoint:
    """Tests for the main create_message endpoint."""

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    async def test_non_streaming_path(self, mock_generate, basic_request):
        """Non-streaming request returns AnthropicMessagesResponse."""
        mock_generate.return_value = {
            "id": "chatcmpl-abc123",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        response = await create_message(basic_request)

        assert isinstance(response, AnthropicMessagesResponse)

    @pytest.mark.asyncio
    @patch("mlx_manager.mlx_server.api.v1.messages.generate_chat_completion")
    async def test_streaming_path(self, mock_generate, streaming_request):
        """Streaming request returns EventSourceResponse."""
        async def mock_stream():
            yield {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        mock_generate.return_value = mock_stream()

        response = await create_message(streaming_request)

        assert isinstance(response, EventSourceResponse)
