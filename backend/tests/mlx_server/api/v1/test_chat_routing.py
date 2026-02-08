"""Tests for chat endpoint routing integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.api.v1.chat import (
    _convert_messages_to_dicts,
    _convert_tool_calls,
    _handle_direct_request,
    _handle_non_streaming,
    _handle_routed_request,
    _handle_streaming,
    _handle_text_request,
    _handle_vision_request,
    create_chat_completion,
)
from mlx_manager.mlx_server.models.types import ModelType
from mlx_manager.mlx_server.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FunctionCall,
    FunctionDefinition,
    ResponseFormat,
    Tool,
    ToolCall,
)


@pytest.fixture
def basic_request():
    """Create a basic chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")],
    )


@pytest.fixture
def streaming_request():
    """Create a streaming chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")],
        stream=True,
    )


@pytest.fixture
def mock_completion_response():
    """Create a mock completion response dict."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


class TestRoutingDisabled:
    """Tests when cloud routing is disabled (default behavior)."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_direct_request")
    async def test_routing_disabled_goes_to_direct(self, mock_direct, mock_settings, basic_request):
        """When enable_cloud_routing=False, requests go to direct inference."""
        # Setup mock settings
        settings = MagicMock()
        settings.enable_cloud_routing = False
        settings.enable_batching = False
        mock_settings.return_value = settings

        # Setup mock response
        mock_direct.return_value = MagicMock()

        # Call the handler
        await _handle_text_request(basic_request)

        # Should go directly to direct handler, not router
        mock_direct.assert_called_once_with(basic_request)

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_batched_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_scheduler_manager")
    async def test_routing_disabled_batching_enabled(
        self, mock_scheduler_mgr, mock_batched, mock_settings, basic_request
    ):
        """When routing disabled but batching enabled, use batching."""
        settings = MagicMock()
        settings.enable_cloud_routing = False
        settings.enable_batching = True
        mock_settings.return_value = settings

        # Setup scheduler manager mock
        mgr = MagicMock()
        mgr.get_priority_for_request.return_value = MagicMock()
        mock_scheduler_mgr.return_value = mgr

        mock_batched.return_value = MagicMock()

        await _handle_text_request(basic_request)

        # Should use batching path
        mock_batched.assert_called_once()


class TestRoutingEnabled:
    """Tests when cloud routing is enabled."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    async def test_routing_enabled_uses_router(
        self, mock_get_router, mock_settings, basic_request, mock_completion_response
    ):
        """When enable_cloud_routing=True, requests go through router."""
        settings = MagicMock()
        settings.enable_cloud_routing = True
        settings.enable_batching = False
        settings.timeout_chat_seconds = 900.0  # Required for timeout handling
        mock_settings.return_value = settings

        # Setup router mock
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=mock_completion_response)
        mock_get_router.return_value = router_mock

        await _handle_text_request(basic_request)

        # Should use router
        router_mock.route_request.assert_called_once()

        # Verify router was called with correct params
        call_kwargs = router_mock.route_request.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["max_tokens"] == 4096  # default
        assert call_kwargs["stream"] is False

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    async def test_routed_request_non_streaming(
        self, mock_get_router, basic_request, mock_completion_response
    ):
        """Test _handle_routed_request returns ChatCompletionResponse."""
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=mock_completion_response)
        mock_get_router.return_value = router_mock

        result = await _handle_routed_request(basic_request)

        # Verify response type and content
        assert result.id == "chatcmpl-test123"
        assert result.model == "test-model"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello! How can I help?"
        assert result.usage.total_tokens == 15


class TestStreamingThroughRouter:
    """Tests for streaming responses through router."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    async def test_routed_request_streaming(self, mock_get_router, streaming_request):
        """Test _handle_routed_request returns EventSourceResponse for streaming."""

        # Create async generator for streaming
        async def mock_stream():
            chunks = [
                {
                    "id": "chatcmpl-test123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {"content": "Hello"}}],
                },
                {
                    "id": "chatcmpl-test123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {"content": "!"}}],
                },
            ]
            for chunk in chunks:
                yield chunk

        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=mock_stream())
        mock_get_router.return_value = router_mock

        result = await _handle_routed_request(streaming_request)

        # Should return EventSourceResponse
        assert isinstance(result, EventSourceResponse)


class TestRoutingFallback:
    """Tests for fallback behavior when routing fails."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_routed_request")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_direct_request")
    async def test_routing_failure_falls_back_to_direct(
        self, mock_direct, mock_routed, mock_settings, basic_request
    ):
        """When routing fails, fall back to direct inference."""
        settings = MagicMock()
        settings.enable_cloud_routing = True
        settings.enable_batching = False
        mock_settings.return_value = settings

        # Make routed request fail
        mock_routed.side_effect = RuntimeError("Router unavailable")

        mock_direct.return_value = MagicMock()

        # Should not raise, should fall back
        await _handle_text_request(basic_request)

        # Direct handler should be called as fallback
        mock_direct.assert_called_once_with(basic_request)

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_routed_request")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_batched_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_scheduler_manager")
    async def test_routing_failure_falls_back_to_batching(
        self, mock_scheduler_mgr, mock_batched, mock_routed, mock_settings, basic_request
    ):
        """When routing fails and batching is enabled, fall back to batching."""
        settings = MagicMock()
        settings.enable_cloud_routing = True
        settings.enable_batching = True
        mock_settings.return_value = settings

        # Make routed request fail
        mock_routed.side_effect = RuntimeError("Router unavailable")

        # Setup scheduler manager mock
        mgr = MagicMock()
        mgr.get_priority_for_request.return_value = MagicMock()
        mock_scheduler_mgr.return_value = mgr

        mock_batched.return_value = MagicMock()

        await _handle_text_request(basic_request)

        # Batching handler should be called as fallback
        mock_batched.assert_called_once()


class TestMessageConversion:
    """Tests for message format conversion."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    async def test_messages_converted_to_dict_format(
        self, mock_get_router, mock_completion_response
    ):
        """Test that ChatMessage objects are converted to dict format for router."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello!"),
                ChatMessage(role="assistant", content="Hi there!"),
                ChatMessage(role="user", content="How are you?"),
            ],
        )

        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=mock_completion_response)
        mock_get_router.return_value = router_mock

        await _handle_routed_request(request)

        # Check messages were converted correctly
        call_kwargs = router_mock.route_request.call_args.kwargs
        messages = call_kwargs["messages"]

        assert len(messages) == 4
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello!"}
        assert messages[2] == {"role": "assistant", "content": "Hi there!"}
        assert messages[3] == {"role": "user", "content": "How are you?"}


# ============================================================================
# Unit tests for _convert_messages_to_dicts
# ============================================================================


class TestConvertMessagesToDicts:
    """Tests for the _convert_messages_to_dicts helper."""

    def test_simple_string_content(self):
        """String content is preserved as-is."""
        messages = [ChatMessage(role="user", content="Hello")]
        result = _convert_messages_to_dicts(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_content_blocks_extract_text(self):
        """Content blocks are converted to text via extract_content_parts."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            )
        ]
        result = _convert_messages_to_dicts(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Describe this image"

    def test_none_content_preserved(self):
        """None content (for assistant messages with only tool_calls) is preserved."""
        messages = [
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(name="get_weather", arguments='{"location":"Tokyo"}'),
                    )
                ],
            )
        ]
        result = _convert_messages_to_dicts(messages)
        assert result[0]["content"] is None
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_call_id_preserved(self):
        """tool_call_id field is preserved in conversion."""
        messages = [
            ChatMessage(
                role="tool",
                content='{"temperature": 22}',
                tool_call_id="call_1",
            )
        ]
        result = _convert_messages_to_dicts(messages)
        assert result[0]["tool_call_id"] == "call_1"

    def test_multi_message_conversation(self):
        """Multi-turn conversation is correctly converted."""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
        ]
        result = _convert_messages_to_dicts(messages)
        assert len(result) == 3
        assert all(isinstance(m, dict) for m in result)
        assert result[0]["role"] == "system"
        assert result[2]["content"] == "Hello!"


# ============================================================================
# Unit tests for _convert_tool_calls
# ============================================================================


class TestConvertToolCalls:
    """Tests for the _convert_tool_calls helper."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert _convert_tool_calls(None) is None

    def test_empty_list_returns_none(self):
        """Empty list returns None."""
        assert _convert_tool_calls([]) is None

    def test_valid_tool_calls_converted(self):
        """Valid tool call dicts are converted to ToolCall objects."""
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'},
            }
        ]
        result = _convert_tool_calls(tool_calls)
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].id == "call_abc"
        assert result[0].function.name == "get_weather"

    def test_multiple_tool_calls(self):
        """Multiple tool calls are all converted."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "func_a", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "func_b", "arguments": '{"x":1}'},
            },
        ]
        result = _convert_tool_calls(tool_calls)
        assert result is not None
        assert len(result) == 2
        assert result[1].function.name == "func_b"


# ============================================================================
# Tests for _handle_direct_request
# ============================================================================


class TestHandleDirectRequest:
    """Tests for the direct inference path."""

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_non_streaming_basic(self, mock_settings, mock_generate):
        """Non-streaming direct request returns ChatCompletionResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = {
            "id": "chatcmpl-abc123",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=False,
        )

        result = await _handle_direct_request(request)
        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].message.content == "Hello there!"
        assert result.usage.total_tokens == 8

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_returns_event_source(self, mock_settings, mock_generate):
        """Streaming direct request returns EventSourceResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_stream():
            yield {"choices": [{"delta": {"content": "Hi"}}]}

        mock_generate.return_value = mock_stream()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        )

        result = await _handle_direct_request(request)
        assert isinstance(result, EventSourceResponse)

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_stop_string_converted_to_list(self, mock_settings, mock_generate):
        """Single stop string is converted to a list."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = {
            "id": "chatcmpl-abc",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stop="END",
        )

        await _handle_direct_request(request)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["stop"] == ["END"]

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_stop_list_passed_through(self, mock_settings, mock_generate):
        """Stop list is passed through unchanged."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = {
            "id": "chatcmpl-abc",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stop=["<end>", "<stop>"],
        )

        await _handle_direct_request(request)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["stop"] == ["<end>", "<stop>"]

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_tools_passed_when_tool_choice_not_none(self, mock_settings, mock_generate):
        """Tools are passed to inference when tool_choice is not 'none'."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = {
            "id": "chatcmpl-abc",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"Tokyo"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Weather in Tokyo?")],
            tools=[
                Tool(
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get weather",
                        parameters={
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    )
                )
            ],
            tool_choice="auto",
        )

        result = await _handle_direct_request(request)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 1

        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].message.tool_calls is not None
        assert result.choices[0].message.tool_calls[0].function.name == "get_weather"

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_tools_excluded_when_tool_choice_is_none(self, mock_settings, mock_generate):
        """Tools are NOT passed when tool_choice is 'none'."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = {
            "id": "chatcmpl-abc",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "No tools used."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            tools=[
                Tool(
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get weather",
                    )
                )
            ],
            tool_choice="none",
        )

        await _handle_direct_request(request)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["tools"] is None

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_reasoning_content_preserved(self, mock_settings, mock_generate):
        """reasoning_content from the service is included in the response."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = {
            "id": "chatcmpl-abc",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "42",
                        "reasoning_content": "Let me think step by step...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="What is 6*7?")],
        )

        result = await _handle_direct_request(request)
        assert result.choices[0].message.reasoning_content == "Let me think step by step..."


# ============================================================================
# Tests for _handle_non_streaming (structured output validation)
# ============================================================================


class TestNonStreamingStructuredOutput:
    """Tests for structured output validation in non-streaming path."""

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_structured_output_valid_json(self, mock_settings, mock_generate):
        """Valid JSON matching schema passes validation."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = {
            "id": "chatcmpl-abc",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"name":"Alice","age":30}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        schema = {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        }

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate a person")],
            response_format=ResponseFormat(type="json_schema", json_schema=schema),
        )

        result = await _handle_non_streaming(
            request, [{"role": "user", "content": "Generate a person"}], None
        )
        assert result.choices[0].message.content == '{"name":"Alice","age":30}'

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_structured_output_invalid_json_raises_400(self, mock_settings, mock_generate):
        """Invalid JSON that doesn't match schema raises HTTPException 400."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = {
            "id": "chatcmpl-abc",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "not valid json at all"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        schema = {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Generate a person")],
            response_format=ResponseFormat(type="json_schema", json_schema=schema),
        )

        with pytest.raises(HTTPException) as exc_info:
            await _handle_non_streaming(
                request, [{"role": "user", "content": "Generate a person"}], None
            )
        assert exc_info.value.status_code == 400
        assert "JSON schema validation" in exc_info.value.detail

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_timeout_raises_408(self, mock_settings, mock_generate):
        """Timeout during generation raises TimeoutHTTPException (408)."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 0.001  # Very short timeout
        mock_settings.return_value = settings

        mock_generate.side_effect = TimeoutError()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        with pytest.raises(HTTPException) as exc_info:
            await _handle_non_streaming(request, [{"role": "user", "content": "Hi"}], None)
        assert exc_info.value.status_code == 408


# ============================================================================
# Tests for _handle_streaming (timeout)
# ============================================================================


class TestHandleStreaming:
    """Tests for the streaming handler."""

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_returns_sse(self, mock_settings, mock_generate):
        """Streaming handler returns EventSourceResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_stream():
            yield {"choices": [{"delta": {"content": "hello"}}]}

        mock_generate.return_value = mock_stream()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=True,
        )

        result = await _handle_streaming(request, [{"role": "user", "content": "Hi"}], None)
        assert isinstance(result, EventSourceResponse)

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_with_tools(self, mock_settings, mock_generate):
        """Streaming with tools returns an EventSourceResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_stream():
            yield {"choices": [{"delta": {"content": "I'll check"}}]}

        mock_generate.return_value = mock_stream()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Weather?")],
            stream=True,
        )

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        result = await _handle_streaming(
            request, [{"role": "user", "content": "Weather?"}], None, tools
        )
        # The generator is lazy; it returns EventSourceResponse immediately
        assert isinstance(result, EventSourceResponse)


# ============================================================================
# Tests for _handle_vision_request
# ============================================================================


class TestHandleVisionRequest:
    """Tests for vision request handling."""

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_vision_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.preprocess_images")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_vision_non_streaming(
        self, mock_settings, mock_detect, mock_preprocess, mock_generate
    ):
        """Vision non-streaming returns ChatCompletionResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_detect.return_value = ModelType.VISION

        mock_preprocess.return_value = [MagicMock()]  # preprocessed image

        mock_generate.return_value = {
            "id": "chatcmpl-vision",
            "created": 1700000000,
            "model": "vision-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "I see a cat."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        }

        request = ChatCompletionRequest(
            model="vision-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What do you see?"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/cat.jpg"}},
                    ],
                )
            ],
        )

        result = await _handle_vision_request(request, ["http://example.com/cat.jpg"])
        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].message.content == "I see a cat."
        assert result.usage.total_tokens == 105

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_vision_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.preprocess_images")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_vision_streaming(
        self, mock_settings, mock_detect, mock_preprocess, mock_generate
    ):
        """Vision streaming returns EventSourceResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_detect.return_value = ModelType.VISION
        mock_preprocess.return_value = [MagicMock()]

        async def mock_stream():
            yield {"choices": [{"delta": {"content": "cat"}}]}

        mock_generate.return_value = mock_stream()

        request = ChatCompletionRequest(
            model="vision-model",
            messages=[ChatMessage(role="user", content="What?")],
            stream=True,
        )

        result = await _handle_vision_request(request, ["http://example.com/img.png"])
        assert isinstance(result, EventSourceResponse)

    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    async def test_vision_request_with_non_vision_model_raises_400(self, mock_detect):
        """Images sent to a non-vision model raises 400."""
        mock_detect.return_value = ModelType.TEXT_GEN

        request = ChatCompletionRequest(
            model="text-only-model",
            messages=[ChatMessage(role="user", content="Describe this")],
        )

        with pytest.raises(HTTPException) as exc_info:
            await _handle_vision_request(request, ["http://example.com/img.png"])
        assert exc_info.value.status_code == 400
        assert "vision model" in exc_info.value.detail.lower()

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_vision_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.preprocess_images")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_vision_timeout_raises_408(
        self, mock_settings, mock_detect, mock_preprocess, mock_generate
    ):
        """Vision timeout raises TimeoutHTTPException (408)."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 0.001
        mock_settings.return_value = settings

        mock_detect.return_value = ModelType.VISION
        mock_preprocess.return_value = [MagicMock()]
        mock_generate.side_effect = TimeoutError()

        request = ChatCompletionRequest(
            model="vision-model",
            messages=[ChatMessage(role="user", content="Describe")],
        )

        with pytest.raises(HTTPException) as exc_info:
            await _handle_vision_request(request, ["http://example.com/img.png"])
        assert exc_info.value.status_code == 408

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_vision_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.preprocess_images")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_vision_prompt_concatenation(
        self, mock_settings, mock_detect, mock_preprocess, mock_generate
    ):
        """Vision prompt correctly concatenates system/user/assistant messages."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_detect.return_value = ModelType.VISION
        mock_preprocess.return_value = [MagicMock()]
        mock_generate.return_value = {
            "id": "chatcmpl-v",
            "created": 1700000000,
            "model": "vision-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "A cat."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52},
        }

        request = ChatCompletionRequest(
            model="vision-model",
            messages=[
                ChatMessage(role="system", content="Be brief."),
                ChatMessage(role="user", content="What's this?"),
                ChatMessage(role="assistant", content="It looks like..."),
                ChatMessage(role="user", content="More detail?"),
            ],
        )

        await _handle_vision_request(request, ["http://example.com/img.png"])
        call_kwargs = mock_generate.call_args.kwargs
        prompt = call_kwargs["text_prompt"]
        assert "System: Be brief." in prompt
        assert "User: What's this?" in prompt
        assert "Assistant: It looks like..." in prompt
        assert "User: More detail?" in prompt


# ============================================================================
# Tests for create_chat_completion (the top-level endpoint)
# ============================================================================


class TestCreateChatCompletion:
    """Tests for the top-level create_chat_completion endpoint function."""

    @patch("mlx_manager.mlx_server.api.v1.chat._handle_text_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.audit_service")
    async def test_text_only_routes_to_text_handler(
        self, mock_audit, mock_detect, mock_text_handler
    ):
        """Text-only request with text model routes to _handle_text_request."""
        from mlx_manager.mlx_server.schemas.openai import (
            ChatCompletionChoice,
            Usage,
        )

        mock_detect.return_value = ModelType.TEXT_GEN

        response = ChatCompletionResponse(
            id="chatcmpl-abc",
            created=1700000000,
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )
        mock_text_handler.return_value = response

        # Setup audit context manager
        ctx = MagicMock()
        ctx.prompt_tokens = 0
        ctx.completion_tokens = 0
        ctx.total_tokens = 0
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        result = await create_chat_completion(request)
        mock_text_handler.assert_called_once_with(request)
        assert result is response

    @patch("mlx_manager.mlx_server.api.v1.chat._handle_vision_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.audit_service")
    async def test_images_route_to_vision_handler(
        self, mock_audit, mock_detect, mock_vision_handler
    ):
        """Request with images routes to _handle_vision_request."""
        from mlx_manager.mlx_server.schemas.openai import (
            ChatCompletionChoice,
            Usage,
        )

        mock_detect.return_value = ModelType.VISION

        response = ChatCompletionResponse(
            id="chatcmpl-vis",
            created=1700000000,
            model="vision-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="I see an image."),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=50, completion_tokens=5, total_tokens=55),
        )
        mock_vision_handler.return_value = response

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = ChatCompletionRequest(
            model="vision-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What's this?"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                    ],
                )
            ],
        )

        result = await create_chat_completion(request)
        mock_vision_handler.assert_called_once()
        assert result is response

    @patch("mlx_manager.mlx_server.api.v1.chat._handle_vision_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.audit_service")
    async def test_vision_model_without_images_routes_to_vision(
        self, mock_audit, mock_detect, mock_vision_handler
    ):
        """Vision model with text-only request still routes to vision handler."""
        from mlx_manager.mlx_server.schemas.openai import (
            ChatCompletionChoice,
            Usage,
        )

        mock_detect.return_value = ModelType.VISION

        response = ChatCompletionResponse(
            id="chatcmpl-vis",
            created=1700000000,
            model="vision-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        mock_vision_handler.return_value = response

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = ChatCompletionRequest(
            model="vision-model",
            messages=[ChatMessage(role="user", content="Hello, vision model!")],
        )

        await create_chat_completion(request)
        mock_vision_handler.assert_called_once()

    @patch("mlx_manager.mlx_server.api.v1.chat._handle_text_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.audit_service")
    async def test_runtime_error_raises_500(self, mock_audit, mock_detect, mock_text_handler):
        """RuntimeError during generation raises HTTPException 500."""
        mock_detect.return_value = ModelType.TEXT_GEN
        mock_text_handler.side_effect = RuntimeError("Model failed to load")

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_chat_completion(request)
        assert exc_info.value.status_code == 500
        assert "Model failed to load" in exc_info.value.detail

    @patch("mlx_manager.mlx_server.api.v1.chat._handle_text_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.audit_service")
    async def test_unexpected_error_raises_500_generic(
        self, mock_audit, mock_detect, mock_text_handler
    ):
        """Unexpected Exception raises generic 500 error."""
        mock_detect.return_value = ModelType.TEXT_GEN
        mock_text_handler.side_effect = ValueError("something unexpected")

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_chat_completion(request)
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal server error"

    @patch("mlx_manager.mlx_server.api.v1.chat._handle_text_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.audit_service")
    async def test_http_exception_reraised(self, mock_audit, mock_detect, mock_text_handler):
        """HTTPException from handler is re-raised, not wrapped."""
        mock_detect.return_value = ModelType.TEXT_GEN
        mock_text_handler.side_effect = HTTPException(status_code=400, detail="Bad request")

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_chat_completion(request)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Bad request"

    @patch("mlx_manager.mlx_server.api.v1.chat._handle_text_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.audit_service")
    async def test_audit_usage_updated_for_non_streaming(
        self, mock_audit, mock_detect, mock_text_handler
    ):
        """Audit context usage is updated for non-streaming response."""
        from mlx_manager.mlx_server.schemas.openai import (
            ChatCompletionChoice,
            Usage,
        )

        mock_detect.return_value = ModelType.TEXT_GEN

        response = ChatCompletionResponse(
            id="chatcmpl-abc",
            created=1700000000,
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_text_handler.return_value = response

        ctx = MagicMock()
        ctx.prompt_tokens = 0
        ctx.completion_tokens = 0
        ctx.total_tokens = 0
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        await create_chat_completion(request)
        assert ctx.prompt_tokens == 10
        assert ctx.completion_tokens == 5
        assert ctx.total_tokens == 15


# ============================================================================
# Tests for _handle_routed_request timeout
# ============================================================================


class TestRoutedRequestTimeout:
    """Tests for timeout in routed requests."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_routed_request_timeout_raises_408(self, mock_settings, mock_get_router):
        """Timeout during routed request raises 408."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 0.001
        mock_settings.return_value = settings

        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(side_effect=TimeoutError())
        mock_get_router.return_value = router_mock

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        with pytest.raises(HTTPException) as exc_info:
            await _handle_routed_request(request)
        assert exc_info.value.status_code == 408


# ============================================================================
# Tests for batching fallback paths
# ============================================================================


class TestBatchingFallback:
    """Tests for batching scheduler initialization fallback."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_scheduler_manager")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_direct_request")
    async def test_scheduler_not_initialized_falls_back_to_direct(
        self, mock_direct, mock_scheduler_mgr, mock_settings
    ):
        """When scheduler not initialized, fall back to direct."""
        settings = MagicMock()
        settings.enable_cloud_routing = False
        settings.enable_batching = True
        mock_settings.return_value = settings

        mock_scheduler_mgr.side_effect = RuntimeError("Scheduler not initialized")
        mock_direct.return_value = MagicMock()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        await _handle_text_request(request)
        mock_direct.assert_called_once()

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_scheduler_manager")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_direct_request")
    async def test_scheduler_other_error_falls_back_to_direct(
        self, mock_direct, mock_scheduler_mgr, mock_settings
    ):
        """When scheduler encounters other error, fall back to direct."""
        settings = MagicMock()
        settings.enable_cloud_routing = False
        settings.enable_batching = True
        mock_settings.return_value = settings

        mgr = MagicMock()
        mgr.get_priority_for_request.side_effect = ValueError("Unexpected")
        mock_scheduler_mgr.return_value = mgr
        mock_direct.return_value = MagicMock()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        await _handle_text_request(request)
        mock_direct.assert_called_once()


# ============================================================================
# Tests that consume SSE generators to cover inner generator bodies
# ============================================================================


class TestStreamingGeneratorConsumption:
    """Tests that actually iterate SSE generators to cover the inner bodies."""

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_generator_yields_chunks_and_done(self, mock_settings, mock_generate):
        """Iterating the SSE generator yields data chunks followed by [DONE]."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        chunk1 = {"id": "c1", "choices": [{"delta": {"content": "Hi"}}]}
        chunk2 = {"id": "c1", "choices": [{"delta": {"content": "!"}}]}

        async def mock_stream():
            yield chunk1
            yield chunk2

        mock_generate.return_value = mock_stream()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        )

        result = await _handle_streaming(request, [{"role": "user", "content": "Hello"}], None)
        # Consume the body iterator
        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "DONE" in event_text

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_generator_timeout_yields_error_event(
        self, mock_settings, mock_generate
    ):
        """When generate_chat_completion times out, the generator yields an error event."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 0.001
        mock_settings.return_value = settings

        mock_generate.side_effect = TimeoutError()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        )

        result = await _handle_streaming(request, [{"role": "user", "content": "Hello"}], None)
        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "error" in event_text.lower() or "timeout" in event_text.lower()

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_vision_completion")
    @patch("mlx_manager.mlx_server.api.v1.chat.preprocess_images")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_vision_streaming_generator_yields_chunks(
        self, mock_settings, mock_detect, mock_preprocess, mock_generate
    ):
        """Vision streaming generator yields data chunks and [DONE]."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_detect.return_value = ModelType.VISION
        mock_preprocess.return_value = [MagicMock()]

        async def mock_stream():
            yield {"id": "v1", "choices": [{"delta": {"content": "A cat"}}]}
            yield {"id": "v1", "choices": [{"delta": {"content": "."}}]}

        mock_generate.return_value = mock_stream()

        request = ChatCompletionRequest(
            model="vision-model",
            messages=[ChatMessage(role="user", content="What?")],
            stream=True,
        )

        result = await _handle_vision_request(request, ["http://example.com/img.png"])
        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "DONE" in event_text

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_routed_streaming_generator_yields_chunks(self, mock_settings, mock_get_router):
        """Routed streaming generator yields data chunks and [DONE]."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_stream():
            yield {"id": "r1", "choices": [{"delta": {"content": "Routed"}}]}

        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=mock_stream())
        mock_get_router.return_value = router_mock

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=True,
        )

        result = await _handle_routed_request(request)
        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "DONE" in event_text


# ============================================================================
# Tests for batching handlers
# ============================================================================


class TestHandleBatchedRequest:
    """Tests for the _handle_batched_request function."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_scheduler_manager")
    @patch("mlx_manager.mlx_server.api.v1.chat._complete_batched_response")
    @patch("mlx_manager.mlx_server.api.v1.chat._stream_batched_response")
    async def test_non_streaming_batched_request(
        self, mock_stream_batch, mock_complete_batch, mock_sched_mgr
    ):
        """Non-streaming batched request calls _complete_batched_response."""
        from mlx_manager.mlx_server.api.v1.chat import _handle_batched_request
        from mlx_manager.mlx_server.services.batching import Priority

        with (
            patch("mlx_manager.mlx_server.models.pool.get_model_pool") as mock_pool_fn,
            patch("mlx_manager.mlx_server.models.adapters.get_adapter") as mock_adapter_fn,
        ):
            loaded = MagicMock()
            loaded.model = MagicMock()
            loaded.tokenizer = MagicMock()
            loaded.tokenizer.tokenizer = MagicMock()
            loaded.tokenizer.tokenizer.encode.return_value = [1, 2, 3]
            pool = AsyncMock()
            pool.get_model = AsyncMock(return_value=loaded)
            mock_pool_fn.return_value = pool

            adapter = MagicMock()
            adapter.apply_chat_template.return_value = "Hello"
            mock_adapter_fn.return_value = adapter

            mgr = AsyncMock()
            scheduler = MagicMock()
            mgr.get_scheduler = AsyncMock(return_value=scheduler)
            mgr.configure_scheduler = AsyncMock()
            mock_sched_mgr.return_value = mgr

            mock_complete_batch.return_value = MagicMock()

            request = ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Hi")],
                stream=False,
            )

            await _handle_batched_request(request, Priority.NORMAL)
            mock_complete_batch.assert_called_once()
            mock_stream_batch.assert_not_called()

    @patch("mlx_manager.mlx_server.api.v1.chat.get_scheduler_manager")
    @patch("mlx_manager.mlx_server.api.v1.chat._complete_batched_response")
    @patch("mlx_manager.mlx_server.api.v1.chat._stream_batched_response")
    async def test_streaming_batched_request(
        self, mock_stream_batch, mock_complete_batch, mock_sched_mgr
    ):
        """Streaming batched request calls _stream_batched_response."""
        from mlx_manager.mlx_server.api.v1.chat import _handle_batched_request
        from mlx_manager.mlx_server.services.batching import Priority

        with (
            patch("mlx_manager.mlx_server.models.pool.get_model_pool") as mock_pool_fn,
            patch("mlx_manager.mlx_server.models.adapters.get_adapter") as mock_adapter_fn,
        ):
            loaded = MagicMock()
            loaded.model = MagicMock()
            loaded.tokenizer = MagicMock()
            loaded.tokenizer.tokenizer = MagicMock()
            loaded.tokenizer.tokenizer.encode.return_value = [1, 2, 3]
            pool = AsyncMock()
            pool.get_model = AsyncMock(return_value=loaded)
            mock_pool_fn.return_value = pool

            adapter = MagicMock()
            adapter.apply_chat_template.return_value = "Hello"
            mock_adapter_fn.return_value = adapter

            mgr = AsyncMock()
            scheduler = MagicMock()
            mgr.get_scheduler = AsyncMock(return_value=scheduler)
            mgr.configure_scheduler = AsyncMock()
            mock_sched_mgr.return_value = mgr

            mock_stream_batch.return_value = MagicMock()

            request = ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Hi")],
                stream=True,
            )

            await _handle_batched_request(request, Priority.NORMAL)
            mock_stream_batch.assert_called_once()
            mock_complete_batch.assert_not_called()


class TestCompleteBatchedResponse:
    """Tests for _complete_batched_response."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_collects_tokens_and_builds_response(self, mock_settings):
        """Collects all tokens and builds ChatCompletionResponse."""
        from mlx_manager.mlx_server.api.v1.chat import _complete_batched_response
        from mlx_manager.mlx_server.services.batching import BatchRequest, Priority

        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_token_stream():
            yield {"text": "Hello"}
            yield {"text": " world"}

        scheduler = MagicMock()
        scheduler.submit.return_value = mock_token_stream()

        batch_request = BatchRequest(
            request_id="chatcmpl-test",
            model_id="test-model",
            prompt_tokens=[1, 2, 3],
            max_tokens=100,
            priority=Priority.NORMAL,
        )

        result = await _complete_batched_response(batch_request, scheduler, "test-model")
        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].message.content == "Hello world"
        assert result.usage.prompt_tokens == 3
        assert result.usage.completion_tokens == 2

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_timeout_raises_408(self, mock_settings):
        """Timeout during batch collection raises 408."""
        import asyncio

        from mlx_manager.mlx_server.api.v1.chat import _complete_batched_response
        from mlx_manager.mlx_server.services.batching import BatchRequest, Priority

        settings = MagicMock()
        settings.timeout_chat_seconds = 0.001
        mock_settings.return_value = settings

        async def slow_token_stream():
            await asyncio.sleep(10)
            yield {"text": "never"}

        scheduler = MagicMock()
        scheduler.submit.return_value = slow_token_stream()

        batch_request = BatchRequest(
            request_id="chatcmpl-test",
            model_id="test-model",
            prompt_tokens=[1, 2, 3],
            max_tokens=100,
            priority=Priority.NORMAL,
        )

        with pytest.raises(HTTPException) as exc_info:
            await _complete_batched_response(batch_request, scheduler, "test-model")
        assert exc_info.value.status_code == 408


class TestStreamBatchedResponse:
    """Tests for _stream_batched_response."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streams_tokens_as_sse(self, mock_settings):
        """Streams tokens as SSE events and ends with [DONE]."""
        from mlx_manager.mlx_server.api.v1.chat import _stream_batched_response
        from mlx_manager.mlx_server.services.batching import BatchRequest, Priority

        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_token_stream():
            yield {"text": "Hi"}
            yield {"text": "!"}

        scheduler = MagicMock()
        scheduler.submit.return_value = mock_token_stream()

        batch_request = BatchRequest(
            request_id="chatcmpl-batch",
            model_id="test-model",
            prompt_tokens=[1, 2],
            max_tokens=100,
            priority=Priority.NORMAL,
        )

        result = await _stream_batched_response(batch_request, scheduler, "test-model")
        assert isinstance(result, EventSourceResponse)

        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "DONE" in event_text
        assert "Hi" in event_text
        assert "finish_reason" in event_text

    @patch("mlx_manager.mlx_server.api.v1.chat.time")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_timeout_yields_error_event(self, mock_settings, mock_time):
        """Timeout during batch streaming yields error event."""
        from mlx_manager.mlx_server.api.v1.chat import _stream_batched_response
        from mlx_manager.mlx_server.services.batching import BatchRequest, Priority

        settings = MagicMock()
        settings.timeout_chat_seconds = 0.0  # Immediate timeout
        mock_settings.return_value = settings

        mock_time.time.return_value = 1700000000

        # monotonic: first call returns start_time=0, second call returns 100 (past timeout)
        call_count = 0

        def monotonic_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return 0.0
            return 100.0

        mock_time.monotonic.side_effect = monotonic_side_effect

        async def mock_token_stream():
            yield {"text": "tok"}

        scheduler = MagicMock()
        scheduler.submit.return_value = mock_token_stream()

        batch_request = BatchRequest(
            request_id="chatcmpl-batch",
            model_id="test-model",
            prompt_tokens=[1],
            max_tokens=100,
            priority=Priority.NORMAL,
        )

        result = await _stream_batched_response(batch_request, scheduler, "test-model")
        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "error" in event_text.lower() or "timeout" in event_text.lower()
