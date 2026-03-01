"""Tests for chat endpoint routing integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.api.v1.chat import (
    _convert_tool_calls,
    _format_ir_complete,
    _format_ir_stream_as_sse,
    _handle_direct_inference,
    _handle_non_streaming,
    _handle_streaming,
    _handle_text_request,
    _route_and_respond,
    create_chat_completion,
)
from mlx_manager.mlx_server.models.ir import (
    InferenceResult,
    InternalRequest,
    RoutingOutcome,
    StreamEvent,
    TextResult,
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
    Usage,
)
from mlx_manager.mlx_server.services.formatters import OpenAIFormatter


def _make_inference_result(
    content: str = "Hello there!",
    finish_reason: str = "stop",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    reasoning_content: str | None = None,
    tool_calls: list[dict] | None = None,
) -> InferenceResult:
    return InferenceResult(
        result=TextResult(
            content=content,
            finish_reason=finish_reason,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _make_ir(request: ChatCompletionRequest) -> InternalRequest:
    """Build an InternalRequest from a ChatCompletionRequest."""
    return OpenAIFormatter.parse_request(request)


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
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_direct_inference")
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
        await _handle_text_request(basic_request, None)

        # Should go directly to direct handler, not router
        mock_direct.assert_called_once()
        # Verify it received an InternalRequest and the original request
        call_args = mock_direct.call_args
        assert isinstance(call_args[0][0], InternalRequest)
        assert call_args[0][1] is basic_request

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

        await _handle_text_request(basic_request, None)

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

        # Setup router mock to return a RoutingOutcome
        outcome = RoutingOutcome(
            raw_response=mock_completion_response,
        )
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        await _handle_text_request(basic_request, None)

        # Should use router
        router_mock.route_request.assert_called_once()

        # Verify router was called with InternalRequest
        call_args = router_mock.route_request.call_args
        ir = call_args[0][0]
        assert isinstance(ir, InternalRequest)
        assert ir.model == "test-model"
        assert ir.stream is False

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_routed_request_non_streaming(
        self, mock_settings, mock_get_router, basic_request, mock_completion_response
    ):
        """Test _route_and_respond returns ChatCompletionResponse for passthrough."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        outcome = RoutingOutcome(raw_response=mock_completion_response)
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        ir = _make_ir(basic_request)
        result = await _route_and_respond(ir, basic_request)

        # Verify response type and content
        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-test123"
        assert result.model == "test-model"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello! How can I help?"
        assert result.usage.total_tokens == 15


class TestStreamingThroughRouter:
    """Tests for streaming responses through router."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_routed_request_streaming_passthrough(
        self, mock_settings, mock_get_router, streaming_request
    ):
        """Test _route_and_respond returns EventSourceResponse for passthrough streaming."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

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

        outcome = RoutingOutcome(raw_stream=mock_stream())
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        ir = _make_ir(streaming_request)
        result = await _route_and_respond(ir, streaming_request)

        # Should return EventSourceResponse
        assert isinstance(result, EventSourceResponse)


class TestRoutingFallback:
    """Tests for fallback behavior when routing fails."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat._route_and_respond")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_direct_inference")
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
        await _handle_text_request(basic_request, None)

        # Direct handler should be called as fallback
        mock_direct.assert_called_once()

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat._route_and_respond")
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

        await _handle_text_request(basic_request, None)

        # Batching handler should be called as fallback
        mock_batched.assert_called_once()


class TestMessageConversion:
    """Tests for message format conversion via OpenAIFormatter.parse_request."""

    def test_simple_string_content(self):
        """String content is preserved as-is in IR."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        ir = _make_ir(request)
        assert ir.messages == [{"role": "user", "content": "Hello"}]

    def test_content_blocks_extract_text(self):
        """Content blocks are converted to text via parse_request."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                    ],
                )
            ],
        )
        ir = _make_ir(request)
        assert ir.messages[0]["role"] == "user"
        assert ir.messages[0]["content"] == "Describe this image"

    def test_none_content_preserved(self):
        """None content (for assistant messages with only tool_calls) is preserved."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionCall(
                                name="get_weather", arguments='{"location":"Tokyo"}'
                            ),
                        )
                    ],
                )
            ],
        )
        ir = _make_ir(request)
        assert ir.messages[0]["content"] is None
        assert "tool_calls" in ir.messages[0]
        assert ir.messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_call_id_preserved(self):
        """tool_call_id field is preserved in conversion."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(
                    role="tool",
                    content='{"temperature": 22}',
                    tool_call_id="call_1",
                )
            ],
        )
        ir = _make_ir(request)
        assert ir.messages[0]["tool_call_id"] == "call_1"

    def test_multi_message_conversation(self):
        """Multi-turn conversation is correctly converted."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello!"),
            ],
        )
        ir = _make_ir(request)
        assert len(ir.messages) == 3
        assert all(isinstance(m, dict) for m in ir.messages)
        assert ir.messages[0]["role"] == "system"
        assert ir.messages[2]["content"] == "Hello!"

    def test_stop_string_normalized_to_list(self):
        """Single stop string is normalized to a list in IR."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stop="END",
        )
        ir = _make_ir(request)
        assert ir.stop == ["END"]

    def test_stop_list_preserved(self):
        """Stop list is preserved in IR."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stop=["<end>", "<stop>"],
        )
        ir = _make_ir(request)
        assert ir.stop == ["<end>", "<stop>"]

    def test_tools_converted_when_tool_choice_not_none(self):
        """Tools are converted to dicts in IR when tool_choice is not 'none'."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Weather?")],
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
        ir = _make_ir(request)
        assert ir.tools is not None
        assert len(ir.tools) == 1

    def test_tools_excluded_when_tool_choice_is_none(self):
        """Tools are NOT included in IR when tool_choice is 'none'."""
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
        ir = _make_ir(request)
        assert ir.tools is None


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
# Tests for _handle_direct_inference
# ============================================================================


class TestHandleDirectInference:
    """Tests for the direct inference path."""

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_complete_response")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_non_streaming_basic(self, mock_settings, mock_generate):
        """Non-streaming direct request returns ChatCompletionResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = _make_inference_result(
            content="Hello there!", prompt_tokens=5, completion_tokens=3
        )

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=False,
        )
        ir = _make_ir(request)

        result = await _handle_direct_inference(ir, request)
        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].message.content == "Hello there!"
        assert result.usage.total_tokens == 8

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_stream")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_returns_event_source(self, mock_settings, mock_stream):
        """Streaming direct request returns EventSourceResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_ir_stream():
            yield StreamEvent(type="content", content="Hi")
            yield TextResult(content="Hi", finish_reason="stop")

        mock_stream.return_value = mock_ir_stream()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        )
        ir = _make_ir(request)

        result = await _handle_direct_inference(ir, request)
        assert isinstance(result, EventSourceResponse)

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_complete_response")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_stop_string_converted_to_list(self, mock_settings, mock_generate):
        """Single stop string is converted to a list via IR."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = _make_inference_result()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stop="END",
        )
        ir = _make_ir(request)

        await _handle_direct_inference(ir, request)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["stop"] == ["END"]

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_complete_response")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_stop_list_passed_through(self, mock_settings, mock_generate):
        """Stop list is passed through unchanged via IR."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = _make_inference_result()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stop=["<end>", "<stop>"],
        )
        ir = _make_ir(request)

        await _handle_direct_inference(ir, request)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["stop"] == ["<end>", "<stop>"]

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_complete_response")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_tools_passed_when_tool_choice_not_none(self, mock_settings, mock_generate):
        """Tools are passed to inference when tool_choice is not 'none'."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = _make_inference_result(
            content="",
            finish_reason="tool_calls",
            prompt_tokens=20,
            completion_tokens=10,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Tokyo"}',
                    },
                }
            ],
        )

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
        ir = _make_ir(request)

        result = await _handle_direct_inference(ir, request)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 1

        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].message.tool_calls is not None
        assert result.choices[0].message.tool_calls[0].function.name == "get_weather"

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_complete_response")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_tools_excluded_when_tool_choice_is_none(self, mock_settings, mock_generate):
        """Tools are NOT passed when tool_choice is 'none'."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = _make_inference_result(content="No tools used.")

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
        ir = _make_ir(request)

        await _handle_direct_inference(ir, request)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["tools"] is None

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_complete_response")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_reasoning_content_preserved(self, mock_settings, mock_generate):
        """reasoning_content from the service is included in the response."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = _make_inference_result(
            content="42",
            reasoning_content="Let me think step by step...",
            prompt_tokens=10,
            completion_tokens=5,
        )

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="What is 6*7?")],
        )
        ir = _make_ir(request)

        result = await _handle_direct_inference(ir, request)
        assert result.choices[0].message.reasoning_content == "Let me think step by step..."


# ============================================================================
# Tests for _handle_non_streaming (structured output validation)
# ============================================================================


class TestNonStreamingStructuredOutput:
    """Tests for structured output validation in non-streaming path."""

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_complete_response")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_structured_output_valid_json(self, mock_settings, mock_generate):
        """Valid JSON matching schema passes validation."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = _make_inference_result(
            content='{"name":"Alice","age":30}', prompt_tokens=10, completion_tokens=5
        )

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
        ir = _make_ir(request)

        result = await _handle_non_streaming(ir, request)
        assert result.choices[0].message.content == '{"name":"Alice","age":30}'

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_complete_response")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_structured_output_invalid_json_raises_400(self, mock_settings, mock_generate):
        """Invalid JSON that doesn't match schema raises HTTPException 400."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = _make_inference_result(
            content="not valid json at all", prompt_tokens=10, completion_tokens=5
        )

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
        ir = _make_ir(request)

        with pytest.raises(HTTPException) as exc_info:
            await _handle_non_streaming(ir, request)
        assert exc_info.value.status_code == 400
        assert "JSON schema validation" in exc_info.value.detail

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_complete_response")
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
        ir = _make_ir(request)

        with pytest.raises(HTTPException) as exc_info:
            await _handle_non_streaming(ir, request)
        assert exc_info.value.status_code == 408


# ============================================================================
# Tests for _handle_streaming (timeout)
# ============================================================================


class TestHandleStreaming:
    """Tests for the streaming handler."""

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_stream")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_returns_sse(self, mock_settings, mock_stream):
        """Streaming handler returns EventSourceResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_ir_stream():
            yield StreamEvent(type="content", content="hello")
            yield TextResult(content="hello", finish_reason="stop")

        mock_stream.return_value = mock_ir_stream()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=True,
        )
        ir = _make_ir(request)

        result = await _handle_streaming(ir, request)
        assert isinstance(result, EventSourceResponse)

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_stream")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_with_tools(self, mock_settings, mock_stream):
        """Streaming with tools returns an EventSourceResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_ir_stream():
            yield StreamEvent(type="content", content="I'll check")
            yield TextResult(content="I'll check", finish_reason="stop")

        mock_stream.return_value = mock_ir_stream()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Weather?")],
            stream=True,
            tools=[
                Tool(
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get weather",
                    )
                )
            ],
        )
        ir = _make_ir(request)

        result = await _handle_streaming(ir, request)
        # The generator is lazy; it returns EventSourceResponse immediately
        assert isinstance(result, EventSourceResponse)


# ============================================================================
# Tests for unified vision handling (now through _handle_text_request)
# ============================================================================


class TestUnifiedVisionHandling:
    """Tests for vision request handling through the unified path."""

    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    async def test_images_with_non_vision_model_raises_400(self, mock_detect):
        """Images sent to a non-vision model raises 400 during validation."""
        mock_detect.return_value = ModelType.TEXT_GEN

        request = ChatCompletionRequest(
            model="text-only-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                    ],
                )
            ],
        )

        # The validation happens in create_chat_completion before routing
        with pytest.raises(HTTPException) as exc_info:
            await create_chat_completion(request)
        assert exc_info.value.status_code == 400
        assert "vision model" in exc_info.value.detail.lower()


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

        # No images, so detect_model_type won't be called
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
        # Should be called with request and empty image_urls list
        mock_text_handler.assert_called_once()
        assert result is response

    @patch("mlx_manager.mlx_server.api.v1.chat._handle_text_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.detect_model_type")
    @patch("mlx_manager.mlx_server.api.v1.chat.audit_service")
    async def test_images_route_to_unified_handler(
        self, mock_audit, mock_detect, mock_text_handler
    ):
        """Request with images routes to _handle_text_request (unified path)."""
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
        mock_text_handler.return_value = response

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
        # Should route through unified text handler with image URLs
        mock_text_handler.assert_called_once()
        assert result is response

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
# Tests for _route_and_respond timeout
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
        ir = _make_ir(request)

        with pytest.raises(HTTPException) as exc_info:
            await _route_and_respond(ir, request)
        assert exc_info.value.status_code == 408


# ============================================================================
# Tests for batching fallback paths
# ============================================================================


class TestBatchingFallback:
    """Tests for batching scheduler initialization fallback."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_scheduler_manager")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_direct_inference")
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

        await _handle_text_request(request, None)
        mock_direct.assert_called_once()

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_scheduler_manager")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_direct_inference")
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

        await _handle_text_request(request, None)
        mock_direct.assert_called_once()


# ============================================================================
# Tests that consume SSE generators to cover inner generator bodies
# ============================================================================


class TestStreamingGeneratorConsumption:
    """Tests that actually iterate SSE generators to cover the inner bodies."""

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_stream")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_generator_yields_chunks_and_done(self, mock_settings, mock_stream):
        """Iterating the SSE generator yields data chunks followed by [DONE]."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_ir_stream():
            yield StreamEvent(type="content", content="Hi")
            yield StreamEvent(type="content", content="!")
            yield TextResult(content="Hi!", finish_reason="stop")

        mock_stream.return_value = mock_ir_stream()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        )
        ir = _make_ir(request)

        result = await _handle_streaming(ir, request)
        # Consume the body iterator
        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "DONE" in event_text

    @patch("mlx_manager.mlx_server.api.v1.chat.generate_chat_stream")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_streaming_generator_timeout_yields_error_event(self, mock_settings, mock_stream):
        """When generate_chat_stream times out, the generator yields an error event."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 0.001
        mock_settings.return_value = settings

        mock_stream.side_effect = TimeoutError()

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        )
        ir = _make_ir(request)

        result = await _handle_streaming(ir, request)
        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "error" in event_text.lower() or "timeout" in event_text.lower()

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_routed_streaming_generator_yields_chunks(self, mock_settings, mock_get_router):
        """Routed streaming generator yields data chunks and [DONE]."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_stream():
            yield {"id": "r1", "choices": [{"delta": {"content": "Routed"}}]}

        outcome = RoutingOutcome(raw_stream=mock_stream())
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=True,
        )
        ir = _make_ir(request)

        result = await _route_and_respond(ir, request)
        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "DONE" in event_text


# ============================================================================
# Tests for _route_and_respond with IR results
# ============================================================================


class TestRouteAndRespondIR:
    """Tests for _route_and_respond handling IR results from router."""

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_ir_result_formatted_as_response(self, mock_settings, mock_get_router):
        """IR non-streaming result is formatted via _format_ir_complete."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        inference_result = _make_inference_result(content="Hello from IR!")
        outcome = RoutingOutcome(ir_result=inference_result)
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        ir = _make_ir(request)

        result = await _route_and_respond(ir, request)
        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].message.content == "Hello from IR!"

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_ir_stream_formatted_as_sse(self, mock_settings, mock_get_router):
        """IR streaming result is formatted as EventSourceResponse."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_ir_gen():
            yield StreamEvent(type="content", content="streamed")
            yield TextResult(content="streamed", finish_reason="stop")

        outcome = RoutingOutcome(ir_stream=mock_ir_gen())
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=True,
        )
        ir = _make_ir(request)

        result = await _route_and_respond(ir, request)
        assert isinstance(result, EventSourceResponse)

    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    async def test_empty_outcome_raises_runtime_error(self, mock_settings, mock_get_router):
        """RoutingOutcome with no result raises RuntimeError."""
        settings = MagicMock()
        settings.timeout_chat_seconds = 900.0
        mock_settings.return_value = settings

        outcome = RoutingOutcome()  # All fields None
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        ir = _make_ir(request)

        with pytest.raises(RuntimeError, match="RoutingOutcome has no result"):
            await _route_and_respond(ir, request)


# ============================================================================
# Tests for _format_ir_complete
# ============================================================================


class TestFormatIRComplete:
    """Tests for the _format_ir_complete helper."""

    def test_basic_formatting(self):
        """Basic InferenceResult is formatted as ChatCompletionResponse."""
        inference_result = _make_inference_result(
            content="Hello!", prompt_tokens=5, completion_tokens=2
        )
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )

        result = _format_ir_complete(inference_result, request)
        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].message.content == "Hello!"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 2
        assert result.usage.total_tokens == 7

    def test_tool_calls_formatting(self):
        """InferenceResult with tool calls is formatted correctly."""
        inference_result = _make_inference_result(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"loc":"NYC"}'},
                }
            ],
        )
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Weather?")],
        )

        result = _format_ir_complete(inference_result, request)
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None
        assert result.choices[0].message.tool_calls[0].function.name == "get_weather"

    def test_reasoning_content_formatting(self):
        """InferenceResult with reasoning_content is formatted correctly."""
        inference_result = _make_inference_result(
            content="42",
            reasoning_content="Let me think...",
        )
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="6*7?")],
        )

        result = _format_ir_complete(inference_result, request)
        assert result.choices[0].message.reasoning_content == "Let me think..."


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

        with patch("mlx_manager.mlx_server.models.pool.get_model_pool") as mock_pool_fn:
            loaded = MagicMock()
            loaded.model = MagicMock()
            loaded.tokenizer = MagicMock()
            loaded.tokenizer.tokenizer = MagicMock()
            loaded.tokenizer.tokenizer.encode.return_value = [1, 2, 3]
            adapter = MagicMock()
            adapter.apply_chat_template.return_value = "Hello"
            loaded.adapter = adapter
            pool = AsyncMock()
            pool.get_model = AsyncMock(return_value=loaded)
            mock_pool_fn.return_value = pool

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
            ir = _make_ir(request)

            await _handle_batched_request(ir, request, Priority.NORMAL)
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

        with patch("mlx_manager.mlx_server.models.pool.get_model_pool") as mock_pool_fn:
            loaded = MagicMock()
            loaded.model = MagicMock()
            loaded.tokenizer = MagicMock()
            loaded.tokenizer.tokenizer = MagicMock()
            loaded.tokenizer.tokenizer.encode.return_value = [1, 2, 3]
            adapter = MagicMock()
            adapter.apply_chat_template.return_value = "Hello"
            loaded.adapter = adapter
            pool = AsyncMock()
            pool.get_model = AsyncMock(return_value=loaded)
            mock_pool_fn.return_value = pool

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
            ir = _make_ir(request)

            await _handle_batched_request(ir, request, Priority.NORMAL)
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


class TestMessageContentNoneSkipped:
    """Tests for message content=None handling in create_chat_completion."""

    async def test_message_content_none_skipped(self) -> None:
        """Messages with content=None are skipped during image extraction."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content=None),
                ChatMessage(role="user", content="Hello"),
            ],
        )

        with (
            patch(
                "mlx_manager.mlx_server.api.v1.chat._handle_text_request",
                new_callable=AsyncMock,
            ) as mock_handler,
            patch("mlx_manager.mlx_server.api.v1.chat.audit_service") as mock_audit,
        ):
            mock_handler.return_value = ChatCompletionResponse(
                id="test",
                created=1,
                model="test-model",
                choices=[],
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
            mock_audit.track_request.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock()
            )
            mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await create_chat_completion(request)
            assert isinstance(result, ChatCompletionResponse)
            mock_handler.assert_called_once()


class TestImagePreprocessingPath:
    """Tests for the image preprocessing path in _handle_text_request."""

    async def test_image_urls_preprocessed_into_ir(self) -> None:
        """Image URLs trigger preprocess_images and update IR."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Describe this")],
        )
        _usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        mock_response = ChatCompletionResponse(
            id="test", created=1, model="test-model", choices=[], usage=_usage
        )

        with (
            patch("mlx_manager.mlx_server.api.v1.chat.get_settings") as mock_settings,
            patch(
                "mlx_manager.mlx_server.api.v1.chat.preprocess_images",
                new_callable=AsyncMock,
            ) as mock_preprocess,
            patch(
                "mlx_manager.mlx_server.api.v1.chat._handle_direct_inference",
                new_callable=AsyncMock,
            ) as mock_direct,
        ):
            settings = MagicMock()
            settings.enable_cloud_routing = False
            settings.enable_batching = False
            mock_settings.return_value = settings

            mock_preprocess.return_value = ["data:image/png;base64,abc"]
            mock_direct.return_value = mock_response

            result = await _handle_text_request(
                request, image_urls=["http://example.com/img.png"]
            )

            assert result == mock_response
            mock_preprocess.assert_called_once_with(["http://example.com/img.png"])
            # Vision path goes directly to _handle_direct_inference
            mock_direct.assert_called_once()
            ir_arg = mock_direct.call_args[0][0]
            assert ir_arg.images == ["data:image/png;base64,abc"]

    async def test_vision_request_bypasses_routing(self) -> None:
        """Vision requests go straight to direct inference, skipping cloud routing."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Describe")],
        )
        _usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        mock_response = ChatCompletionResponse(
            id="test", created=1, model="test-model", choices=[], usage=_usage
        )

        with (
            patch("mlx_manager.mlx_server.api.v1.chat.get_settings") as mock_settings,
            patch(
                "mlx_manager.mlx_server.api.v1.chat.preprocess_images",
                new_callable=AsyncMock,
            ) as mock_preprocess,
            patch(
                "mlx_manager.mlx_server.api.v1.chat._handle_direct_inference",
                new_callable=AsyncMock,
            ) as mock_direct,
            patch(
                "mlx_manager.mlx_server.api.v1.chat._route_and_respond",
                new_callable=AsyncMock,
            ) as mock_route,
        ):
            settings = MagicMock()
            settings.enable_cloud_routing = True
            settings.enable_batching = False
            mock_settings.return_value = settings
            mock_preprocess.return_value = ["data:image/png;base64,abc"]
            mock_direct.return_value = mock_response

            await _handle_text_request(request, image_urls=["http://example.com/img.png"])

            # Direct inference called, routing NOT called
            mock_direct.assert_called_once()
            mock_route.assert_not_called()


class TestFormatIRStreamAsSSE:
    """Tests for _format_ir_stream_as_sse consuming IR stream events."""

    async def test_format_ir_stream_produces_events(self) -> None:
        """_format_ir_stream_as_sse consumes stream events and produces SSE."""

        async def mock_ir_stream():
            yield StreamEvent(type="content", content="Hello")
            yield StreamEvent(type="content", content=" world")
            yield TextResult(content="Hello world", finish_reason="stop")

        response = _format_ir_stream_as_sse(mock_ir_stream(), "test-model")
        assert isinstance(response, EventSourceResponse)

        # Consume the body_iterator
        events = []
        async for event in response.body_iterator:
            events.append(str(event))

        combined = "".join(events)
        # Should contain delta content events and a final stop event
        assert "Hello" in combined
        assert "world" in combined
        assert "stop" in combined

    async def test_format_ir_stream_with_tool_calls(self) -> None:
        """_format_ir_stream_as_sse handles tool calls in TextResult."""

        async def mock_ir_stream():
            yield StreamEvent(type="content", content="Let me look that up")
            yield TextResult(
                content="Let me look that up",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            )

        response = _format_ir_stream_as_sse(mock_ir_stream(), "test-model")
        events = []
        async for event in response.body_iterator:
            events.append(str(event))

        combined = "".join(events)
        assert "tool_calls" in combined
