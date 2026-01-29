"""Tests for chat endpoint routing integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.api.v1.chat import (
    _handle_routed_request,
    _handle_text_request,
    router,
)
from mlx_manager.mlx_server.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
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
    async def test_routing_disabled_goes_to_direct(
        self, mock_direct, mock_settings, basic_request
    ):
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
        mock_settings.return_value = settings

        # Setup router mock
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=mock_completion_response)
        mock_get_router.return_value = router_mock

        result = await _handle_text_request(basic_request)

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
    async def test_routed_request_streaming(
        self, mock_get_router, streaming_request
    ):
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
    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_direct_request")
    async def test_routing_failure_falls_back_to_direct(
        self, mock_direct, mock_get_router, mock_settings, basic_request
    ):
        """When routing fails, fall back to direct inference."""
        settings = MagicMock()
        settings.enable_cloud_routing = True
        settings.enable_batching = False
        mock_settings.return_value = settings

        # Make router fail
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(
            side_effect=Exception("Router unavailable")
        )
        mock_get_router.return_value = router_mock

        mock_direct.return_value = MagicMock()

        # Should not raise, should fall back
        await _handle_text_request(basic_request)

        # Direct handler should be called as fallback
        mock_direct.assert_called_once_with(basic_request)

    @patch("mlx_manager.mlx_server.api.v1.chat.get_settings")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_router")
    @patch("mlx_manager.mlx_server.api.v1.chat._handle_batched_request")
    @patch("mlx_manager.mlx_server.api.v1.chat.get_scheduler_manager")
    async def test_routing_failure_falls_back_to_batching(
        self, mock_scheduler_mgr, mock_batched, mock_get_router, mock_settings, basic_request
    ):
        """When routing fails and batching is enabled, fall back to batching."""
        settings = MagicMock()
        settings.enable_cloud_routing = True
        settings.enable_batching = True
        mock_settings.return_value = settings

        # Make router fail
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(
            side_effect=Exception("Router unavailable")
        )
        mock_get_router.return_value = router_mock

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
