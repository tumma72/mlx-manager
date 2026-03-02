"""Tests for Anthropic messages endpoint routing integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.api.v1.messages import (
    _format_ir_complete,
    _format_ir_stream,
    _route_and_respond,
    create_message,
)
from mlx_manager.mlx_server.models.ir import (
    InferenceResult,
    InternalRequest,
    RoutingOutcome,
    StreamEvent,
    TextResult,
)
from mlx_manager.mlx_server.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    MessageParam,
)
from mlx_manager.models.enums import ApiType
from mlx_manager.models.value_objects import InferenceParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ir(model: str = "test-model", stream: bool = False) -> InternalRequest:
    """Build a minimal InternalRequest for Anthropic tests."""
    return InternalRequest(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        params=InferenceParams(max_tokens=1000, temperature=1.0),
        stream=stream,
        original_protocol=ApiType.ANTHROPIC,
    )


def _make_inference_result(
    content: str = "Hello!",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls: list[dict] | None = None,
) -> InferenceResult:
    """Build an InferenceResult for use in formatter tests."""
    return InferenceResult(
        result=TextResult(
            content=content,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _make_request(model: str = "test-model", stream: bool = False) -> AnthropicMessagesRequest:
    """Build a minimal AnthropicMessagesRequest."""
    return AnthropicMessagesRequest(
        model=model,
        max_tokens=1000,
        messages=[MessageParam(role="user", content="Hello")],
        stream=stream,
    )


def _raw_anthropic_response(
    model: str = "test-model",
    content: str = "Hello!",
) -> dict:
    """Build a raw Anthropic-format response dict for passthrough tests."""
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


# ---------------------------------------------------------------------------
# TestRouteAndRespond
# ---------------------------------------------------------------------------


class TestRouteAndRespond:
    """Tests for _route_and_respond: all four RoutingOutcome variants."""

    @patch("mlx_manager.mlx_server.api.v1.messages.get_router")
    async def test_passthrough_non_streaming_raw_dict(self, mock_get_router):
        """raw_response dict is validated and returned as AnthropicMessagesResponse."""
        raw = _raw_anthropic_response()
        outcome = RoutingOutcome(raw_response=raw)
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        ir = _make_ir()
        request = _make_request()

        result = await _route_and_respond(ir, request)

        assert isinstance(result, AnthropicMessagesResponse)
        assert result.id == "msg_test123"
        assert result.model == "test-model"
        assert len(result.content) == 1
        assert result.content[0].text == "Hello!"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    @patch("mlx_manager.mlx_server.api.v1.messages.get_router")
    async def test_passthrough_non_streaming_raw_model_instance(self, mock_get_router):
        """raw_response that is already an AnthropicMessagesResponse is returned directly."""
        raw = _raw_anthropic_response()
        response_model = AnthropicMessagesResponse.model_validate(raw)
        outcome = RoutingOutcome(raw_response=response_model)
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        ir = _make_ir()
        request = _make_request()

        result = await _route_and_respond(ir, request)

        assert result is response_model

    @patch("mlx_manager.mlx_server.api.v1.messages.get_router")
    async def test_passthrough_streaming_raw_stream(self, mock_get_router):
        """raw_stream async generator is wrapped as EventSourceResponse."""

        async def mock_stream():
            yield {"event": "message_start", "data": json.dumps({"type": "message_start"})}
            yield {"event": "message_stop", "data": json.dumps({"type": "message_stop"})}

        outcome = RoutingOutcome(raw_stream=mock_stream())
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        ir = _make_ir(stream=True)
        request = _make_request(stream=True)

        result = await _route_and_respond(ir, request)

        assert isinstance(result, EventSourceResponse)

    @patch("mlx_manager.mlx_server.api.v1.messages.get_router")
    async def test_ir_non_streaming_ir_result(self, mock_get_router):
        """ir_result is formatted via AnthropicFormatter."""
        inference_result = _make_inference_result(
            content="IR response", prompt_tokens=8, completion_tokens=4
        )
        outcome = RoutingOutcome(ir_result=inference_result)
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        ir = _make_ir()
        request = _make_request()

        result = await _route_and_respond(ir, request)

        assert isinstance(result, AnthropicMessagesResponse)
        assert result.model == "test-model"
        # Content comes from IR result
        assert any(
            block.text == "IR response" for block in result.content if hasattr(block, "text")
        )
        assert result.usage.input_tokens == 8
        assert result.usage.output_tokens == 4

    @patch("mlx_manager.mlx_server.api.v1.messages.get_router")
    async def test_ir_streaming_ir_stream(self, mock_get_router):
        """ir_stream async generator is formatted via AnthropicFormatter as EventSourceResponse."""

        async def ir_stream():
            yield StreamEvent(type="content", content="Hello")
            yield TextResult(content="Hello", finish_reason="stop")

        outcome = RoutingOutcome(ir_stream=ir_stream())
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        ir = _make_ir(stream=True)
        request = _make_request(stream=True)

        result = await _route_and_respond(ir, request)

        assert isinstance(result, EventSourceResponse)

    @patch("mlx_manager.mlx_server.api.v1.messages.get_router")
    async def test_empty_outcome_raises_runtime_error(self, mock_get_router):
        """An empty RoutingOutcome (all None) raises RuntimeError."""
        outcome = RoutingOutcome()  # All fields None
        router_mock = AsyncMock()
        router_mock.route_request = AsyncMock(return_value=outcome)
        mock_get_router.return_value = router_mock

        ir = _make_ir()
        request = _make_request()

        with pytest.raises(RuntimeError, match="RoutingOutcome has no result"):
            await _route_and_respond(ir, request)


# ---------------------------------------------------------------------------
# TestFormatIRComplete
# ---------------------------------------------------------------------------


class TestFormatIRComplete:
    """Tests for _format_ir_complete: InferenceResult → AnthropicMessagesResponse."""

    def test_basic_formatting(self):
        """Basic text result is formatted with correct model, content, stop_reason, usage."""
        inference_result = _make_inference_result(
            content="Hello!", finish_reason="stop", prompt_tokens=10, completion_tokens=5
        )
        request = _make_request(model="claude-test")

        result = _format_ir_complete(inference_result, request)

        assert isinstance(result, AnthropicMessagesResponse)
        assert result.model == "claude-test"
        # Text block must contain our content
        text_blocks = [b for b in result.content if hasattr(b, "text")]
        assert any(b.text == "Hello!" for b in text_blocks)
        # stop → end_turn via openai_stop_to_anthropic
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_tool_calls_included_in_response(self):
        """Tool calls from InferenceResult appear as ToolUseBlock in content."""
        inference_result = _make_inference_result(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "toolu_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
                }
            ],
        )
        request = _make_request()

        result = _format_ir_complete(inference_result, request)

        assert isinstance(result, AnthropicMessagesResponse)
        # stop_reason should be "tool_use" for tool calls
        assert result.stop_reason == "tool_use"
        # Find ToolUseBlock in content
        tool_blocks = [b for b in result.content if hasattr(b, "name")]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "get_weather"
        assert tool_blocks[0].id == "toolu_abc123"
        assert tool_blocks[0].input == {"location": "Tokyo"}

    def test_max_tokens_finish_reason_maps_correctly(self):
        """finish_reason 'length' maps to Anthropic 'max_tokens' stop_reason."""
        inference_result = _make_inference_result(content="Truncated...", finish_reason="length")
        request = _make_request()

        result = _format_ir_complete(inference_result, request)

        assert result.stop_reason == "max_tokens"


# ---------------------------------------------------------------------------
# TestFormatIRStream
# ---------------------------------------------------------------------------


class TestFormatIRStream:
    """Tests for _format_ir_stream: async IR generator → EventSourceResponse."""

    def test_returns_event_source_response(self):
        """_format_ir_stream always returns an EventSourceResponse (lazy evaluation)."""

        async def ir_stream():
            yield StreamEvent(type="content", content="Hi")
            yield TextResult(content="Hi", finish_reason="stop")

        request = _make_request()
        result = _format_ir_stream(ir_stream(), request)

        assert isinstance(result, EventSourceResponse)

    async def test_stream_events_include_anthropic_event_types(self):
        """Consuming the EventSourceResponse generator emits all required Anthropic event types."""

        async def ir_stream():
            yield StreamEvent(type="content", content="Hello")
            yield StreamEvent(type="content", content=" world")
            yield TextResult(content="Hello world", finish_reason="stop")

        request = _make_request()
        result = _format_ir_stream(ir_stream(), request)

        assert isinstance(result, EventSourceResponse)

        # Collect all event types from the body_iterator
        event_types: list[str] = []
        async for chunk in result.body_iterator:
            if isinstance(chunk, dict) and "event" in chunk:
                event_types.append(chunk["event"])
            elif isinstance(chunk, bytes):
                text = chunk.decode()
                for line in text.splitlines():
                    if line.startswith("event:"):
                        event_types.append(line.split(":", 1)[1].strip())
            elif isinstance(chunk, str):
                for line in chunk.splitlines():
                    if line.startswith("event:"):
                        event_types.append(line.split(":", 1)[1].strip())

        # The Anthropic event sequence must include these event types
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types


# ---------------------------------------------------------------------------
# TestCreateMessageRouting
# ---------------------------------------------------------------------------


class TestCreateMessageRouting:
    """Tests for create_message endpoint: routing integration and fallback."""

    @patch("mlx_manager.mlx_server.api.v1.messages._route_and_respond")
    async def test_routing_enabled_routing_succeeds(self, mock_route):
        """When routing succeeds, result is returned."""
        raw = _raw_anthropic_response()
        expected_response = AnthropicMessagesResponse.model_validate(raw)
        mock_route.return_value = expected_response

        request = _make_request()

        with patch("mlx_manager.mlx_server.api.v1.messages.audit_service") as mock_audit:
            mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await create_message(request)

        mock_route.assert_called_once()
        assert result is expected_response

    @patch("mlx_manager.mlx_server.api.v1.messages._route_and_respond")
    @patch("mlx_manager.mlx_server.api.v1.messages._handle_non_streaming")
    async def test_routing_enabled_routing_fails_falls_back_to_local(self, mock_local, mock_route):
        """When routing throws, local inference is used as fallback."""
        mock_route.side_effect = RuntimeError("Cloud unavailable")

        raw = _raw_anthropic_response()
        expected_response = AnthropicMessagesResponse.model_validate(raw)
        mock_local.return_value = expected_response

        request = _make_request()  # stream=False

        with patch("mlx_manager.mlx_server.api.v1.messages.audit_service") as mock_audit:
            mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
            mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await create_message(request)

        mock_route.assert_called_once()
        mock_local.assert_called_once()
        assert result is expected_response
