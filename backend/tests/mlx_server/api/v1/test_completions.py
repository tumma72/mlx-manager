"""Tests for legacy completions endpoint."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse

from mlx_manager.mlx_server.api.v1.completions import (
    _handle_non_streaming,
    _handle_streaming,
    create_completion,
)
from mlx_manager.mlx_server.schemas.openai import (
    CompletionRequest,
    CompletionResponse,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_request():
    """Create a basic completion request."""
    return CompletionRequest(
        model="test-model",
        prompt="Once upon a time",
    )


@pytest.fixture
def streaming_request():
    """Create a streaming completion request."""
    return CompletionRequest(
        model="test-model",
        prompt="Once upon a time",
        stream=True,
    )


@pytest.fixture
def mock_completion_result() -> dict[str, Any]:
    """Create a mock completion result dict from inference service."""
    return {
        "id": "cmpl-test123",
        "object": "text_completion",
        "created": 1700000000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "text": " there was a brave knight",
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 6,
            "total_tokens": 11,
        },
    }


# ============================================================================
# Tests for create_completion (top-level endpoint)
# ============================================================================


class TestCreateCompletion:
    """Tests for the create_completion endpoint function."""

    @patch("mlx_manager.mlx_server.api.v1.completions.audit_service")
    @patch("mlx_manager.mlx_server.api.v1.completions._handle_non_streaming")
    async def test_non_streaming_returns_response(self, mock_handler, mock_audit, basic_request):
        """Non-streaming request returns CompletionResponse."""
        from mlx_manager.mlx_server.schemas.openai import (
            CompletionChoice,
            Usage,
        )

        response = CompletionResponse(
            id="cmpl-test",
            created=1700000000,
            model="test-model",
            choices=[
                CompletionChoice(
                    index=0,
                    text="hello world",
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=5,
                completion_tokens=2,
                total_tokens=7,
            ),
        )
        mock_handler.return_value = response

        ctx = MagicMock()
        ctx.prompt_tokens = 0
        ctx.completion_tokens = 0
        ctx.total_tokens = 0
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await create_completion(basic_request)
        assert result is response
        assert ctx.prompt_tokens == 5
        assert ctx.completion_tokens == 2
        assert ctx.total_tokens == 7

    @patch("mlx_manager.mlx_server.api.v1.completions.audit_service")
    @patch("mlx_manager.mlx_server.api.v1.completions._handle_streaming")
    async def test_streaming_returns_event_source(
        self, mock_handler, mock_audit, streaming_request
    ):
        """Streaming request returns EventSourceResponse."""
        mock_handler.return_value = MagicMock(spec=EventSourceResponse)

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        await create_completion(streaming_request)
        mock_handler.assert_called_once()

    @patch("mlx_manager.mlx_server.api.v1.completions.audit_service")
    @patch("mlx_manager.mlx_server.api.v1.completions._handle_non_streaming")
    async def test_runtime_error_raises_500(self, mock_handler, mock_audit, basic_request):
        """RuntimeError during generation raises HTTPException 500."""
        mock_handler.side_effect = RuntimeError("Model not loaded")

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        with pytest.raises(HTTPException) as exc_info:
            await create_completion(basic_request)
        assert exc_info.value.status_code == 500
        assert "Model not loaded" in exc_info.value.detail

    @patch("mlx_manager.mlx_server.api.v1.completions.audit_service")
    @patch("mlx_manager.mlx_server.api.v1.completions._handle_non_streaming")
    async def test_unexpected_error_raises_generic_500(
        self, mock_handler, mock_audit, basic_request
    ):
        """Unexpected Exception raises generic 500 error."""
        mock_handler.side_effect = ValueError("Something unexpected")

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        with pytest.raises(HTTPException) as exc_info:
            await create_completion(basic_request)
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal server error"


class TestStopParameterHandling:
    """Tests for stop parameter handling in create_completion."""

    @patch("mlx_manager.mlx_server.api.v1.completions.audit_service")
    @patch("mlx_manager.mlx_server.api.v1.completions._handle_non_streaming")
    async def test_stop_string_converted_to_list(self, mock_handler, mock_audit):
        """Single stop string is wrapped in a list."""
        from mlx_manager.mlx_server.schemas.openai import (
            CompletionChoice,
            Usage,
        )

        response = CompletionResponse(
            id="cmpl-test",
            created=1700000000,
            model="test-model",
            choices=[CompletionChoice(index=0, text="output", finish_reason="stop")],
            usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        mock_handler.return_value = response

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            stop="END",
        )

        await create_completion(request)
        call_args = mock_handler.call_args
        # The stop param is passed through to the handler
        assert call_args[0][1] == ["END"]

    @patch("mlx_manager.mlx_server.api.v1.completions.audit_service")
    @patch("mlx_manager.mlx_server.api.v1.completions._handle_non_streaming")
    async def test_stop_list_passed_through(self, mock_handler, mock_audit):
        """Stop list is passed through unchanged."""
        from mlx_manager.mlx_server.schemas.openai import (
            CompletionChoice,
            Usage,
        )

        response = CompletionResponse(
            id="cmpl-test",
            created=1700000000,
            model="test-model",
            choices=[CompletionChoice(index=0, text="output", finish_reason="stop")],
            usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        mock_handler.return_value = response

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            stop=["<end>", "<stop>"],
        )

        await create_completion(request)
        call_args = mock_handler.call_args
        assert call_args[0][1] == ["<end>", "<stop>"]

    @patch("mlx_manager.mlx_server.api.v1.completions.audit_service")
    @patch("mlx_manager.mlx_server.api.v1.completions._handle_non_streaming")
    async def test_stop_none_passed_as_none(self, mock_handler, mock_audit):
        """No stop parameter results in None."""
        from mlx_manager.mlx_server.schemas.openai import (
            CompletionChoice,
            Usage,
        )

        response = CompletionResponse(
            id="cmpl-test",
            created=1700000000,
            model="test-model",
            choices=[CompletionChoice(index=0, text="output", finish_reason="stop")],
            usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        mock_handler.return_value = response

        ctx = MagicMock()
        mock_audit.track_request.return_value.__aenter__ = AsyncMock(return_value=ctx)
        mock_audit.track_request.return_value.__aexit__ = AsyncMock(return_value=False)

        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
        )

        await create_completion(request)
        call_args = mock_handler.call_args
        assert call_args[0][1] is None


# ============================================================================
# Tests for _handle_non_streaming
# ============================================================================


class TestHandleNonStreaming:
    """Tests for the non-streaming completion handler."""

    @patch("mlx_manager.mlx_server.api.v1.completions.generate_completion")
    @patch("mlx_manager.mlx_server.api.v1.completions.get_settings")
    async def test_basic_completion(self, mock_settings, mock_generate, mock_completion_result):
        """Basic non-streaming returns CompletionResponse."""
        settings = MagicMock()
        settings.timeout_completions_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = mock_completion_result

        request = CompletionRequest(
            model="test-model",
            prompt="Once upon a time",
        )

        result = await _handle_non_streaming(request, None)
        assert isinstance(result, CompletionResponse)
        assert result.id == "cmpl-test123"
        assert result.model == "test-model"
        assert result.choices[0].text == " there was a brave knight"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.total_tokens == 11

    @patch("mlx_manager.mlx_server.api.v1.completions.generate_completion")
    @patch("mlx_manager.mlx_server.api.v1.completions.get_settings")
    async def test_echo_parameter_passed(
        self, mock_settings, mock_generate, mock_completion_result
    ):
        """Echo parameter is passed through to generate_completion."""
        settings = MagicMock()
        settings.timeout_completions_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = mock_completion_result

        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            echo=True,
        )

        await _handle_non_streaming(request, None)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["echo"] is True

    @patch("mlx_manager.mlx_server.api.v1.completions.generate_completion")
    @patch("mlx_manager.mlx_server.api.v1.completions.get_settings")
    async def test_custom_parameters(self, mock_settings, mock_generate, mock_completion_result):
        """Custom max_tokens, temperature, top_p are passed through."""
        settings = MagicMock()
        settings.timeout_completions_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = mock_completion_result

        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        await _handle_non_streaming(request, ["<stop>"])
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["stop"] == ["<stop>"]
        assert call_kwargs["stream"] is False

    @patch("mlx_manager.mlx_server.api.v1.completions.generate_completion")
    @patch("mlx_manager.mlx_server.api.v1.completions.get_settings")
    async def test_default_max_tokens_is_16(
        self, mock_settings, mock_generate, mock_completion_result
    ):
        """Default max_tokens is 16 when not specified."""
        settings = MagicMock()
        settings.timeout_completions_seconds = 900.0
        mock_settings.return_value = settings

        mock_generate.return_value = mock_completion_result

        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
        )

        await _handle_non_streaming(request, None)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["max_tokens"] == 16

    @patch("mlx_manager.mlx_server.api.v1.completions.generate_completion")
    @patch("mlx_manager.mlx_server.api.v1.completions.get_settings")
    async def test_timeout_raises_408(self, mock_settings, mock_generate):
        """Timeout during completion raises 408."""
        settings = MagicMock()
        settings.timeout_completions_seconds = 0.001
        mock_settings.return_value = settings

        mock_generate.side_effect = TimeoutError()

        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
        )

        with pytest.raises(HTTPException) as exc_info:
            await _handle_non_streaming(request, None)
        assert exc_info.value.status_code == 408


# ============================================================================
# Tests for _handle_streaming
# ============================================================================


class TestHandleStreaming:
    """Tests for the streaming completion handler."""

    @patch("mlx_manager.mlx_server.api.v1.completions.generate_completion")
    @patch("mlx_manager.mlx_server.api.v1.completions.get_settings")
    async def test_streaming_returns_event_source(self, mock_settings, mock_generate):
        """Streaming request returns EventSourceResponse."""
        settings = MagicMock()
        settings.timeout_completions_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_stream():
            yield {"choices": [{"text": "hello"}]}

        mock_generate.return_value = mock_stream()

        request = CompletionRequest(
            model="test-model",
            prompt="Hi",
            stream=True,
        )

        result = await _handle_streaming(request, None)
        assert isinstance(result, EventSourceResponse)

    @patch("mlx_manager.mlx_server.api.v1.completions.generate_completion")
    @patch("mlx_manager.mlx_server.api.v1.completions.get_settings")
    async def test_streaming_generator_yields_chunks_and_done(self, mock_settings, mock_generate):
        """Iterating SSE generator yields data chunks and [DONE]."""
        settings = MagicMock()
        settings.timeout_completions_seconds = 900.0
        mock_settings.return_value = settings

        chunk1 = {"id": "c1", "choices": [{"text": "Hello"}]}
        chunk2 = {"id": "c1", "choices": [{"text": " world"}]}

        async def mock_stream():
            yield chunk1
            yield chunk2

        mock_generate.return_value = mock_stream()

        request = CompletionRequest(
            model="test-model",
            prompt="Hi",
            stream=True,
        )

        result = await _handle_streaming(request, None)

        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "DONE" in event_text

    @patch("mlx_manager.mlx_server.api.v1.completions.generate_completion")
    @patch("mlx_manager.mlx_server.api.v1.completions.get_settings")
    async def test_streaming_echo_passed_through(self, mock_settings, mock_generate):
        """Echo parameter is passed through in streaming mode."""
        settings = MagicMock()
        settings.timeout_completions_seconds = 900.0
        mock_settings.return_value = settings

        async def mock_stream():
            yield {"choices": [{"text": "echo"}]}

        mock_generate.return_value = mock_stream()

        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            stream=True,
            echo=True,
        )

        result = await _handle_streaming(request, None)
        # Must consume the generator to trigger the actual call
        async for _ in result.body_iterator:
            pass
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["echo"] is True
        assert call_kwargs["stream"] is True

    @patch("mlx_manager.mlx_server.api.v1.completions.generate_completion")
    @patch("mlx_manager.mlx_server.api.v1.completions.get_settings")
    async def test_streaming_timeout_yields_error(self, mock_settings, mock_generate):
        """Timeout during streaming yields error event."""
        settings = MagicMock()
        settings.timeout_completions_seconds = 0.001
        mock_settings.return_value = settings

        mock_generate.side_effect = TimeoutError()

        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            stream=True,
        )

        result = await _handle_streaming(request, None)

        events = []
        async for event in result.body_iterator:
            events.append(event)

        event_text = "".join(str(e) for e in events)
        assert "error" in event_text.lower() or "timeout" in event_text.lower()
