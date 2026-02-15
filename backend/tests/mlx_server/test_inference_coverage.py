"""Coverage tests for inference.py uncovered lines.

This file targets specific uncovered code paths identified by coverage analysis:
- Modern IR path (generate_chat_stream, generate_chat_complete_response)
- Vision legacy path (pixel_values handling)
- Logfire instrumentation
- Adapter fallback logic

Philosophy: Minimize mocking, use REAL adapters with fake tokenizers.
Mock ONLY at GPU/thread boundary (run_on_metal_thread, stream_from_metal_thread).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mlx_manager.mlx_server.models.ir import StreamEvent, TextResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_tokenizer() -> MagicMock:
    """Create a lightweight fake tokenizer for real adapters."""
    tok = MagicMock()
    tok.eos_token_id = 128009
    tok.unk_token_id = 0
    tok.convert_tokens_to_ids = MagicMock(return_value=128001)
    tok.encode = MagicMock(return_value=list(range(10)))
    tok.apply_chat_template = MagicMock(return_value="<formatted_prompt>")
    tok.tokenizer = tok  # For Processor-wrapped tokenizers
    return tok


def _make_loaded_model_with_real_adapter(
    family: str = "qwen",
) -> tuple[MagicMock, MagicMock, MagicMock, Any]:
    """Return (loaded, model, tokenizer, adapter) with REAL adapter."""
    from mlx_manager.mlx_server.models.adapters.composable import create_adapter

    tokenizer = _make_fake_tokenizer()
    model = MagicMock()
    adapter = create_adapter(family, tokenizer)

    loaded = MagicMock()
    loaded.model = model
    loaded.tokenizer = tokenizer
    loaded.adapter = adapter

    return loaded, model, tokenizer, adapter


# ---------------------------------------------------------------------------
# NEW IR-returning functions (generate_chat_stream, generate_chat_complete_response)
# ---------------------------------------------------------------------------


class TestModernIRPath:
    """Test modern IR-returning functions with ctx.messages."""

    async def test_generate_chat_stream_yields_ir_events(self) -> None:
        """generate_chat_stream yields StreamEvent and TextResult."""
        from mlx_manager.mlx_server.services.inference import generate_chat_stream

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        # Mock adapter.generate_step to yield IR events
        async def fake_generate_step(**kwargs):
            yield StreamEvent(type="content", content="Hello")
            yield StreamEvent(type="content", content=" world")
            yield TextResult(content="Hello world", finish_reason="stop")

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch.object(adapter, "generate_step", side_effect=fake_generate_step),
        ):
            events = []
            async for event in await generate_chat_stream(
                model_id="test/model",
                messages=[{"role": "user", "content": "Hi"}],
            ):
                events.append(event)

        # Should have 2 StreamEvents + 1 TextResult
        assert len(events) == 3
        assert isinstance(events[0], StreamEvent)
        assert events[0].content == "Hello"
        assert isinstance(events[1], StreamEvent)
        assert events[1].content == " world"
        assert isinstance(events[2], TextResult)
        assert events[2].content == "Hello world"
        assert events[2].finish_reason == "stop"

    async def test_generate_chat_stream_with_tools(self) -> None:
        """generate_chat_stream with tools detected in final result."""
        from mlx_manager.mlx_server.services.inference import generate_chat_stream

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")
        # Set model_id to prevent config loading
        adapter._model_id = "test/model"

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        # tool_calls are dicts, not ToolCall objects
        tool_call_dict = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "test"}'},
        }

        async def fake_generate_step(**kwargs):
            yield StreamEvent(type="content", content="I'll search")
            yield TextResult(
                content="I'll search", finish_reason="tool_calls", tool_calls=[tool_call_dict]
            )

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch.object(adapter, "generate_step", side_effect=fake_generate_step),
        ):
            events = []
            async for event in await generate_chat_stream(
                model_id="test/model",
                messages=[{"role": "user", "content": "Search"}],
                tools=[{"type": "function", "function": {"name": "search"}}],
            ):
                events.append(event)

        # Final event should be TextResult with tool_calls (dict not ToolCall object)
        final = events[-1]
        assert isinstance(final, TextResult)
        assert final.finish_reason == "tool_calls"
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0]["function"]["name"] == "search"

    async def test_generate_chat_complete_response_returns_inference_result(self) -> None:
        """generate_chat_complete_response returns InferenceResult with TextResult."""
        from mlx_manager.mlx_server.services.inference import generate_chat_complete_response

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        # Mock adapter.generate to return TextResult
        async def fake_generate(**kwargs):
            return TextResult(content="The answer is 42", finish_reason="stop")

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch.object(adapter, "generate", side_effect=fake_generate),
        ):
            result = await generate_chat_complete_response(
                model_id="test/model",
                messages=[{"role": "user", "content": "What is the answer?"}],
            )

        # Should return InferenceResult
        assert result.result.content == "The answer is 42"
        assert result.result.finish_reason == "stop"
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0

    async def test_generate_chat_complete_response_with_vision(self) -> None:
        """generate_chat_complete_response with images uses word count for tokens."""
        from mlx_manager.mlx_server.services.inference import generate_chat_complete_response

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")
        adapter._model_id = "test/model"

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        # Mock adapter.generate with vision response
        async def fake_generate(**kwargs):
            return TextResult(content="This is a picture of a cat", finish_reason="stop")

        # Mock load_config to prevent HF calls
        mock_config = {"model_type": "qwen2_vl"}

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch.object(adapter, "generate", side_effect=fake_generate),
            patch("mlx_vlm.utils.load_config", return_value=mock_config),
        ):
            result = await generate_chat_complete_response(
                model_id="test/model",
                messages=[{"role": "user", "content": "What do you see?"}],
                images=["fake_image_data"],  # Triggers word count path
            )

        # Should use word count instead of tokenizer for completion_tokens
        assert result.result.content == "This is a picture of a cat"
        # Word count: 7 words
        assert result.completion_tokens == 7


# ---------------------------------------------------------------------------
# Vision legacy path (pixel_values)
# ---------------------------------------------------------------------------


class TestVisionLegacyPath:
    """Test legacy vision path with pixel_values in streaming and complete."""

    async def test_stream_chat_ir_vision_legacy(self) -> None:
        """_stream_chat_ir with pixel_values uses vlm_generate in legacy path."""
        from mlx_manager.mlx_server.services.inference import _GenContext, _stream_chat_ir

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")

        # Create context with pixel_values (legacy vision path)
        ctx = _GenContext(
            model=model,
            tokenizer=tokenizer,
            prompt="Describe this image",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop_token_ids={128009},
            adapter=adapter,
            model_id="test/model",
            completion_id="chatcmpl-vision",
            created=1700000000,
            tools=None,
            pixel_values=MagicMock(),  # Trigger vision path
            messages=None,  # No messages = legacy path
        )

        # Mock run_on_metal_thread to return vision response
        async def mock_vision_run(fn, **kwargs):
            return "A cat sitting on a mat"

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=mock_vision_run,
        ):
            events = []
            async for event in _stream_chat_ir(ctx):
                events.append(event)

        # Should yield StreamEvent with full text, then TextResult
        assert len(events) == 2
        assert isinstance(events[0], StreamEvent)
        assert events[0].content == "A cat sitting on a mat"
        assert isinstance(events[1], TextResult)
        assert events[1].content == "A cat sitting on a mat"
        assert events[1].finish_reason == "stop"

    async def test_complete_chat_ir_vision_legacy(self) -> None:
        """_complete_chat_ir with pixel_values uses vlm_generate in legacy path."""
        from mlx_manager.mlx_server.services.inference import _complete_chat_ir, _GenContext

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")

        ctx = _GenContext(
            model=model,
            tokenizer=tokenizer,
            prompt="What is this?",
            max_tokens=50,
            temperature=0.5,
            top_p=1.0,
            stop_token_ids={128009},
            adapter=adapter,
            model_id="test/model",
            completion_id="chatcmpl-vision-complete",
            created=1700000000,
            tools=None,
            pixel_values=MagicMock(),  # Trigger vision path
            messages=None,  # Legacy path
        )

        # Mock run_on_metal_thread to return vision result
        async def mock_vision_run(fn, **kwargs):
            return ("A dog playing fetch", "stop")

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=mock_vision_run,
        ):
            result = await _complete_chat_ir(ctx)

        # Should return InferenceResult with word-count tokens
        assert result.result.content == "A dog playing fetch"
        assert result.result.finish_reason == "stop"
        # Word count: 4 words
        assert result.completion_tokens == 4


# ---------------------------------------------------------------------------
# Adapter fallback for completion API
# ---------------------------------------------------------------------------


class TestCompletionAdapterFallback:
    """Test adapter fallback in generate_completion when adapter is None."""

    async def test_completion_creates_default_adapter_when_none(self) -> None:
        """generate_completion creates default adapter when loaded.adapter is None."""
        from mlx_manager.mlx_server.services.inference import generate_completion

        tokenizer = _make_fake_tokenizer()
        model = MagicMock()
        loaded = MagicMock()
        loaded.model = model
        loaded.tokenizer = tokenizer
        loaded.adapter = None  # No adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        # Mock _generate_raw_completion
        mock_result = {
            "id": "cmpl-test",
            "object": "text_completion",
            "choices": [{"text": "response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.services.inference._generate_raw_completion",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = await generate_completion(
                model_id="test/model",
                prompt="Hello",
                stream=False,
            )

        # Should succeed and return result
        assert result["id"] == "cmpl-test"
        assert result["choices"][0]["text"] == "response"


# ---------------------------------------------------------------------------
# Logfire instrumentation
# ---------------------------------------------------------------------------


class TestLogfireInstrumentation:
    """Test logfire span creation when logfire is available."""

    async def test_generate_chat_stream_with_logfire(self) -> None:
        """generate_chat_stream creates logfire span when available."""
        from mlx_manager.mlx_server.services.inference import generate_chat_stream

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        async def fake_generate_step(**kwargs):
            yield TextResult(content="Hi", finish_reason="stop")

        # Mock logfire module
        mock_logfire = MagicMock()
        mock_span_context = MagicMock()
        mock_logfire.span = MagicMock(return_value=mock_span_context)

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch.object(adapter, "generate_step", side_effect=fake_generate_step),
            patch("mlx_manager.mlx_server.services.inference.LOGFIRE_AVAILABLE", True),
            patch("mlx_manager.mlx_server.services.inference.logfire", mock_logfire),
        ):
            events = []
            async for event in await generate_chat_stream(
                model_id="test/model",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[{"type": "function", "function": {"name": "test"}}],
            ):
                events.append(event)

        # Logfire span should be created with correct params
        mock_logfire.span.assert_called_once_with(
            "chat_completion",
            model="test/model",
            max_tokens=4096,
            temperature=1.0,
            stream=True,
            has_tools=True,
        )
        # Span context manager should be entered and exited
        mock_span_context.__enter__.assert_called_once()
        mock_span_context.__exit__.assert_called_once()

    async def test_generate_chat_complete_response_with_logfire(self) -> None:
        """generate_chat_complete_response creates logfire span when available."""
        from mlx_manager.mlx_server.services.inference import generate_chat_complete_response

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        async def fake_generate(**kwargs):
            return TextResult(content="Complete", finish_reason="stop")

        # Mock logfire
        mock_logfire = MagicMock()
        mock_span_context = MagicMock()
        mock_logfire.span = MagicMock(return_value=mock_span_context)

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch.object(adapter, "generate", side_effect=fake_generate),
            patch("mlx_manager.mlx_server.services.inference.LOGFIRE_AVAILABLE", True),
            patch("mlx_manager.mlx_server.services.inference.logfire", mock_logfire),
        ):
            await generate_chat_complete_response(
                model_id="test/model",
                messages=[{"role": "user", "content": "Test"}],
            )

        # Logfire span should be created for non-streaming
        mock_logfire.span.assert_called_once_with(
            "chat_completion",
            model="test/model",
            max_tokens=4096,
            temperature=1.0,
            stream=False,
            has_tools=False,
        )
        mock_span_context.__enter__.assert_called_once()
        mock_span_context.__exit__.assert_called_once()

    async def test_logfire_info_on_stream_completion(self) -> None:
        """Logfire info logged on stream completion when available."""
        from mlx_manager.mlx_server.services.inference import _GenContext, _stream_chat_ir

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")

        ctx = _GenContext(
            model=model,
            tokenizer=tokenizer,
            prompt="test",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop_token_ids={128009},
            adapter=adapter,
            model_id="test/model",
            completion_id="chatcmpl-logfire",
            created=1700000000,
            tools=None,
            pixel_values=None,
            messages=None,  # Legacy path to exercise logfire.info in legacy stream
        )

        # Mock stream_from_metal_thread
        async def mock_stream(produce_fn, **kwargs):
            yield ("Hello", 1, False)
            yield ("", 128009, True)

        mock_logfire = MagicMock()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.stream_from_metal_thread",
                side_effect=mock_stream,
            ),
            patch("mlx_manager.mlx_server.services.inference.LOGFIRE_AVAILABLE", True),
            patch("mlx_manager.mlx_server.services.inference.logfire", mock_logfire),
        ):
            events = []
            async for event in _stream_chat_ir(ctx):
                events.append(event)

        # Logfire.info should be called with completion metadata
        mock_logfire.info.assert_called_once()
        call_args = mock_logfire.info.call_args
        assert call_args[0][0] == "stream_completion_finished"
        assert call_args[1]["completion_id"] == "chatcmpl-logfire"
        assert call_args[1]["finish_reason"] == "stop"

    async def test_logfire_info_on_complete_chat(self) -> None:
        """Logfire info logged on complete chat when available."""
        from mlx_manager.mlx_server.services.inference import _complete_chat_ir, _GenContext

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")

        ctx = _GenContext(
            model=model,
            tokenizer=tokenizer,
            prompt="test",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop_token_ids={128009},
            adapter=adapter,
            model_id="test/model",
            completion_id="chatcmpl-complete-logfire",
            created=1700000000,
            tools=None,
            pixel_values=None,
            messages=None,  # Legacy path
        )

        # Mock run_on_metal_thread
        async def mock_run(fn, **kwargs):
            return ("Response text", "stop")

        mock_logfire = MagicMock()

        with (
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                side_effect=mock_run,
            ),
            patch("mlx_manager.mlx_server.services.inference.LOGFIRE_AVAILABLE", True),
            patch("mlx_manager.mlx_server.services.inference.logfire", mock_logfire),
        ):
            await _complete_chat_ir(ctx)

        # Logfire.info should be called with completion metadata
        mock_logfire.info.assert_called_once()
        call_args = mock_logfire.info.call_args
        assert call_args[0][0] == "completion_finished"
        assert call_args[1]["completion_id"] == "chatcmpl-complete-logfire"
        assert call_args[1]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Tool warning and debug logging
# ---------------------------------------------------------------------------


class TestToolWarnings:
    """Test warning log when tools requested but not supported."""

    async def test_warning_when_tools_not_supported(self) -> None:
        """Log warning when tools requested but model doesn't support them."""
        from mlx_manager.mlx_server.services.inference import _prepare_generation

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("default")

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        # Override adapter to not support tools
        adapter.supports_tool_calling = MagicMock(return_value=False)

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch("mlx_manager.mlx_server.services.inference.logger") as mock_logger,
        ):
            await _prepare_generation(
                model_id="test/model",
                messages=[{"role": "user", "content": "Test"}],
                tools=[{"type": "function", "function": {"name": "search"}}],
                enable_prompt_injection=False,
            )

        # Logger.info should be called with warning message
        mock_logger.info.assert_called()
        # Check that warning message contains expected text
        call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("does not support tool calling" in str(msg) for msg in call_args)


# ---------------------------------------------------------------------------
# Modern path logging (tool detection in streaming)
# ---------------------------------------------------------------------------


class TestModernPathLogging:
    """Test debug logging in modern adapter path."""

    async def test_modern_stream_logs_tool_calls(self) -> None:
        """Modern streaming path logs when tool calls detected."""
        from mlx_manager.mlx_server.services.inference import _GenContext, _stream_chat_ir

        loaded, model, tokenizer, adapter = _make_loaded_model_with_real_adapter("qwen")

        # tool_calls are dicts
        tool_call_dict = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "test"}'},
        }

        # Mock adapter.generate_step to yield tool calls
        async def fake_generate_step(**kwargs):
            yield StreamEvent(type="content", content="Searching")
            yield TextResult(
                content="Searching", finish_reason="tool_calls", tool_calls=[tool_call_dict]
            )

        ctx = _GenContext(
            model=model,
            tokenizer=tokenizer,
            prompt="test",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop_token_ids={128009},
            adapter=adapter,
            model_id="test/model",
            completion_id="chatcmpl-modern-log",
            created=1700000000,
            tools=[{"type": "function", "function": {"name": "search"}}],
            pixel_values=None,
            messages=[{"role": "user", "content": "Search"}],  # Modern path
        )

        with (
            patch.object(adapter, "generate_step", side_effect=fake_generate_step),
            patch("mlx_manager.mlx_server.services.inference.logger") as mock_logger,
        ):
            events = []
            async for event in _stream_chat_ir(ctx):
                events.append(event)

        # Logger.debug should be called with tool call detection
        mock_logger.debug.assert_called()
        call_args = [call[0][0] for call in mock_logger.debug.call_args_list]
        # Should log detected tool calls
        assert any("Detected" in str(msg) and "tool call" in str(msg) for msg in call_args)
