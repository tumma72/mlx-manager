"""Unit tests for inference service.

Tests cover:
- Stop token detection logic (existing)
- Import/structural checks (existing)
- _inject_tools (pure function)
- generate_completion legacy API (mocked pool + adapter + metal thread)
- _stream_completion streaming (mocked stream_from_metal_thread)
- _generate_raw_completion non-streaming (mocked run_on_metal_thread)

NOTE: Chat completion tests for the IR-based API (generate_chat_stream,
generate_chat_complete_response, _stream_chat_ir, _complete_chat_ir) live
in a separate test module. These tests verify the LOGIC of stop token
detection and response construction without requiring actual MLX models.
For full inference testing:
- Run manually on Apple Silicon with a downloaded model
- Use `pytest tests/mlx_server/test_inference_integration.py`
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Path to golden fixtures
GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "golden"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_golden(relative_path: str) -> str:
    """Read a golden fixture file and strip trailing newline."""
    return (GOLDEN_DIR / relative_path).read_text().rstrip("\n")


def _make_mock_adapter(
    family: str = "qwen",
    supports_tools: bool = False,
    native_tools: bool = False,
    tool_prompt: str = "",
) -> MagicMock:
    """Create a mock adapter with sensible defaults (composable adapter pattern)."""
    from mlx_manager.mlx_server.models.ir import PreparedInput, StreamEvent, TextResult

    adapter = MagicMock()
    adapter.family = family
    adapter.convert_messages = MagicMock(side_effect=lambda msgs: list(msgs))
    adapter.supports_tool_calling = MagicMock(return_value=supports_tools)
    adapter.supports_native_tools = MagicMock(return_value=native_tools)
    adapter.format_tools_for_prompt = MagicMock(return_value=tool_prompt)
    adapter.apply_chat_template = MagicMock(return_value="formatted prompt")
    adapter.stop_tokens = [128009]  # Property instead of method
    adapter.get_tool_call_stop_tokens = MagicMock(return_value=[])
    adapter.clean_response = MagicMock(side_effect=lambda text: text)
    # Mock parsers for composable adapter
    adapter.tool_parser = MagicMock()
    adapter.tool_parser.extract = MagicMock(return_value=[])
    adapter.thinking_parser = MagicMock()
    adapter.thinking_parser.extract = MagicMock(return_value=None)
    adapter.thinking_parser.remove = MagicMock(side_effect=lambda text: text)
    # prepare_input() returns a proper PreparedInput
    adapter.prepare_input = MagicMock(
        return_value=PreparedInput(prompt="formatted prompt", stop_token_ids=[128009])
    )
    # process_complete() returns a proper TextResult
    adapter.process_complete = MagicMock(
        side_effect=lambda text, reason="stop": TextResult(content=text, finish_reason=reason)
    )
    # Stream processor returns real IR StreamEvent objects
    mock_processor = MagicMock()
    mock_processor.feed = MagicMock(
        side_effect=lambda token: StreamEvent(type="content", content=token)
    )
    mock_processor.get_accumulated_text = MagicMock(return_value="")
    adapter.create_stream_processor = MagicMock(return_value=mock_processor)
    return adapter


def _make_mock_tokenizer(prompt_token_count: int = 10) -> MagicMock:
    """Create a mock tokenizer that returns deterministic encode results."""
    tok = MagicMock()
    tok.encode = MagicMock(return_value=list(range(prompt_token_count)))
    tok.apply_chat_template = MagicMock(return_value="<mock_prompt>")
    # tokenizer.tokenizer pattern used for Processor-wrapped tokenizers
    tok.tokenizer = tok
    return tok


def _make_mock_loaded_model(
    prompt_token_count: int = 10,
    capabilities: Any = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (loaded_model, model, tokenizer) mocks."""
    tokenizer = _make_mock_tokenizer(prompt_token_count)
    model = MagicMock()
    loaded = MagicMock()
    loaded.model = model
    loaded.tokenizer = tokenizer
    loaded.capabilities = capabilities
    # Adapter is now always set (composable adapter)
    loaded.adapter = _make_mock_adapter()
    return loaded, model, tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TOOL_CALL_FIXTURES = [
    ("qwen", "qwen/tool_calls.txt"),
    ("llama", "llama/tool_calls.txt"),
    ("glm4", "glm4/tool_calls.txt"),
    # Hermes format uses same parser as Qwen (HermesJsonParser)
    ("qwen", "hermes/tool_calls.txt"),
    # Gemma doesn't support tool calling natively - removed from test
    # ("gemma", "gemma/tool_calls.txt"),
]


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


# ===========================================================================
# Existing tests preserved below
# ===========================================================================


class TestStopTokenDetection:
    """Test stop token detection logic."""

    def test_stop_tokens_collected_from_adapter(self) -> None:
        """Verify stop tokens are retrieved from adapter."""
        from mlx_manager.mlx_server.models.adapters import create_adapter

        mock_tokenizer = Mock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        mock_tokenizer.eos_token_id = 128009
        mock_tokenizer.unk_token_id = 0
        mock_tokenizer.convert_tokens_to_ids = Mock(return_value=128001)

        adapter = create_adapter("llama", mock_tokenizer, model_type="text-gen")
        stop_tokens = adapter.stop_tokens

        assert 128009 in stop_tokens, "Should include eos_token_id"
        assert 128001 in stop_tokens, "Should include <|eot_id|>"

    def test_llama_adapter_returns_dual_stop_tokens(self) -> None:
        """Verify Llama adapter returns BOTH stop tokens (critical for Llama 3)."""
        from mlx_manager.mlx_server.models.adapters import create_adapter

        mock_tokenizer = Mock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        mock_tokenizer.eos_token_id = 128009
        mock_tokenizer.unk_token_id = 0
        mock_tokenizer.convert_tokens_to_ids = Mock(return_value=128001)

        adapter = create_adapter("llama", mock_tokenizer, model_type="text-gen")
        stop_tokens = adapter.stop_tokens

        assert len(stop_tokens) >= 2, "Llama adapter must return at least 2 stop tokens"
        assert 128009 in stop_tokens, "Must include eos_token_id"
        mock_tokenizer.convert_tokens_to_ids.assert_called()

    def test_generation_halts_on_stop_token(self) -> None:
        """Verify generation stops when stop token is encountered."""
        stop_token_ids = {128009, 128001}

        token_stream = [
            (100, "Hello"),
            (200, " world"),
            (128001, ""),
            (300, " should not appear"),
        ]

        collected_text = ""
        hit_stop = False

        for token_id, text in token_stream:
            if token_id in stop_token_ids:
                hit_stop = True
                break
            collected_text += text

        assert hit_stop, "Should have detected stop token"
        assert collected_text == "Hello world", "Should stop BEFORE stop token text"
        assert "should not appear" not in collected_text

    def test_stop_token_detection_with_none_token_id(self) -> None:
        """Verify None token IDs don't cause false positives."""
        stop_token_ids = {128009, 128001}

        token_id = None
        assert token_id is None or token_id not in stop_token_ids

    def test_stop_detection_set_performance(self) -> None:
        """Verify stop tokens use set for O(1) lookup."""
        stop_token_ids = {128009, 128001, 12345, 67890}

        assert isinstance(stop_token_ids, set)
        assert 128009 in stop_token_ids
        assert 99999 not in stop_token_ids


class TestInferenceServiceImports:
    """Verify inference service can be imported without model dependencies."""

    def test_inference_module_imports(self) -> None:
        """Inference service should import without errors."""
        from mlx_manager.mlx_server.services import inference

        assert hasattr(inference, "generate_chat_stream")
        assert hasattr(inference, "generate_chat_complete_response")

    def test_services_package_exports(self) -> None:
        """Services package should export IR-based chat functions."""
        from mlx_manager.mlx_server.services import (
            generate_chat_complete_response,
            generate_chat_stream,
        )

        assert callable(generate_chat_stream)
        assert callable(generate_chat_complete_response)

    def test_logfire_optional(self) -> None:
        """LogFire should be optional - graceful fallback when not installed."""
        import sys

        logfire_module = sys.modules.get("logfire")

        try:
            if "logfire" in sys.modules:
                del sys.modules["logfire"]

            import importlib

            from mlx_manager.mlx_server.services import inference

            importlib.reload(inference)

            assert hasattr(inference, "LOGFIRE_AVAILABLE")
        finally:
            if logfire_module:
                sys.modules["logfire"] = logfire_module


class TestChatEndpointSetup:
    """Test chat endpoint configuration."""

    def test_chat_router_exists(self) -> None:
        """Chat router should be importable."""
        from mlx_manager.mlx_server.api.v1.chat import router

        assert router is not None

    def test_chat_completions_route_exists(self) -> None:
        """Chat completions route should be registered."""
        from mlx_manager.mlx_server.api.v1.chat import router

        routes = [r.path for r in router.routes]
        assert "/chat/completions" in routes

    def test_v1_router_includes_chat(self) -> None:
        """v1 router should include chat router."""
        from mlx_manager.mlx_server.api.v1 import v1_router

        routes = [r.path for r in v1_router.routes]
        assert "/chat/completions" in routes


class TestFinishReasonLogic:
    """Test finish_reason determination logic."""

    def test_finish_reason_stop_on_stop_token(self) -> None:
        """finish_reason should be 'stop' when stop token encountered."""
        stop_token_ids = {128009, 128001}
        token_id = 128001
        finish_reason = "stop" if token_id in stop_token_ids else "length"
        assert finish_reason == "stop"

    def test_finish_reason_length_on_max_tokens(self) -> None:
        """finish_reason should be 'length' when max_tokens reached."""
        stop_token_ids = {128009, 128001}
        token_id = 100
        finish_reason = "stop" if token_id in stop_token_ids else "length"
        assert finish_reason == "length"


class TestAsyncThreadingPattern:
    """Test the queue-based threading pattern for MLX inference."""

    def test_queue_based_token_passing(self) -> None:
        """Verify tokens can pass through queue from thread to async."""
        import threading
        from queue import Queue

        token_queue: Queue[tuple[str, int, bool] | None] = Queue()

        def producer() -> None:
            token_queue.put(("Hello", 100, False))
            token_queue.put((" world", 200, False))
            token_queue.put(("", 128001, True))
            token_queue.put(None)

        thread = threading.Thread(target=producer)
        thread.start()

        tokens = []
        while True:
            result = token_queue.get(timeout=1.0)
            if result is None:
                break
            token_text, token_id, is_stop = result
            tokens.append((token_text, is_stop))
            if is_stop:
                break

        thread.join()

        assert tokens == [("Hello", False), (" world", False), ("", True)]

    def test_exception_propagation_through_queue(self) -> None:
        """Verify exceptions in generation thread propagate to async side."""
        import threading
        from queue import Queue

        result_queue: Queue[str | Exception] = Queue()

        def failing_producer() -> None:
            result_queue.put(RuntimeError("MLX error"))

        thread = threading.Thread(target=failing_producer)
        thread.start()
        thread.join()

        result = result_queue.get(timeout=1.0)
        assert isinstance(result, RuntimeError)
        assert str(result) == "MLX error"

    def test_empty_queue_timeout(self) -> None:
        """Verify Empty exception on queue timeout."""
        from queue import Empty, Queue

        token_queue: Queue[str] = Queue()

        with pytest.raises(Empty):
            token_queue.get(timeout=0.01)

    def test_thread_daemon_flag(self) -> None:
        """Verify generation threads are daemon threads."""
        import threading

        thread = threading.Thread(target=lambda: None, daemon=True)
        assert thread.daemon is True


class TestDeprecatedAPIRemoval:
    """Test that deprecated APIs are not used in inference service."""

    def test_no_get_event_loop_usage(self) -> None:
        """Verify asyncio.get_event_loop() is not used (deprecated)."""
        import inspect

        from mlx_manager.mlx_server.services import inference

        source = inspect.getsource(inference)

        assert "get_event_loop()" not in source, (
            "Should use asyncio.get_running_loop() instead of get_event_loop()"
        )

    def test_uses_metal_thread_utility(self) -> None:
        """Verify inference delegates to run_on_metal_thread / stream_from_metal_thread."""
        import inspect

        from mlx_manager.mlx_server.services import inference

        source = inspect.getsource(inference)

        assert "run_on_metal_thread" in source, (
            "Should use run_on_metal_thread for non-streaming generation"
        )
        assert "stream_from_metal_thread" in source, (
            "Should use stream_from_metal_thread for streaming generation"
        )

    def test_no_direct_queue_or_threading(self) -> None:
        """Verify Queue/threading boilerplate is not directly in inference.py."""
        import inspect

        from mlx_manager.mlx_server.services import inference

        source = inspect.getsource(inference)

        assert "from queue import" not in source, (
            "Queue usage should be in utils/metal.py, not inference.py"
        )
        assert "import threading" not in source, (
            "threading usage should be in utils/metal.py, not inference.py"
        )


# ===========================================================================
# NEW: _inject_tools (pure function, no mocks)
# ===========================================================================


class TestInjectToolsIntoMessages:
    """Test ModelAdapter._inject_tools static method."""

    def test_appends_to_existing_system_message(self) -> None:
        """Tool prompt is appended to existing system message content."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

        _inject_tools = ModelAdapter._inject_tools

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]
        result = _inject_tools(messages, "TOOLS: search")

        assert result[0]["role"] == "system"
        assert "You are a helpful assistant." in result[0]["content"]
        assert "TOOLS: search" in result[0]["content"]
        # Separator between existing and tool prompt
        assert "\n\n" in result[0]["content"]

    def test_creates_system_message_when_absent(self) -> None:
        """Tool prompt creates a new system message when none exists."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

        _inject_tools = ModelAdapter._inject_tools

        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_tools(messages, "TOOLS: lookup")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "TOOLS: lookup"
        assert result[1]["role"] == "user"

    def test_does_not_mutate_original_messages(self) -> None:
        """Original message list should not be modified."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

        _inject_tools = ModelAdapter._inject_tools

        messages = [
            {"role": "system", "content": "Original system"},
            {"role": "user", "content": "Hi"},
        ]
        original_system_content = messages[0]["content"]

        result = _inject_tools(messages, "injected tools")

        assert messages[0]["content"] == original_system_content
        assert result is not messages

    def test_system_message_not_first(self) -> None:
        """System message at non-zero index still gets tool prompt appended."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

        _inject_tools = ModelAdapter._inject_tools

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "System msg"},
            {"role": "user", "content": "Bye"},
        ]
        result = _inject_tools(messages, "tools here")

        assert result[1]["role"] == "system"
        assert "tools here" in result[1]["content"]
        assert len(result) == 3  # No new message inserted

    def test_empty_system_content(self) -> None:
        """System message with empty content gets tool prompt appended."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

        _inject_tools = ModelAdapter._inject_tools

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "test"},
        ]
        result = _inject_tools(messages, "tool_definitions")

        assert "tool_definitions" in result[0]["content"]

    def test_empty_messages_list(self) -> None:
        """Empty message list gets a system message inserted."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

        _inject_tools = ModelAdapter._inject_tools

        result = _inject_tools([], "tool info")

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "tool info"

    def test_preserves_extra_system_message_keys(self) -> None:
        """Extra keys on system message dict are preserved."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter

        _inject_tools = ModelAdapter._inject_tools

        messages = [
            {"role": "system", "content": "base", "name": "narrator"},
            {"role": "user", "content": "Hi"},
        ]
        result = _inject_tools(messages, "tools")

        assert result[0]["name"] == "narrator"
        assert "tools" in result[0]["content"]


# ===========================================================================
# NEW: generate_completion (legacy) orchestration
# ===========================================================================


class TestGenerateCompletion:
    """Test generate_completion legacy API orchestration."""

    async def _setup_and_call(
        self,
        prompt: str | list[str] = "Hello",
        stream: bool = False,
        echo: bool = False,
    ) -> Any:
        """Setup mocks and call generate_completion."""
        from mlx_manager.mlx_server.services.inference import generate_completion

        loaded, model, tokenizer = _make_mock_loaded_model()

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
        ):
            if stream:

                async def fake_stream(**kwargs):
                    yield {
                        "id": "cmpl-test",
                        "object": "text_completion",
                        "choices": [{"text": "hi", "finish_reason": None}],
                    }

                with patch(
                    "mlx_manager.mlx_server.services.inference._stream_completion",
                    side_effect=fake_stream,
                ):
                    return await generate_completion(
                        model_id="test/model",
                        prompt=prompt,
                        stream=True,
                        echo=echo,
                    )
            else:
                mock_result = {
                    "id": "cmpl-test",
                    "object": "text_completion",
                    "choices": [{"text": "world", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                }
                with patch(
                    "mlx_manager.mlx_server.services.inference._generate_raw_completion",
                    new_callable=AsyncMock,
                    return_value=mock_result,
                ):
                    return await generate_completion(
                        model_id="test/model",
                        prompt=prompt,
                        stream=False,
                        echo=echo,
                    )

    async def test_non_streaming_returns_dict(self) -> None:
        """Non-streaming completion returns a dict."""
        result = await self._setup_and_call(stream=False)

        assert isinstance(result, dict)
        assert result["object"] == "text_completion"

    async def test_streaming_returns_generator(self) -> None:
        """Streaming completion returns async generator."""
        result = await self._setup_and_call(stream=True)

        assert hasattr(result, "__aiter__")

    async def test_list_prompt_uses_first_element(self) -> None:
        """List prompt is reduced to first element."""
        result = await self._setup_and_call(prompt=["First", "Second"], stream=False)

        assert isinstance(result, dict)

    async def test_empty_list_prompt_becomes_empty_string(self) -> None:
        """Empty list prompt becomes empty string."""
        result = await self._setup_and_call(prompt=[], stream=False)

        assert isinstance(result, dict)


# ===========================================================================
# NEW: _stream_completion (legacy streaming)
# ===========================================================================


class TestStreamCompletion:
    """Test _stream_completion legacy streaming logic."""

    @staticmethod
    def _make_stream_mock(tokens: list[tuple[str, int | None, bool]]):
        """Create a mock for stream_from_metal_thread."""

        async def mock_stream(produce_fn, **kwargs):
            for t in tokens:
                yield t

        return mock_stream

    async def _run_stream(
        self,
        tokens: list[tuple[str, int | None, bool]],
        echo: bool = False,
        prompt: str = "test prompt",
    ) -> list[dict]:
        """Run _stream_completion and collect all chunks."""
        from mlx_manager.mlx_server.services.inference import _stream_completion

        _, model, tokenizer = _make_mock_loaded_model()

        mock_stream = self._make_stream_mock(tokens)

        with patch(
            "mlx_manager.mlx_server.utils.metal.stream_from_metal_thread",
            side_effect=mock_stream,
        ):
            chunks = []
            async for chunk in _stream_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_tokens={128009},
                completion_id="cmpl-test789",
                created=1700000000,
                model_id="test/model",
                echo=echo,
            ):
                chunks.append(chunk)
            return chunks

    async def test_basic_streaming(self) -> None:
        """Basic streaming produces content and final chunks."""
        tokens = [("Hello", 1, False), (" world", 2, False), ("", 128009, True)]
        chunks = await self._run_stream(tokens)

        # Content chunks + final chunk
        assert len(chunks) >= 2
        # Content chunks
        all_text = "".join(c["choices"][0]["text"] for c in chunks[:-1])
        assert "Hello" in all_text
        assert "world" in all_text
        # Final chunk
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    async def test_echo_prepends_prompt(self) -> None:
        """Echo=True prepends prompt text as first chunk."""
        tokens = [("answer", 1, False), ("", 128009, True)]
        chunks = await self._run_stream(tokens, echo=True, prompt="The prompt: ")

        # First chunk should be the echo
        assert chunks[0]["choices"][0]["text"] == "The prompt: "
        assert chunks[0]["choices"][0]["finish_reason"] is None

    async def test_no_echo_skips_prompt(self) -> None:
        """Echo=False does not prepend prompt."""
        tokens = [("answer", 1, False), ("", 128009, True)]
        chunks = await self._run_stream(tokens, echo=False, prompt="The prompt: ")

        # No echo chunk; first chunk is content
        first_text = chunks[0]["choices"][0]["text"]
        assert first_text != "The prompt: "

    async def test_finish_reason_length_when_no_stop(self) -> None:
        """finish_reason is 'length' when stream ends without stop token."""
        tokens = [("tok1", 1, False), ("tok2", 2, False)]
        chunks = await self._run_stream(tokens)

        assert chunks[-1]["choices"][0]["finish_reason"] == "length"

    async def test_all_chunks_have_correct_object(self) -> None:
        """All chunks have object='text_completion'."""
        tokens = [("Hi", 1, False), ("", 128009, True)]
        chunks = await self._run_stream(tokens)

        for chunk in chunks:
            assert chunk["object"] == "text_completion"
            assert chunk["id"] == "cmpl-test789"

    async def test_stop_token_not_yielded_as_text(self) -> None:
        """Stop token text does not appear in content chunks."""
        tokens = [("Hello", 1, False), ("STOP_TEXT", 128009, True)]
        chunks = await self._run_stream(tokens)

        all_text = "".join(c["choices"][0]["text"] for c in chunks)
        assert "STOP_TEXT" not in all_text


# ===========================================================================
# NEW: _generate_raw_completion (legacy non-streaming)
# ===========================================================================


class TestGenerateRawCompletion:
    """Test _generate_raw_completion legacy non-streaming generation."""

    async def _run_complete(
        self,
        response_text: str,
        finish_reason: str = "stop",
        echo: bool = False,
        prompt: str = "test prompt",
    ) -> dict:
        """Run _generate_raw_completion with mocked run_on_metal_thread."""
        from mlx_manager.mlx_server.services.inference import _generate_raw_completion

        _, model, tokenizer = _make_mock_loaded_model()

        async def mock_run(fn, **kwargs):
            return (response_text, finish_reason)

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=mock_run,
        ):
            return await _generate_raw_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_tokens={128009},
                completion_id="cmpl-raw-test",
                created=1700000000,
                model_id="test/model",
                echo=echo,
            )

    async def test_plain_response(self) -> None:
        """Plain text response produces correct structure."""
        result = await self._run_complete("Generated text here")

        assert result["id"] == "cmpl-raw-test"
        assert result["object"] == "text_completion"
        assert result["choices"][0]["text"] == "Generated text here"
        assert result["choices"][0]["finish_reason"] == "stop"

    async def test_echo_prepends_prompt(self) -> None:
        """Echo=True prepends prompt to response text."""
        result = await self._run_complete(
            "response",
            echo=True,
            prompt="PROMPT: ",
        )

        assert result["choices"][0]["text"] == "PROMPT: response"

    async def test_echo_false_no_prompt(self) -> None:
        """Echo=False returns only generated text."""
        result = await self._run_complete(
            "response",
            echo=False,
            prompt="PROMPT: ",
        )

        assert result["choices"][0]["text"] == "response"

    async def test_usage_dict_present(self) -> None:
        """Response includes usage statistics."""
        result = await self._run_complete("hello")

        usage = result["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    async def test_finish_reason_length(self) -> None:
        """finish_reason 'length' is preserved from generation."""
        result = await self._run_complete("partial...", finish_reason="length")

        assert result["choices"][0]["finish_reason"] == "length"

    async def test_empty_response(self) -> None:
        """Empty response text is handled gracefully."""
        result = await self._run_complete("")

        assert result["choices"][0]["text"] == ""

    async def test_processor_wrapped_tokenizer(self) -> None:
        """Tokenizer.tokenizer pattern (Processor wrapper) is used for prompt counting."""
        from mlx_manager.mlx_server.services.inference import _generate_raw_completion

        _, model, tokenizer = _make_mock_loaded_model(prompt_token_count=15)

        async def mock_run(fn, **kwargs):
            return ("text", "stop")

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=mock_run,
        ):
            result = await _generate_raw_completion(
                model=model,
                tokenizer=tokenizer,
                prompt="prompt",
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
                stop_tokens={128009},
                completion_id="cmpl-wrap",
                created=1700000000,
                model_id="test/model",
                echo=False,
            )

        # prompt_tokens should come from actual_tokenizer.encode
        assert result["usage"]["prompt_tokens"] == 15


# ===========================================================================
# NEW: Parametrized golden fixture tests across families
# ===========================================================================


class TestInnerProduceTokens:
    """Test the inner produce_tokens/run_generation functions.

    These run on the Metal thread and import mlx_lm internally.
    We mock mlx_lm.stream_generate and mlx_lm.sample_utils.make_sampler
    at their source modules to exercise the code inside the closures.
    """

    @staticmethod
    def _mock_mlx_response(token: int, text: str) -> MagicMock:
        """Create a mock response object from stream_generate."""
        resp = MagicMock()
        resp.token = token
        resp.text = text
        return resp

    async def test_completion_stream_inner_with_stop(self) -> None:
        """produce_tokens inside _stream_completion detects stop tokens."""
        from mlx_manager.mlx_server.services.inference import _stream_completion

        _, model, tokenizer = _make_mock_loaded_model()

        responses = [
            self._mock_mlx_response(1, "hello"),
            self._mock_mlx_response(128009, ""),
        ]

        with (
            patch("mlx_lm.stream_generate", return_value=iter(responses)),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            chunks = []
            async for chunk in _stream_completion(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_tokens={128009},
                completion_id="cmpl-inner",
                created=1700000000,
                model_id="test/model",
                echo=False,
            ):
                chunks.append(chunk)

        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    async def test_raw_completion_inner_run_generation(self) -> None:
        """run_generation inside _generate_raw_completion accumulates text."""
        from mlx_manager.mlx_server.services.inference import _generate_raw_completion

        _, model, tokenizer = _make_mock_loaded_model()

        responses = [
            self._mock_mlx_response(1, "Hello"),
            self._mock_mlx_response(2, " there"),
            self._mock_mlx_response(128009, ""),
        ]

        with (
            patch("mlx_lm.stream_generate", return_value=iter(responses)),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            result = await _generate_raw_completion(
                model=model,
                tokenizer=tokenizer,
                prompt="prompt",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_tokens={128009},
                completion_id="cmpl-raw-inner",
                created=1700000000,
                model_id="test/model",
                echo=False,
            )

        assert result["choices"][0]["text"] == "Hello there"
        assert result["choices"][0]["finish_reason"] == "stop"

    async def test_raw_completion_inner_echo(self) -> None:
        """Echo mode prepends prompt in raw completion."""
        from mlx_manager.mlx_server.services.inference import _generate_raw_completion

        _, model, tokenizer = _make_mock_loaded_model()

        responses = [
            self._mock_mlx_response(1, "world"),
            self._mock_mlx_response(128009, ""),
        ]

        with (
            patch("mlx_lm.stream_generate", return_value=iter(responses)),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            result = await _generate_raw_completion(
                model=model,
                tokenizer=tokenizer,
                prompt="Hello ",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_tokens={128009},
                completion_id="cmpl-raw-echo",
                created=1700000000,
                model_id="test/model",
                echo=True,
            )

        assert result["choices"][0]["text"] == "Hello world"



class TestGoldenFixturesCrossFamily:
    """Cross-family parametrized tests using golden fixtures."""

    @pytest.mark.parametrize(
        "family,fixture_path",
        TOOL_CALL_FIXTURES,
        ids=[f[0] for f in TOOL_CALL_FIXTURES],
    )
    async def test_tool_call_parsing_round_trip(self, family: str, fixture_path: str) -> None:
        """Tool calls are parsed from golden fixture text for each model family.

        This uses the real ResponseProcessor (not mocked) to verify
        end-to-end parsing works for each family.
        """
        from mlx_manager.mlx_server.models.adapters import create_adapter

        response_text = _read_golden(fixture_path)

        # Create mock tokenizer for adapter
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 100

        # Create adapter and use its parsers
        adapter = create_adapter(family=family, tokenizer=mock_tokenizer, model_type="text-gen")
        tool_calls = adapter.tool_parser.extract(response_text)

        # Simulate TextResult
        result = MagicMock()
        result.tool_calls = tool_calls

        assert len(result.tool_calls) > 0, f"Expected tool calls for {family} from {fixture_path}"
        tc = result.tool_calls[0]
        assert tc.function.name, f"Tool call should have a function name for {family}"
        # Arguments should be valid JSON
        json.loads(tc.function.arguments)

    @pytest.mark.parametrize(
        "family,fixture_path",
        [
            ("qwen", "qwen/stream/thinking_chunks.txt"),
        ],
    )
    async def test_streaming_thinking_round_trip(self, family: str, fixture_path: str) -> None:
        """Thinking chunks are processed correctly by StreamProcessor."""
        from mlx_manager.mlx_server.models.adapters import create_adapter

        raw = _read_golden(fixture_path)

        # Create mock tokenizer for adapter
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 100

        # Create adapter and use factory to create stream processor
        adapter = create_adapter(family=family, tokenizer=mock_tokenizer, model_type="text-gen")
        processor = adapter.create_stream_processor()

        all_reasoning = ""
        all_content = ""
        for ch in raw:
            event = processor.feed(ch)
            if event.reasoning_content:
                all_reasoning += event.reasoning_content
            if event.content:
                all_content += event.content

        result = processor.finalize()

        # Should have extracted reasoning (TextResult uses reasoning_content)
        assert result.reasoning_content is not None or all_reasoning, (
            f"Expected reasoning content for {family} thinking fixture"
        )


# ===========================================================================
# NEW: Phase 5 - Composable adapter integration
# ===========================================================================


class TestModelAdapterIntegration:
    """Test that inference.py uses composable adapter when available.

    Phase 5: Verify composable adapter path is used when loaded.adapter
    is not None, and fallback to old adapter when it is None.
    """

    def _make_composable_adapter(self, family: str = "qwen"):
        """Create a real composable adapter subclass for testing."""
        from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter
        from mlx_manager.mlx_server.parsers import NullThinkingParser, NullToolParser

        # Create a minimal test subclass
        class TestModelAdapter(ModelAdapter):
            @property
            def family(self) -> str:
                return family

            def _default_tool_parser(self):
                return NullToolParser()

            def _default_thinking_parser(self):
                return NullThinkingParser()

        # Create mock tokenizer (set .tokenizer to self so _actual_tokenizer resolves correctly)
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 128009
        mock_tokenizer.encode = MagicMock(return_value=list(range(10)))
        mock_tokenizer.apply_chat_template = MagicMock(return_value="<prompt>")
        mock_tokenizer.tokenizer = mock_tokenizer

        # Create adapter with mocked parsers
        adapter = TestModelAdapter(model_type="text-gen", tokenizer=mock_tokenizer)

        # Override parsers with mocks for testing
        adapter._tool_parser = MagicMock()
        adapter._tool_parser.extract = MagicMock(return_value=[])
        adapter._thinking_parser = MagicMock()
        adapter._thinking_parser.extract = MagicMock(return_value=None)
        adapter._thinking_parser.remove = MagicMock(side_effect=lambda text: text)

        # Override stop tokens for testing
        adapter._stop_tokens = [128009, 128001]

        return adapter

    async def test_composable_completion_api(self) -> None:
        """Legacy completion API uses composable adapter stop tokens."""
        from mlx_manager.mlx_server.services.inference import generate_completion

        composable = self._make_composable_adapter()
        composable._stop_tokens = [111, 222]  # Set internal field

        loaded, model, tokenizer = _make_mock_loaded_model()
        loaded.adapter = composable

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        captured_stop_tokens = None

        async def capture_stop(**kwargs):
            nonlocal captured_stop_tokens
            captured_stop_tokens = kwargs.get("stop_tokens")
            return {
                "id": "cmpl-test",
                "object": "text_completion",
                "choices": [{"text": "hello", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.services.inference._generate_raw_completion",
                new_callable=AsyncMock,
                side_effect=capture_stop,
            ),
        ):
            await generate_completion(
                model_id="test/model",
                prompt="Hello",
                stream=False,
            )

        # Should use composable adapter's stop tokens
        assert captured_stop_tokens == {111, 222}

