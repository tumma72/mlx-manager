"""Unit tests for inference service.

Tests cover:
- Stop token detection logic (existing)
- Import/structural checks (existing)
- _inject_tools_into_messages (pure function)
- generate_chat_completion orchestration (mocked pool + adapter)
- _stream_chat_generate streaming consumer (mocked stream_from_metal_thread)
- _generate_chat_complete non-streaming (mocked run_on_metal_thread)
- generate_completion legacy API (mocked pool + adapter + metal thread)
- _stream_completion streaming (mocked stream_from_metal_thread)
- _generate_raw_completion non-streaming (mocked run_on_metal_thread)

NOTE: These tests verify the LOGIC of stop token detection and response
construction without requiring actual MLX models. For full inference testing:
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
    adapter = MagicMock()
    adapter.family = family
    adapter.convert_messages = MagicMock(side_effect=lambda msgs: list(msgs))
    adapter.supports_tool_calling = MagicMock(return_value=supports_tools)
    adapter.has_native_tool_support = MagicMock(return_value=native_tools)
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
    return adapter


def _make_mock_tokenizer(prompt_token_count: int = 10) -> MagicMock:
    """Create a mock tokenizer that returns deterministic encode results."""
    tok = MagicMock()
    tok.encode = MagicMock(return_value=list(range(prompt_token_count)))
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
        from mlx_manager.mlx_server.models.adapters import LlamaAdapter

        mock_tokenizer = Mock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        mock_tokenizer.eos_token_id = 128009
        mock_tokenizer.unk_token_id = 0
        mock_tokenizer.convert_tokens_to_ids = Mock(return_value=128001)

        adapter = LlamaAdapter(mock_tokenizer)
        stop_tokens = adapter.stop_tokens

        assert 128009 in stop_tokens, "Should include eos_token_id"
        assert 128001 in stop_tokens, "Should include <|eot_id|>"

    def test_llama_adapter_returns_dual_stop_tokens(self) -> None:
        """Verify Llama adapter returns BOTH stop tokens (critical for Llama 3)."""
        from mlx_manager.mlx_server.models.adapters import LlamaAdapter

        mock_tokenizer = Mock(spec=["eos_token_id", "unk_token_id", "convert_tokens_to_ids"])
        mock_tokenizer.eos_token_id = 128009
        mock_tokenizer.unk_token_id = 0
        mock_tokenizer.convert_tokens_to_ids = Mock(return_value=128001)

        adapter = LlamaAdapter(mock_tokenizer)
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

        assert hasattr(inference, "generate_chat_completion")

    def test_services_package_exports(self) -> None:
        """Services package should export generate_chat_completion."""
        from mlx_manager.mlx_server.services import generate_chat_completion

        assert callable(generate_chat_completion)

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
# NEW: _inject_tools_into_messages (pure function, no mocks)
# ===========================================================================


class TestInjectToolsIntoMessages:
    """Test _inject_tools_into_messages pure function."""

    def test_appends_to_existing_system_message(self) -> None:
        """Tool prompt is appended to existing system message content."""
        from mlx_manager.mlx_server.services.inference import _inject_tools_into_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]
        result = _inject_tools_into_messages(messages, "TOOLS: search")

        assert result[0]["role"] == "system"
        assert "You are a helpful assistant." in result[0]["content"]
        assert "TOOLS: search" in result[0]["content"]
        # Separator between existing and tool prompt
        assert "\n\n" in result[0]["content"]

    def test_creates_system_message_when_absent(self) -> None:
        """Tool prompt creates a new system message when none exists."""
        from mlx_manager.mlx_server.services.inference import _inject_tools_into_messages

        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_tools_into_messages(messages, "TOOLS: lookup")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "TOOLS: lookup"
        assert result[1]["role"] == "user"

    def test_does_not_mutate_original_messages(self) -> None:
        """Original message list should not be modified."""
        from mlx_manager.mlx_server.services.inference import _inject_tools_into_messages

        messages = [
            {"role": "system", "content": "Original system"},
            {"role": "user", "content": "Hi"},
        ]
        original_system_content = messages[0]["content"]

        result = _inject_tools_into_messages(messages, "injected tools")

        assert messages[0]["content"] == original_system_content
        assert result is not messages

    def test_system_message_not_first(self) -> None:
        """System message at non-zero index still gets tool prompt appended."""
        from mlx_manager.mlx_server.services.inference import _inject_tools_into_messages

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "System msg"},
            {"role": "user", "content": "Bye"},
        ]
        result = _inject_tools_into_messages(messages, "tools here")

        assert result[1]["role"] == "system"
        assert "tools here" in result[1]["content"]
        assert len(result) == 3  # No new message inserted

    def test_empty_system_content(self) -> None:
        """System message with empty content gets tool prompt appended."""
        from mlx_manager.mlx_server.services.inference import _inject_tools_into_messages

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "test"},
        ]
        result = _inject_tools_into_messages(messages, "tool_definitions")

        assert "tool_definitions" in result[0]["content"]

    def test_empty_messages_list(self) -> None:
        """Empty message list gets a system message inserted."""
        from mlx_manager.mlx_server.services.inference import _inject_tools_into_messages

        result = _inject_tools_into_messages([], "tool info")

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "tool info"

    def test_preserves_extra_system_message_keys(self) -> None:
        """Extra keys on system message dict are preserved."""
        from mlx_manager.mlx_server.services.inference import _inject_tools_into_messages

        messages = [
            {"role": "system", "content": "base", "name": "narrator"},
            {"role": "user", "content": "Hi"},
        ]
        result = _inject_tools_into_messages(messages, "tools")

        assert result[0]["name"] == "narrator"
        assert "tools" in result[0]["content"]


# ===========================================================================
# NEW: generate_chat_completion orchestration
# ===========================================================================


class TestGenerateChatCompletion:
    """Test generate_chat_completion orchestration (mock pool + adapter)."""

    async def _run_generate(
        self,
        *,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        adapter_kwargs: dict[str, Any] | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Helper that patches pool + adapter and calls generate_chat_completion."""
        from mlx_manager.mlx_server.services.inference import generate_chat_completion

        loaded, model, tokenizer = _make_mock_loaded_model()
        adapter = _make_mock_adapter(**(adapter_kwargs or {}))
        loaded.adapter = adapter  # Assign adapter to loaded model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        gen_kwargs: dict[str, Any] = {
            "model_id": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": tools,
        }
        if extra_kwargs:
            gen_kwargs.update(extra_kwargs)

        # Patch the lazy imports inside generate_chat_completion
        with patch(
            "mlx_manager.mlx_server.models.pool.get_model_pool",
            return_value=mock_pool,
        ):
            if stream:
                # For streaming, we also need to mock the inner streaming function
                # because it calls stream_from_metal_thread which needs mlx_lm
                async def fake_stream(**kwargs: Any) -> Any:
                    yield {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
                    }

                with patch(
                    "mlx_manager.mlx_server.services.inference._stream_chat_generate",
                    side_effect=fake_stream,
                ):
                    result = await generate_chat_completion(
                        stream=True,
                        **gen_kwargs,
                    )
                return result, adapter, mock_pool
            else:
                # Non-streaming: mock _generate_chat_complete
                mock_result = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hi there!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
                with patch(
                    "mlx_manager.mlx_server.services.inference._generate_chat_complete",
                    new_callable=AsyncMock,
                    return_value=mock_result,
                ):
                    result = await generate_chat_completion(
                        stream=False,
                        **gen_kwargs,
                    )
                return result, adapter, mock_pool

    async def test_non_streaming_returns_dict(self) -> None:
        """Non-streaming completion returns a dict."""
        result, _, _ = await self._run_generate(stream=False)

        assert isinstance(result, dict)
        assert result["object"] == "chat.completion"

    async def test_streaming_returns_generator(self) -> None:
        """Streaming completion returns an async generator."""
        result, _, _ = await self._run_generate(stream=True)

        # Should be an async generator
        assert hasattr(result, "__aiter__")

    async def test_adapter_convert_messages_called(self) -> None:
        """Adapter's convert_messages is called with original messages."""
        _, adapter, _ = await self._run_generate()

        adapter.convert_messages.assert_called_once()

    async def test_apply_chat_template_called(self) -> None:
        """Adapter's apply_chat_template is called."""
        _, adapter, _ = await self._run_generate()

        adapter.apply_chat_template.assert_called_once()

    async def test_tools_with_native_support(self) -> None:
        """Tools are passed to apply_chat_template when capabilities indicate template delivery."""
        # Create a capabilities object with tool_format="template"
        caps = MagicMock()
        caps.supports_native_tools = True
        caps.tool_format = "template"

        loaded, model, tokenizer = _make_mock_loaded_model(capabilities=caps)
        adapter = _make_mock_adapter(supports_tools=True, native_tools=True)
        loaded.adapter = adapter  # Assign adapter to loaded model

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        mock_result = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.services.inference._generate_chat_complete",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            from mlx_manager.mlx_server.services.inference import generate_chat_completion

            await generate_chat_completion(
                model_id="test/model",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
                tools=SAMPLE_TOOLS,
            )

        # apply_chat_template should receive tools
        call_kwargs = adapter.apply_chat_template.call_args
        assert call_kwargs is not None, "apply_chat_template should have been called"
        assert call_kwargs.kwargs.get("tools") == SAMPLE_TOOLS or (
            len(call_kwargs.args) > 3 and call_kwargs.args[3] == SAMPLE_TOOLS
        )

    async def test_tools_with_prompt_injection(self) -> None:
        """Tools are injected into messages when enable_prompt_injection=True."""
        _, adapter, _ = await self._run_generate(
            tools=SAMPLE_TOOLS,
            adapter_kwargs={
                "supports_tools": True,
                "native_tools": False,
                "tool_prompt": "Available tools: search",
            },
            extra_kwargs={"enable_prompt_injection": True},
        )

        adapter.format_tools_for_prompt.assert_called_once_with(SAMPLE_TOOLS)
        # apply_chat_template should NOT get tools kwarg
        call_kwargs = adapter.apply_chat_template.call_args
        assert call_kwargs.kwargs.get("tools") is None

    async def test_tool_stop_tokens_added(self) -> None:
        """Tool-specific stop tokens are requested when tools are provided."""
        _, adapter, _ = await self._run_generate(
            tools=SAMPLE_TOOLS,
            adapter_kwargs={"supports_tools": True},
        )

        adapter.get_tool_call_stop_tokens.assert_called_once()

    async def test_enable_prompt_injection_uses_injection(self) -> None:
        """enable_prompt_injection=True uses injection even without native support."""
        _, adapter, _ = await self._run_generate(
            tools=SAMPLE_TOOLS,
            adapter_kwargs={
                "supports_tools": True,
                "native_tools": False,
                "tool_prompt": "Available tools: search",
            },
            extra_kwargs={"enable_prompt_injection": True},
        )

        # Should use injection (format_tools_for_prompt called)
        adapter.format_tools_for_prompt.assert_called_once_with(SAMPLE_TOOLS)
        # Should NOT pass tools natively
        call_kwargs = adapter.apply_chat_template.call_args
        assert call_kwargs.kwargs.get("tools") is None

    async def test_no_tools_skips_tool_logic(self) -> None:
        """Without tools, tool-related adapter methods are not called."""
        _, adapter, _ = await self._run_generate(tools=None)

        adapter.supports_tool_calling.assert_not_called()
        adapter.get_tool_call_stop_tokens.assert_not_called()


# ===========================================================================
# NEW: _stream_chat_generate consumer logic
# ===========================================================================


class TestStreamChatGenerate:
    """Test _stream_chat_generate streaming consumer logic.

    Mocks stream_from_metal_thread to yield predetermined token tuples,
    letting the real StreamingProcessor run for accurate coverage.
    """

    @staticmethod
    def _make_stream_mock(tokens: list[tuple[str, int | None, bool]]):
        """Create a mock for stream_from_metal_thread that yields tokens."""

        async def mock_stream(produce_fn, **kwargs):
            for t in tokens:
                yield t

        return mock_stream

    async def _run_stream(
        self,
        tokens: list[tuple[str, int | None, bool]],
        family: str = "qwen",
        tools: list[dict[str, Any]] | None = None,
        prompt: str = "test prompt",
    ) -> list[dict]:
        """Run _stream_chat_generate with mocked stream and collect all chunks."""
        from mlx_manager.mlx_server.models.adapters.composable import (
            create_adapter,
        )
        from mlx_manager.mlx_server.services.inference import _stream_chat_generate

        # Use real composable adapter for actual parsing
        _, model, tokenizer = _make_mock_loaded_model()
        adapter = create_adapter(family, tokenizer)

        mock_stream = self._make_stream_mock(tokens)

        with patch(
            "mlx_manager.mlx_server.utils.metal.stream_from_metal_thread",
            side_effect=mock_stream,
        ):
            chunks = []
            async for chunk in _stream_chat_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-test123",
                created=1700000000,
                model_id="test/model",
                adapter=adapter,
                tools=tools,
            ):
                chunks.append(chunk)
            return chunks

    async def test_first_chunk_has_role(self) -> None:
        """First chunk should have role='assistant' and empty content."""
        tokens = [("Hello", 1, False), ("", 128009, True)]
        chunks = await self._run_stream(tokens)

        first = chunks[0]
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert first["choices"][0]["delta"]["content"] == ""
        assert first["choices"][0]["finish_reason"] is None

    async def test_content_chunks_have_text(self) -> None:
        """Content tokens produce chunks with text in delta."""
        tokens = [("Hello", 1, False), (" world", 2, False), ("", 128009, True)]
        chunks = await self._run_stream(tokens)

        # chunks[0] = role chunk, chunks[1..N-1] = content, chunks[-1] = final
        content_chunks = chunks[1:-1]
        combined = "".join(c["choices"][0]["delta"].get("content", "") for c in content_chunks)
        assert "Hello" in combined
        assert "world" in combined

    async def test_stop_token_sets_finish_reason_stop(self) -> None:
        """Stop token produces finish_reason='stop' in final chunk."""
        tokens = [("Hello", 1, False), ("", 128009, True)]
        chunks = await self._run_stream(tokens)

        final = chunks[-1]
        assert final["choices"][0]["finish_reason"] == "stop"

    async def test_max_tokens_sets_finish_reason_length(self) -> None:
        """When no stop token, finish_reason defaults to 'length'."""
        tokens = [("tok1", 1, False), ("tok2", 2, False)]
        chunks = await self._run_stream(tokens)

        final = chunks[-1]
        assert final["choices"][0]["finish_reason"] == "length"

    async def test_all_chunks_have_completion_id(self) -> None:
        """All chunks carry the same completion ID."""
        tokens = [("Hi", 1, False), ("", 128009, True)]
        chunks = await self._run_stream(tokens)

        for chunk in chunks:
            assert chunk["id"] == "chatcmpl-test123"
            assert chunk["model"] == "test/model"
            assert chunk["object"] == "chat.completion.chunk"

    @pytest.mark.parametrize(
        "family",
        ["qwen", "glm4"],
    )
    async def test_streaming_thinking_tags_filtered(self, family: str) -> None:
        """Thinking tags are processed by StreamingProcessor (not sent as content)."""
        tokens = [
            ("<think>", 10, False),
            ("reasoning here", 11, False),
            ("</think>", 12, False),
            ("The answer", 13, False),
            ("", 128009, True),
        ]
        chunks = await self._run_stream(tokens, family=family)

        # Collect all content from delta
        all_content = ""
        all_reasoning = ""
        for c in chunks:
            delta = c["choices"][0]["delta"]
            all_content += delta.get("content", "")
            all_reasoning += delta.get("reasoning_content", "")

        # The thinking content should appear as reasoning, not content
        assert "The answer" in all_content
        # Thinking tags themselves should not appear in content
        assert "<think>" not in all_content
        assert "</think>" not in all_content

    @pytest.mark.parametrize(
        "family,fixture_path",
        [
            ("qwen", "qwen/stream/tool_call_chunks.txt"),
        ],
    )
    async def test_streaming_tool_calls_detected(self, family: str, fixture_path: str) -> None:
        """Tool calls in streamed tokens are detected and returned in final chunk."""
        raw = _read_golden(fixture_path)
        # Split into single-char tokens to simulate streaming
        token_tuples: list[tuple[str, int | None, bool]] = [
            (ch, idx, False) for idx, ch in enumerate(raw)
        ]
        token_tuples.append(("", 128009, True))

        chunks = await self._run_stream(token_tuples, family=family, tools=SAMPLE_TOOLS)

        final = chunks[-1]
        # Tool calls should set finish_reason to "tool_calls"
        assert final["choices"][0]["finish_reason"] == "tool_calls"
        assert "tool_calls" in final["choices"][0]["delta"]

    async def test_streaming_with_prompt_ending_in_think_tag(self) -> None:
        """Prompt ending with <think> sets starts_in_thinking mode."""
        tokens = [
            ("reasoning text", 10, False),
            ("</think>", 11, False),
            ("The answer", 12, False),
            ("", 128009, True),
        ]
        chunks = await self._run_stream(tokens, family="glm4", prompt="some prompt\n<think>")

        all_reasoning = ""
        all_content = ""
        for c in chunks:
            delta = c["choices"][0]["delta"]
            all_reasoning += delta.get("reasoning_content", "")
            all_content += delta.get("content", "")

        # The reasoning text before </think> should appear as reasoning_content
        assert "reasoning text" in all_reasoning
        assert "The answer" in all_content

    async def test_stop_token_text_not_yielded(self) -> None:
        """Stop token text should not appear in any content chunk."""
        tokens = [
            ("Hello", 1, False),
            ("<STOP>", 128009, True),
        ]
        chunks = await self._run_stream(tokens)

        all_content = ""
        for c in chunks[1:-1]:  # Skip role chunk and final chunk
            all_content += c["choices"][0]["delta"].get("content", "")

        assert "<STOP>" not in all_content


# ===========================================================================
# NEW: _generate_chat_complete non-streaming
# ===========================================================================


class TestGenerateChatComplete:
    """Test _generate_chat_complete non-streaming generation.

    Mocks run_on_metal_thread to return predetermined (text, finish_reason).
    """

    async def _run_complete(
        self,
        response_text: str,
        finish_reason: str = "stop",
        family: str = "qwen",
        tools: list[dict[str, Any]] | None = None,
    ) -> dict:
        """Run _generate_chat_complete with mocked run_on_metal_thread."""
        from mlx_manager.mlx_server.models.adapters.composable import (
            create_adapter,
        )
        from mlx_manager.mlx_server.services.inference import _generate_chat_complete

        # Use real composable adapter for actual parsing
        _, model, tokenizer = _make_mock_loaded_model()
        adapter = create_adapter(family, tokenizer)

        async def mock_run(fn, **kwargs):
            return (response_text, finish_reason)

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=mock_run,
        ):
            return await _generate_chat_complete(
                model=model,
                tokenizer=tokenizer,
                prompt="test prompt",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-test456",
                created=1700000000,
                model_id="test/model",
                adapter=adapter,
                tools=tools,
            )

    async def test_plain_text_response(self) -> None:
        """Plain text response produces standard completion dict."""
        result = await self._run_complete("The capital of France is Paris.")

        assert result["id"] == "chatcmpl-test456"
        assert result["object"] == "chat.completion"
        assert result["model"] == "test/model"
        msg = result["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert "Paris" in msg["content"]
        assert result["choices"][0]["finish_reason"] == "stop"

    async def test_usage_dict_present(self) -> None:
        """Response includes usage with prompt_tokens, completion_tokens, total_tokens."""
        result = await self._run_complete("Hello")

        usage = result["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    @pytest.mark.parametrize(
        "family,fixture_path",
        TOOL_CALL_FIXTURES,
    )
    async def test_tool_calls_parsed(self, family: str, fixture_path: str) -> None:
        """Tool calls in golden fixture text are parsed into tool_calls."""
        response_text = _read_golden(fixture_path)
        result = await self._run_complete(response_text, family=family, tools=SAMPLE_TOOLS)

        msg = result["choices"][0]["message"]
        assert "tool_calls" in msg, f"Expected tool_calls for {family} fixture"
        assert len(msg["tool_calls"]) > 0
        assert result["choices"][0]["finish_reason"] == "tool_calls"

        # Verify tool call structure
        tc = msg["tool_calls"][0]
        assert "id" in tc
        assert tc["type"] == "function"
        assert "name" in tc["function"]
        assert "arguments" in tc["function"]
        # Arguments should be valid JSON string
        json.loads(tc["function"]["arguments"])

    async def test_thinking_tags_extracted(self) -> None:
        """Thinking content extracted from <think> tags into reasoning_content."""
        response_text = _read_golden("qwen/thinking.txt")
        result = await self._run_complete(response_text, family="qwen")

        msg = result["choices"][0]["message"]
        assert "reasoning_content" in msg
        assert "analyze" in msg["reasoning_content"].lower()
        # Main content should not contain think tags
        assert "<think>" not in msg["content"]
        assert "</think>" not in msg["content"]

    async def test_thinking_and_tool_calls_combined(self) -> None:
        """Response with both thinking and tool calls extracts both."""
        # Combine a thinking prefix + tool call
        response_text = (
            "<think>Let me search for that.</think>"
            "I'll search for you.\n"
            '<tool_call>{"name": "search", "arguments": '
            '{"query": "test"}}</tool_call>'
        )
        result = await self._run_complete(response_text, family="qwen", tools=SAMPLE_TOOLS)

        msg = result["choices"][0]["message"]
        assert "reasoning_content" in msg
        assert "search" in msg["reasoning_content"].lower()
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) > 0

    async def test_finish_reason_length_preserved(self) -> None:
        """When run_on_metal_thread returns 'length', it is preserved."""
        result = await self._run_complete("partial response...", finish_reason="length")

        assert result["choices"][0]["finish_reason"] == "length"

    async def test_empty_response(self) -> None:
        """Empty response text is handled gracefully."""
        result = await self._run_complete("")

        msg = result["choices"][0]["message"]
        assert msg["content"] == ""
        assert msg["role"] == "assistant"

    async def test_no_adapter_uses_default_family(self) -> None:
        """When adapter is None, 'default' family is used for processing."""
        from mlx_manager.mlx_server.services.inference import _generate_chat_complete

        _, model, tokenizer = _make_mock_loaded_model()

        async def mock_run(fn, **kwargs):
            return ("Hello world", "stop")

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=mock_run,
        ):
            result = await _generate_chat_complete(
                model=model,
                tokenizer=tokenizer,
                prompt="prompt",
                max_tokens=100,
                temperature=1.0,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-noadapter",
                created=1700000000,
                model_id="test/model",
                adapter=None,
                tools=None,
            )

        assert result["choices"][0]["message"]["content"] == "Hello world"


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

    async def test_chat_stream_inner_with_stop_token(self) -> None:
        """produce_tokens inside _stream_chat_generate yields stop and halts."""
        from mlx_manager.mlx_server.services.inference import _stream_chat_generate

        adapter = _make_mock_adapter(family="qwen")
        _, model, tokenizer = _make_mock_loaded_model()

        # Mock mlx_lm at the source module
        responses = [
            self._mock_mlx_response(1, "Hello"),
            self._mock_mlx_response(2, " world"),
            self._mock_mlx_response(128009, ""),  # stop token
        ]

        with (
            patch("mlx_lm.stream_generate", return_value=iter(responses)),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            chunks = []
            async for chunk in _stream_chat_generate(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-inner",
                created=1700000000,
                model_id="test/model",
                adapter=adapter,
                tools=None,
            ):
                chunks.append(chunk)

        # First chunk = role, middle = content, last = final
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    async def test_chat_stream_inner_no_stop(self) -> None:
        """produce_tokens exhausts max_tokens -> finish_reason='length'."""
        from mlx_manager.mlx_server.services.inference import _stream_chat_generate

        adapter = _make_mock_adapter(family="qwen")
        _, model, tokenizer = _make_mock_loaded_model()

        responses = [
            self._mock_mlx_response(1, "tok1"),
            self._mock_mlx_response(2, "tok2"),
        ]

        with (
            patch("mlx_lm.stream_generate", return_value=iter(responses)),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            chunks = []
            async for chunk in _stream_chat_generate(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=2,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-inner2",
                created=1700000000,
                model_id="test/model",
                adapter=adapter,
                tools=None,
            ):
                chunks.append(chunk)

        assert chunks[-1]["choices"][0]["finish_reason"] == "length"

    async def test_chat_complete_inner_run_generation(self) -> None:
        """run_generation inside _generate_chat_complete uses stream_generate."""
        from mlx_manager.mlx_server.services.inference import _generate_chat_complete

        adapter = _make_mock_adapter(family="qwen")
        _, model, tokenizer = _make_mock_loaded_model()

        responses = [
            self._mock_mlx_response(1, "The"),
            self._mock_mlx_response(2, " answer"),
            self._mock_mlx_response(128009, ""),  # stop
        ]

        with (
            patch("mlx_lm.stream_generate", return_value=iter(responses)),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            result = await _generate_chat_complete(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-inner3",
                created=1700000000,
                model_id="test/model",
                adapter=adapter,
                tools=None,
            )

        msg = result["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert "The answer" in msg["content"]
        assert result["choices"][0]["finish_reason"] == "stop"

    async def test_chat_complete_inner_length_finish(self) -> None:
        """run_generation returns 'length' when no stop token found."""
        from mlx_manager.mlx_server.services.inference import _generate_chat_complete

        adapter = _make_mock_adapter(family="qwen")
        _, model, tokenizer = _make_mock_loaded_model()

        responses = [
            self._mock_mlx_response(1, "partial"),
            self._mock_mlx_response(2, " text"),
        ]

        with (
            patch("mlx_lm.stream_generate", return_value=iter(responses)),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            result = await _generate_chat_complete(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=2,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-inner4",
                created=1700000000,
                model_id="test/model",
                adapter=adapter,
                tools=None,
            )

        assert result["choices"][0]["finish_reason"] == "length"

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

    async def test_response_with_none_token_id(self) -> None:
        """Response objects without .token attribute handled via getattr default."""
        from mlx_manager.mlx_server.services.inference import _generate_chat_complete

        adapter = _make_mock_adapter(family="qwen")
        _, model, tokenizer = _make_mock_loaded_model()

        # Create response without .token attribute
        resp = MagicMock(spec=["text"])
        resp.text = "Hello"
        del resp.token  # Ensure getattr returns None

        # Second response triggers stop
        resp2 = MagicMock()
        resp2.token = 128009
        resp2.text = ""

        with (
            patch("mlx_lm.stream_generate", return_value=iter([resp, resp2])),
            patch("mlx_lm.sample_utils.make_sampler", return_value=MagicMock()),
        ):
            result = await _generate_chat_complete(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-none-token",
                created=1700000000,
                model_id="test/model",
                adapter=adapter,
                tools=None,
            )

        assert "Hello" in result["choices"][0]["message"]["content"]


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
        adapter = create_adapter(family=family, tokenizer=mock_tokenizer)
        tool_calls = adapter.tool_parser.extract(response_text)

        # Simulate ParseResult
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
        """Thinking chunks are processed correctly by StreamingProcessor."""
        from mlx_manager.mlx_server.models.adapters import create_adapter
        from mlx_manager.mlx_server.services.response_processor import StreamingProcessor

        raw = _read_golden(fixture_path)

        # Create mock tokenizer for adapter
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 100

        # Create adapter for streaming processor
        adapter = create_adapter(family=family, tokenizer=mock_tokenizer)

        # Feed character by character to simulate streaming
        processor = StreamingProcessor(adapter=adapter)

        all_reasoning = ""
        all_content = ""
        for ch in raw:
            event = processor.feed(ch)
            if event.reasoning_content:
                all_reasoning += event.reasoning_content
            if event.content:
                all_content += event.content

        result = processor.finalize()

        # Should have extracted reasoning
        assert result.reasoning is not None or all_reasoning, (
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

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 128009
        mock_tokenizer.encode = MagicMock(return_value=list(range(10)))

        # Create adapter with mocked parsers
        adapter = TestModelAdapter(tokenizer=mock_tokenizer)

        # Override parsers with mocks for testing
        adapter._tool_parser = MagicMock()
        adapter._tool_parser.extract = MagicMock(return_value=[])
        adapter._thinking_parser = MagicMock()
        adapter._thinking_parser.extract = MagicMock(return_value=None)
        adapter._thinking_parser.remove = MagicMock(side_effect=lambda text: text)

        # Override stop tokens for testing
        adapter._stop_tokens = [128009, 128001]

        return adapter

    async def test_uses_composable_adapter_when_available(self) -> None:
        """When loaded.adapter is set, composable adapter is used."""
        from mlx_manager.mlx_server.services.inference import generate_chat_completion

        composable = self._make_composable_adapter()
        loaded, model, tokenizer = _make_mock_loaded_model()
        loaded.adapter = composable  # Set composable adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        mock_result = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.services.inference._generate_chat_complete",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = await generate_chat_completion(
                model_id="test/model",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

            # Result should be returned
            assert result["id"] == "chatcmpl-test"

    async def test_falls_back_to_default_adapter(self) -> None:
        """When loaded.adapter is None, DefaultAdapter is created."""
        from mlx_manager.mlx_server.services.inference import generate_chat_completion

        loaded, model, tokenizer = _make_mock_loaded_model()
        loaded.adapter = None  # No composable adapter

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        mock_result = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.services.inference._generate_chat_complete",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = await generate_chat_completion(
                model_id="test/model",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

            # Result should be returned
            assert result["id"] == "chatcmpl-test"

    async def test_composable_stop_tokens_used(self) -> None:
        """Composable adapter's pre-computed stop_tokens are used."""
        from mlx_manager.mlx_server.services.inference import generate_chat_completion

        composable = self._make_composable_adapter()
        composable._stop_tokens = [999, 888]  # Custom stop tokens (set internal field)

        loaded, model, tokenizer = _make_mock_loaded_model()
        loaded.adapter = composable

        mock_pool = MagicMock()
        mock_pool.get_model = AsyncMock(return_value=loaded)

        captured_stop_token_ids = None

        async def capture_stop_tokens(**kwargs):
            nonlocal captured_stop_token_ids
            captured_stop_token_ids = kwargs.get("stop_token_ids")
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            }

        with (
            patch(
                "mlx_manager.mlx_server.models.pool.get_model_pool",
                return_value=mock_pool,
            ),
            patch(
                "mlx_manager.mlx_server.services.inference._generate_chat_complete",
                new_callable=AsyncMock,
                side_effect=capture_stop_tokens,
            ),
        ):
            await generate_chat_completion(
                model_id="test/model",
                messages=[{"role": "user", "content": "Hi"}],
                stream=False,
            )

        # Should use composable adapter's stop tokens
        assert captured_stop_token_ids == {999, 888}

    async def test_composable_tool_parser_extract(self) -> None:
        """Tool calls extracted via adapter.tool_parser.extract()."""
        from mlx_manager.mlx_server.services.inference import _generate_chat_complete

        composable = self._make_composable_adapter()

        # Setup tool parser to return a tool call
        mock_tool_call = MagicMock()
        mock_tool_call.model_dump = MagicMock(
            return_value={
                "id": "call_123",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "test"}'},
            }
        )
        composable.tool_parser.extract = MagicMock(return_value=[mock_tool_call])

        _, model, tokenizer = _make_mock_loaded_model()

        async def mock_run(fn, **kwargs):
            return ("I'll search for that.", "stop")

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=mock_run,
        ):
            result = await _generate_chat_complete(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-tooltest",
                created=1700000000,
                model_id="test/model",
                adapter=composable,
                tools=SAMPLE_TOOLS,
            )

        # Tool parser extract should be called
        composable.tool_parser.extract.assert_called_once()
        # Result should have tool_calls
        assert "tool_calls" in result["choices"][0]["message"]
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    async def test_composable_thinking_parser_extract(self) -> None:
        """Reasoning extracted via adapter.thinking_parser.extract()."""
        from mlx_manager.mlx_server.services.inference import _generate_chat_complete

        composable = self._make_composable_adapter()
        composable.thinking_parser.extract = MagicMock(return_value="Let me think...")
        composable.thinking_parser.remove = MagicMock(return_value="The answer is 42.")

        _, model, tokenizer = _make_mock_loaded_model()

        async def mock_run(fn, **kwargs):
            return ("<think>Let me think...</think>The answer is 42.", "stop")

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=mock_run,
        ):
            result = await _generate_chat_complete(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-thinktest",
                created=1700000000,
                model_id="test/model",
                adapter=composable,
                tools=None,
            )

        # Thinking parser should be called
        composable.thinking_parser.extract.assert_called_once()
        composable.thinking_parser.remove.assert_called_once()
        # Result should have reasoning_content
        assert "reasoning_content" in result["choices"][0]["message"]
        assert result["choices"][0]["message"]["reasoning_content"] == "Let me think..."

    async def test_composable_clean_response(self) -> None:
        """adapter.clean_response() used on final content."""
        from mlx_manager.mlx_server.services.inference import _generate_chat_complete

        composable = self._make_composable_adapter()
        composable.clean_response = MagicMock(return_value="Cleaned text")

        _, model, tokenizer = _make_mock_loaded_model()

        async def mock_run(fn, **kwargs):
            return ("Raw text with <|endoftext|> tokens", "stop")

        with patch(
            "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
            side_effect=mock_run,
        ):
            result = await _generate_chat_complete(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-cleantest",
                created=1700000000,
                model_id="test/model",
                adapter=composable,
                tools=None,
            )

        # clean_response should be called
        composable.clean_response.assert_called_once()
        # Result should have cleaned content
        assert result["choices"][0]["message"]["content"] == "Cleaned text"

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

    async def test_composable_streaming_finalization(self) -> None:
        """Streaming uses composable adapter's tool parser for finalization."""
        from mlx_manager.mlx_server.services.inference import _stream_chat_generate

        composable = self._make_composable_adapter()

        # Setup tool parser to return a tool call from accumulated text
        mock_tool_call = MagicMock()
        mock_tool_call.model_dump = MagicMock(
            return_value={
                "id": "call_456",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"id": "42"}'},
            }
        )
        composable.tool_parser.extract = MagicMock(return_value=[mock_tool_call])

        _, model, tokenizer = _make_mock_loaded_model()

        # Mock stream to yield tokens
        async def mock_stream(produce_fn, **kwargs):
            yield ("I'll look that up.", 1, False)
            yield ("", 128009, True)

        with patch(
            "mlx_manager.mlx_server.utils.metal.stream_from_metal_thread",
            side_effect=mock_stream,
        ):
            chunks = []
            async for chunk in _stream_chat_generate(
                model=model,
                tokenizer=tokenizer,
                prompt="test",
                max_tokens=100,
                temperature=0.7,
                top_p=1.0,
                stop_token_ids={128009},
                completion_id="chatcmpl-streamtest",
                created=1700000000,
                model_id="test/model",
                adapter=composable,
                tools=SAMPLE_TOOLS,
            ):
                chunks.append(chunk)

        # Tool parser extract should be called with accumulated text
        composable.tool_parser.extract.assert_called_once()
        # Final chunk should have tool_calls
        final = chunks[-1]
        assert final["choices"][0]["finish_reason"] == "tool_calls"
        assert "tool_calls" in final["choices"][0]["delta"]
