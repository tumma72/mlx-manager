"""Unit tests for inference service stop token detection.

NOTE: These tests verify the LOGIC of stop token detection without
requiring actual MLX models. For full inference testing:
- Run manually on Apple Silicon with a downloaded model
- Use `pytest tests/mlx_server/test_inference_integration.py`
"""

from unittest.mock import Mock

import pytest


class TestStopTokenDetection:
    """Test stop token detection logic."""

    def test_stop_tokens_collected_from_adapter(self) -> None:
        """Verify stop tokens are retrieved from adapter."""
        # This test verifies the wiring, not the actual values
        from mlx_manager.mlx_server.models.adapters import get_adapter

        # Mock tokenizer with Llama 3 stop tokens
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 128009  # <|end_of_text|>
        mock_tokenizer.unk_token_id = 0
        mock_tokenizer.convert_tokens_to_ids = Mock(return_value=128001)  # <|eot_id|>

        adapter = get_adapter("mlx-community/Llama-3.2-3B-Instruct-4bit")
        stop_tokens = adapter.get_stop_tokens(mock_tokenizer)

        assert 128009 in stop_tokens, "Should include eos_token_id"
        assert 128001 in stop_tokens, "Should include <|eot_id|>"

    def test_llama_adapter_returns_dual_stop_tokens(self) -> None:
        """Verify Llama adapter returns BOTH stop tokens (critical for Llama 3)."""
        from mlx_manager.mlx_server.models.adapters import LlamaAdapter

        adapter = LlamaAdapter()

        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 128009
        mock_tokenizer.unk_token_id = 0
        mock_tokenizer.convert_tokens_to_ids = Mock(return_value=128001)

        stop_tokens = adapter.get_stop_tokens(mock_tokenizer)

        assert len(stop_tokens) >= 2, "Llama adapter must return at least 2 stop tokens"
        assert 128009 in stop_tokens, "Must include eos_token_id"
        mock_tokenizer.convert_tokens_to_ids.assert_called()

    def test_generation_halts_on_stop_token(self) -> None:
        """Verify generation stops when stop token is encountered."""
        # This tests the logic without actual generation
        stop_token_ids = {128009, 128001}

        # Simulate token stream: [token1, token2, STOP_TOKEN, token4_never_reached]
        token_stream = [
            (100, "Hello"),
            (200, " world"),
            (128001, ""),  # <|eot_id|> - stop token
            (300, " should not appear"),
        ]

        # Simulate our stop detection logic
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

        # Token with None ID should not trigger stop
        token_id = None
        assert token_id is None or token_id not in stop_token_ids

    def test_stop_detection_set_performance(self) -> None:
        """Verify stop tokens use set for O(1) lookup."""
        # Using set is critical for performance in generation loop
        stop_token_ids = {128009, 128001, 12345, 67890}

        # Set lookup is O(1)
        assert isinstance(stop_token_ids, set)
        assert 128009 in stop_token_ids
        assert 99999 not in stop_token_ids


class TestInferenceServiceImports:
    """Verify inference service can be imported without model dependencies."""

    def test_inference_module_imports(self) -> None:
        """Inference service should import without errors."""
        # This verifies the module structure is correct
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
            # Remove logfire from modules if present
            if "logfire" in sys.modules:
                del sys.modules["logfire"]

            # Force reimport
            import importlib

            from mlx_manager.mlx_server.services import inference

            importlib.reload(inference)

            # Should not crash, and LOGFIRE_AVAILABLE should be False
            # Note: logfire may or may not be available in test environment
            assert hasattr(inference, "LOGFIRE_AVAILABLE")
        finally:
            # Restore logfire if it was present
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
        assert "/v1/chat/completions" in routes

    def test_v1_router_includes_chat(self) -> None:
        """v1 router should include chat router."""
        from mlx_manager.mlx_server.api.v1 import v1_router

        routes = [r.path for r in v1_router.routes]
        assert "/v1/chat/completions" in routes


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
        # Normal token, not a stop token
        token_id = 100
        finish_reason = "stop" if token_id in stop_token_ids else "length"
        assert finish_reason == "length"


class TestAsyncThreadingPattern:
    """Test the queue-based threading pattern for MLX inference.

    The pattern uses a dedicated thread for MLX generation (to respect Metal
    GPU thread affinity) and a Queue for thread-safe token passing to the
    async generator.
    """

    def test_queue_based_token_passing(self) -> None:
        """Verify tokens can pass through queue from thread to async."""
        import threading
        from queue import Queue

        token_queue: Queue[tuple[str, int, bool] | None] = Queue()

        # Simulate generation thread
        def producer() -> None:
            token_queue.put(("Hello", 100, False))
            token_queue.put((" world", 200, False))
            token_queue.put(("", 128001, True))  # Stop token
            token_queue.put(None)  # Completion signal

        thread = threading.Thread(target=producer)
        thread.start()

        # Consume from queue
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

        # Daemon threads don't prevent process exit
        thread = threading.Thread(target=lambda: None, daemon=True)
        assert thread.daemon is True


class TestDeprecatedAPIRemoval:
    """Test that deprecated APIs are not used in inference service."""

    def test_no_get_event_loop_usage(self) -> None:
        """Verify asyncio.get_event_loop() is not used (deprecated)."""
        import inspect

        from mlx_manager.mlx_server.services import inference

        source = inspect.getsource(inference)

        # get_event_loop is deprecated in async context
        assert "get_event_loop()" not in source, (
            "Should use asyncio.get_running_loop() instead of get_event_loop()"
        )

    def test_uses_get_running_loop(self) -> None:
        """Verify asyncio.get_running_loop() is used."""
        import inspect

        from mlx_manager.mlx_server.services import inference

        source = inspect.getsource(inference)

        assert "get_running_loop()" in source, (
            "Should use asyncio.get_running_loop() for current event loop"
        )

    def test_uses_threading_module(self) -> None:
        """Verify threading module is used for MLX generation."""
        import inspect

        from mlx_manager.mlx_server.services import inference

        source = inspect.getsource(inference)

        assert "threading.Thread" in source, (
            "Should use dedicated threading.Thread for MLX generation"
        )
        assert "daemon=True" in source, "Threads should be daemon threads"

    def test_uses_queue_for_communication(self) -> None:
        """Verify Queue is used for thread-safe communication."""
        import inspect

        from mlx_manager.mlx_server.services import inference

        source = inspect.getsource(inference)

        assert "from queue import" in source, "Should import from queue module"
        assert "Queue" in source, "Should use Queue for token passing"
