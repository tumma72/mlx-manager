"""Tests for run_in_executor usage in inference.py token counting.

Gap 5: Verify that tokenizer.encode is called through loop.run_in_executor
(off-thread, with None executor = default ThreadPoolExecutor) rather than
directly on the event-loop thread.

The function under test is _complete_chat_ir() in services/inference.py.
We intercept asyncio.get_running_loop() to return a spy loop that records
run_in_executor calls, then confirm the tokenizer.encode was submitted to
the executor with None as the first argument.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_manager.mlx_server.models.ir import TextResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_tokenizer() -> MagicMock:
    """Create a lightweight fake tokenizer."""
    tok = MagicMock()
    tok.eos_token_id = 128009
    tok.unk_token_id = 0
    tok.encode = MagicMock(return_value=list(range(10)))
    tok.apply_chat_template = MagicMock(return_value="<formatted_prompt>")
    tok.tokenizer = tok  # Processor-wrapper pattern
    return tok


def _make_gen_context(tokenizer=None, messages=None, adapter=None) -> Any:
    """Build a _GenContext for _complete_chat_ir."""
    from mlx_manager.mlx_server.services.inference import _GenContext

    tokenizer = tokenizer or _make_fake_tokenizer()
    return _GenContext(
        model=MagicMock(),
        tokenizer=tokenizer,
        prompt="What is 2+2?",
        max_tokens=16,
        temperature=1.0,
        top_p=1.0,
        stop_token_ids=set(),
        adapter=adapter,
        model_id="test-model",
        completion_id="chatcmpl-test",
        created=1234567890,
        tools=None,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Gap 5: run_in_executor call verification
# ---------------------------------------------------------------------------


class TestRunInExecutorTokenizerEncode:
    """tokenizer.encode is called via loop.run_in_executor(None, ...) in _complete_chat_ir."""

    @pytest.mark.asyncio
    async def test_prompt_tokenization_uses_run_in_executor(self):
        """Prompt token counting submits tokenizer.encode to executor, not direct call.

        We verify:
        1. run_in_executor was called at least once
        2. The first argument (executor) is None (default thread pool)
        3. The second argument is tokenizer.encode
        """
        tokenizer = _make_fake_tokenizer()
        ctx = _make_gen_context(tokenizer=tokenizer)

        executor_calls: list[tuple] = []

        async def fake_run_in_executor(executor, fn, *args):
            executor_calls.append((executor, fn, args))
            # Actually call the function synchronously for the test
            return fn(*args)

        mock_loop = MagicMock()
        mock_loop.run_in_executor = fake_run_in_executor

        # run_on_metal_thread is imported locally inside _complete_chat_ir
        # so we patch it at the utils.metal module level
        mock_metal_result = ("Hello!", "stop")
        metal_mock = AsyncMock(return_value=mock_metal_result)

        with (
            patch(
                "mlx_manager.mlx_server.services.inference.asyncio.get_running_loop",
                return_value=mock_loop,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                metal_mock,
            ),
        ):
            from mlx_manager.mlx_server.services.inference import _complete_chat_ir

            await _complete_chat_ir(ctx)

        # run_in_executor must have been called
        assert len(executor_calls) >= 1, "run_in_executor was never called"

        # First call: prompt tokenization; executor must be None (default thread pool)
        first_executor, first_fn, first_args = executor_calls[0]
        assert first_executor is None, (
            f"Expected None (default ThreadPoolExecutor) as executor, got: {first_executor}"
        )

        # The function submitted must be tokenizer.encode (or the actual_tokenizer.encode)
        assert first_fn is tokenizer.encode, (
            f"Expected tokenizer.encode to be submitted, got: {first_fn}"
        )

    @pytest.mark.asyncio
    async def test_response_tokenization_uses_run_in_executor(self):
        """Completion token counting also submits tokenizer.encode through executor."""
        tokenizer = _make_fake_tokenizer()
        ctx = _make_gen_context(tokenizer=tokenizer)

        executor_calls: list[tuple] = []

        async def fake_run_in_executor(executor, fn, *args):
            executor_calls.append((executor, fn, args))
            return fn(*args)

        mock_loop = MagicMock()
        mock_loop.run_in_executor = fake_run_in_executor

        mock_metal_result = ("Response text here", "stop")
        metal_mock = AsyncMock(return_value=mock_metal_result)

        with (
            patch(
                "mlx_manager.mlx_server.services.inference.asyncio.get_running_loop",
                return_value=mock_loop,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                metal_mock,
            ),
        ):
            from mlx_manager.mlx_server.services.inference import _complete_chat_ir

            await _complete_chat_ir(ctx)

        # There should be exactly two run_in_executor calls:
        # one for prompt token counting, one for completion token counting
        assert len(executor_calls) >= 2, (
            f"Expected 2+ run_in_executor calls (prompt + completion), got: {len(executor_calls)}"
        )

        # All calls must use None executor
        for i, (executor, fn, args) in enumerate(executor_calls):
            assert executor is None, f"Call {i}: executor must be None, got {executor}"

    @pytest.mark.asyncio
    async def test_executor_called_with_correct_text(self):
        """The prompt text passed to executor matches ctx.prompt."""
        tokenizer = _make_fake_tokenizer()
        ctx = _make_gen_context(tokenizer=tokenizer)

        encoder_inputs: list[str] = []

        def capturing_encode(text: str) -> list[int]:
            encoder_inputs.append(text)
            return list(range(5))

        tokenizer.encode = capturing_encode

        async def fake_run_in_executor(executor, fn, *args):
            return fn(*args)

        mock_loop = MagicMock()
        mock_loop.run_in_executor = fake_run_in_executor

        mock_metal_result = ("result text", "stop")
        metal_mock = AsyncMock(return_value=mock_metal_result)

        with (
            patch(
                "mlx_manager.mlx_server.services.inference.asyncio.get_running_loop",
                return_value=mock_loop,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                metal_mock,
            ),
        ):
            from mlx_manager.mlx_server.services.inference import _complete_chat_ir

            await _complete_chat_ir(ctx)

        # The prompt must have been passed to encode
        assert ctx.prompt in encoder_inputs, (
            f"ctx.prompt '{ctx.prompt}' not found in encoder inputs: {encoder_inputs}"
        )

    @pytest.mark.asyncio
    async def test_modern_path_also_uses_run_in_executor(self):
        """When ctx.messages is set (modern path), completion encoding goes through executor."""
        from mlx_manager.mlx_server.models.adapters.composable import create_adapter

        tokenizer = _make_fake_tokenizer()
        adapter = create_adapter("default", tokenizer, model_type="text-gen")
        ctx = _make_gen_context(
            tokenizer=tokenizer,
            messages=[{"role": "user", "content": "Hello"}],
            adapter=adapter,
        )

        executor_calls: list[tuple] = []

        async def fake_run_in_executor(executor, fn, *args):
            executor_calls.append((executor, fn, args))
            return fn(*args)

        mock_loop = MagicMock()
        mock_loop.run_in_executor = fake_run_in_executor

        # Mock adapter.generate to return a TextResult
        fake_text_result = TextResult(content="Hi there!", finish_reason="stop")

        with (
            patch(
                "mlx_manager.mlx_server.services.inference.asyncio.get_running_loop",
                return_value=mock_loop,
            ),
            patch.object(
                adapter, "generate", new_callable=AsyncMock, return_value=fake_text_result
            ),
        ):
            from mlx_manager.mlx_server.services.inference import _complete_chat_ir

            await _complete_chat_ir(ctx)

        # Should call run_in_executor at least once (for prompt tokens)
        assert len(executor_calls) >= 1
        # All executors must be None
        for executor, fn, args in executor_calls:
            assert executor is None

    @pytest.mark.asyncio
    async def test_executor_uses_default_thread_pool_not_custom(self):
        """The None executor argument means 'use the default ThreadPoolExecutor'.

        This is a behavioral contract: passing None to run_in_executor means
        the default executor (configured on the event loop) is used, not a
        custom one. This keeps MLX inference tokenization off the event loop
        thread without requiring a separately managed executor.
        """
        tokenizer = _make_fake_tokenizer()
        ctx = _make_gen_context(tokenizer=tokenizer)

        non_none_executor_calls = []

        async def spy_run_in_executor(executor, fn, *args):
            if executor is not None:
                non_none_executor_calls.append(executor)
            return fn(*args)

        mock_loop = MagicMock()
        mock_loop.run_in_executor = spy_run_in_executor

        mock_metal_result = ("output", "stop")
        metal_mock = AsyncMock(return_value=mock_metal_result)

        with (
            patch(
                "mlx_manager.mlx_server.services.inference.asyncio.get_running_loop",
                return_value=mock_loop,
            ),
            patch(
                "mlx_manager.mlx_server.utils.metal.run_on_metal_thread",
                metal_mock,
            ),
        ):
            from mlx_manager.mlx_server.services.inference import _complete_chat_ir

            await _complete_chat_ir(ctx)

        # No custom executor should have been used
        assert len(non_none_executor_calls) == 0, (
            f"Custom executor was used unexpectedly: {non_none_executor_calls}"
        )
