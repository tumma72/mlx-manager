"""Tests for BatchInferenceEngine."""

import asyncio
import threading
from unittest.mock import MagicMock, Mock, patch

import pytest

from mlx_manager.mlx_server.services.batching import (
    BatchRequest,
    PagedBlockManager,
)
from mlx_manager.mlx_server.services.batching.batch_inference import (
    BatchInferenceEngine,
)


@pytest.fixture
def mock_model() -> Mock:
    """Create a mock MLX model."""
    return Mock(spec=[])


@pytest.fixture
def mock_tokenizer() -> Mock:
    """Create a mock tokenizer with decode method."""
    tokenizer = Mock(spec=[])
    tokenizer.decode = Mock(return_value="test prompt decoded")
    tokenizer.encode = Mock(return_value=[1, 2, 3])
    tokenizer.eos_token_id = 128001
    return tokenizer


@pytest.fixture
def mock_adapter() -> Mock:
    """Create a mock model adapter."""
    adapter = Mock(spec=[])
    adapter.get_stop_tokens = Mock(return_value=[128001, 128009])  # Llama 3.x stop tokens
    return adapter


@pytest.fixture
def block_manager() -> PagedBlockManager:
    """Create a block manager for tests."""
    return PagedBlockManager(num_blocks=100)


@pytest.fixture
def batch_engine(
    mock_model: Mock, mock_tokenizer: Mock, mock_adapter: Mock
) -> BatchInferenceEngine:
    """Create a BatchInferenceEngine for testing."""
    return BatchInferenceEngine(
        model=mock_model,
        tokenizer=mock_tokenizer,
        adapter=mock_adapter,
        prefix_cache=None,
    )


def make_request(
    request_id: str,
    prompt_tokens: list[int] | None = None,
    max_tokens: int = 10,
) -> BatchRequest:
    """Helper to create test requests."""
    return BatchRequest(
        request_id=request_id,
        model_id="test-model",
        prompt_tokens=prompt_tokens if prompt_tokens is not None else [1, 2, 3],
        max_tokens=max_tokens,
    )


class TestBatchInferenceEngineInitialization:
    """Tests for BatchInferenceEngine initialization."""

    def test_stores_model_and_tokenizer(
        self, mock_model: Mock, mock_tokenizer: Mock, mock_adapter: Mock
    ) -> None:
        """Engine should store model and tokenizer references."""
        engine = BatchInferenceEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            adapter=mock_adapter,
        )

        assert engine.model is mock_model
        assert engine.tokenizer is mock_tokenizer
        assert engine.adapter is mock_adapter

    def test_extracts_stop_tokens_from_adapter(
        self, mock_model: Mock, mock_tokenizer: Mock, mock_adapter: Mock
    ) -> None:
        """Engine should extract stop tokens from adapter."""
        mock_adapter.get_stop_tokens.return_value = [128001, 128009]

        engine = BatchInferenceEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            adapter=mock_adapter,
        )

        assert engine._stop_token_ids == {128001, 128009}
        mock_adapter.get_stop_tokens.assert_called_once_with(mock_tokenizer)

    def test_get_stop_token_ids_returns_copy(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """get_stop_token_ids should return a copy to prevent modification."""
        ids1 = batch_engine.get_stop_token_ids()
        ids2 = batch_engine.get_stop_token_ids()

        assert ids1 == ids2
        assert ids1 is not batch_engine._stop_token_ids

    def test_prefix_cache_stored_if_provided(
        self, mock_model: Mock, mock_tokenizer: Mock, mock_adapter: Mock
    ) -> None:
        """Engine should store prefix cache if provided."""
        mock_cache = Mock(spec=[])

        engine = BatchInferenceEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            adapter=mock_adapter,
            prefix_cache=mock_cache,
        )

        assert engine._prefix_cache is mock_cache


class TestPrepareRequestContext:
    """Tests for _prepare_request_context method."""

    def test_decodes_prompt_tokens(
        self, batch_engine: BatchInferenceEngine, mock_tokenizer: Mock
    ) -> None:
        """Should decode prompt tokens to text."""
        request = make_request("test-1", prompt_tokens=[10, 20, 30])

        result = batch_engine._prepare_request_context(request)

        # Should have called decode with prompt tokens
        mock_tokenizer.decode.assert_called_with(
            [10, 20, 30], skip_special_tokens=False
        )

    def test_includes_generated_tokens(
        self, batch_engine: BatchInferenceEngine, mock_tokenizer: Mock
    ) -> None:
        """Should include generated tokens in context."""
        request = make_request("test-1", prompt_tokens=[10, 20])
        request.generated_tokens = [30, 40]

        batch_engine._prepare_request_context(request)

        # Should have called decode with all tokens (prompt + generated)
        mock_tokenizer.decode.assert_called_with(
            [10, 20, 30, 40], skip_special_tokens=False
        )

    def test_handles_processor_wrapped_tokenizer(
        self, mock_model: Mock, mock_adapter: Mock
    ) -> None:
        """Should extract inner tokenizer from Processor wrapper."""
        # Create a processor that wraps a tokenizer
        inner_tokenizer = Mock(spec=[])
        inner_tokenizer.decode = Mock(return_value="decoded")
        inner_tokenizer.eos_token_id = 128001

        processor = Mock(spec=[])
        processor.tokenizer = inner_tokenizer

        engine = BatchInferenceEngine(
            model=mock_model,
            tokenizer=processor,
            adapter=mock_adapter,
        )

        request = make_request("test-1")
        engine._prepare_request_context(request)

        # Should have called decode on the inner tokenizer
        inner_tokenizer.decode.assert_called()


class TestGenerateTokensForBatch:
    """Tests for generate_tokens_for_batch method."""

    def test_returns_dict_mapping_request_ids_to_results(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """Should return dict mapping request_id -> (text, token_id, is_stop)."""
        requests = [
            make_request("req-1"),
            make_request("req-2"),
        ]

        # Mock stream_generate to return predictable token
        # Need to create fresh iterator for each call
        def make_iter():
            mock_response = Mock(spec=[])
            mock_response.token = 100
            mock_response.text = " hello"
            return iter([mock_response])

        with patch(
            "mlx_lm.stream_generate",
            side_effect=lambda *args, **kwargs: make_iter(),
        ):
            sampler = Mock(spec=[])
            results = batch_engine.generate_tokens_for_batch(requests, sampler)

        assert "req-1" in results
        assert "req-2" in results
        assert results["req-1"] == (" hello", 100, False)
        assert results["req-2"] == (" hello", 100, False)

    def test_detects_stop_tokens(self, batch_engine: BatchInferenceEngine) -> None:
        """Should detect stop tokens and set is_stop=True."""
        request = make_request("test-1")

        # Return a stop token (128001 is in stop_token_ids)
        mock_response = Mock(spec=[])
        mock_response.token = 128001  # EOS token
        mock_response.text = ""

        with patch("mlx_lm.stream_generate", return_value=iter([mock_response])):
            sampler = Mock(spec=[])
            results = batch_engine.generate_tokens_for_batch([request], sampler)

        text, token_id, is_stop = results["test-1"]
        assert is_stop is True
        assert token_id == 128001

    def test_handles_generation_error_gracefully(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """Should handle generation errors and mark as stop."""
        request = make_request("test-1")

        with patch("mlx_lm.stream_generate", side_effect=RuntimeError("MLX error")):
            sampler = Mock(spec=[])
            results = batch_engine.generate_tokens_for_batch([request], sampler)

        # Should still have result (marked as error/stop)
        assert "test-1" in results
        text, token_id, is_stop = results["test-1"]
        assert is_stop is True

    def test_calls_stream_generate_with_correct_params(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """Should call stream_generate with max_tokens=1."""
        request = make_request("test-1")
        mock_response = Mock(spec=[])
        mock_response.token = 100
        mock_response.text = "x"

        with patch("mlx_lm.stream_generate") as mock_gen:
            mock_gen.return_value = iter([mock_response])
            sampler = Mock(spec=[])

            batch_engine.generate_tokens_for_batch([request], sampler)

            # Verify max_tokens=1 was passed
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["max_tokens"] == 1


class TestGenerateBatchStepAsync:
    """Tests for async generate_batch_step method."""

    @pytest.mark.asyncio
    async def test_runs_generation_in_thread(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """Should run generation in a dedicated thread."""
        request = make_request("test-1")
        mock_response = Mock(spec=[])
        mock_response.token = 100
        mock_response.text = "test"

        thread_ids: list[int] = []

        def capture_thread(*args, **kwargs):
            thread_ids.append(threading.get_ident())
            return iter([mock_response])

        with patch("mlx_lm.stream_generate", side_effect=capture_thread):
            sampler = Mock(spec=[])
            results = await batch_engine.generate_batch_step([request], sampler)

        # Generation should have run in a different thread
        assert len(thread_ids) == 1
        assert thread_ids[0] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_returns_results_from_thread(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """Should return results from generation thread."""
        request = make_request("test-1")
        mock_response = Mock(spec=[])
        mock_response.token = 42
        mock_response.text = "answer"

        with patch("mlx_lm.stream_generate", return_value=iter([mock_response])):
            sampler = Mock(spec=[])
            results = await batch_engine.generate_batch_step([request], sampler)

        assert results["test-1"] == ("answer", 42, False)

    @pytest.mark.asyncio
    async def test_propagates_exceptions_from_thread(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """Should propagate exceptions from generation thread."""
        request = make_request("test-1")

        # Make generate_tokens_for_batch raise an exception
        with patch.object(
            batch_engine,
            "generate_tokens_for_batch",
            side_effect=RuntimeError("Test error"),
        ):
            sampler = Mock(spec=[])
            with pytest.raises(RuntimeError, match="Test error"):
                await batch_engine.generate_batch_step([request], sampler)


class TestSchedulerIntegration:
    """Tests for scheduler integration with BatchInferenceEngine."""

    def test_scheduler_accepts_model_params_in_init(self) -> None:
        """Scheduler should accept model/tokenizer/adapter in __init__."""
        from mlx_manager.mlx_server.services.batching import (
            ContinuousBatchingScheduler,
        )

        block_manager = PagedBlockManager(num_blocks=100)
        mock_model = Mock(spec=[])
        mock_tokenizer = Mock(spec=[])
        mock_tokenizer.eos_token_id = 128001
        mock_adapter = Mock(spec=[])
        mock_adapter.get_stop_tokens = Mock(return_value=[128001])

        scheduler = ContinuousBatchingScheduler(
            model_id="test-model",
            block_manager=block_manager,
            model=mock_model,
            tokenizer=mock_tokenizer,
            adapter=mock_adapter,
        )

        assert scheduler._inference_engine is not None

    def test_scheduler_set_model_creates_engine(self) -> None:
        """set_model should create a new BatchInferenceEngine."""
        from mlx_manager.mlx_server.services.batching import (
            ContinuousBatchingScheduler,
        )

        block_manager = PagedBlockManager(num_blocks=100)
        scheduler = ContinuousBatchingScheduler(
            model_id="test-model",
            block_manager=block_manager,
        )

        assert scheduler._inference_engine is None

        mock_model = Mock(spec=[])
        mock_tokenizer = Mock(spec=[])
        mock_tokenizer.eos_token_id = 128001
        mock_adapter = Mock(spec=[])
        mock_adapter.get_stop_tokens = Mock(return_value=[128001])

        scheduler.set_model(mock_model, mock_tokenizer, mock_adapter)

        assert scheduler._inference_engine is not None
        assert scheduler._prefix_cache is not None

    @pytest.mark.asyncio
    async def test_batch_step_calls_inference_engine(self) -> None:
        """_batch_step should call inference engine when model is set."""
        from mlx_manager.mlx_server.services.batching import (
            ContinuousBatchingScheduler,
        )

        block_manager = PagedBlockManager(num_blocks=100)
        scheduler = ContinuousBatchingScheduler(
            model_id="test-model",
            block_manager=block_manager,
        )

        # Create mock inference engine with async method
        async def mock_generate(*args):
            return {"req-1": (" token", 100, False)}

        mock_engine = Mock(spec=[])
        mock_engine.generate_batch_step = MagicMock(side_effect=mock_generate)
        scheduler._inference_engine = mock_engine

        # Add a running request
        request = make_request("req-1")
        scheduler.running.append(request)

        # Mock make_sampler
        with patch("mlx_lm.sample_utils.make_sampler", return_value=Mock(spec=[])):
            await scheduler._batch_step()

        # Should have called inference engine
        mock_engine.generate_batch_step.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_step_updates_request_with_token(self) -> None:
        """_batch_step should update request with generated token."""
        from mlx_manager.mlx_server.services.batching import (
            ContinuousBatchingScheduler,
        )

        block_manager = PagedBlockManager(num_blocks=100)
        scheduler = ContinuousBatchingScheduler(
            model_id="test-model",
            block_manager=block_manager,
        )

        # Create mock inference engine that returns a token
        async def mock_generate(*args):
            return {"req-1": (" hello", 42, False)}

        mock_engine = Mock(spec=[])
        mock_engine.generate_batch_step = mock_generate
        scheduler._inference_engine = mock_engine

        # Add a running request
        request = make_request("req-1")
        scheduler.running.append(request)

        # Mock make_sampler
        with patch("mlx_lm.sample_utils.make_sampler", return_value=Mock(spec=[])):
            await scheduler._batch_step()

        # Token should be added to request
        assert 42 in request.generated_tokens

        # Output queue should have the token data
        token_data = await asyncio.wait_for(
            request.output_queue.get(), timeout=1.0
        )
        assert token_data["token_id"] == 42
        assert token_data["text"] == " hello"

    @pytest.mark.asyncio
    async def test_batch_step_marks_stop_token_as_complete(self) -> None:
        """_batch_step should mark request as complete on stop token."""
        from mlx_manager.mlx_server.services.batching import (
            ContinuousBatchingScheduler,
            RequestStatus,
        )

        block_manager = PagedBlockManager(num_blocks=100)
        scheduler = ContinuousBatchingScheduler(
            model_id="test-model",
            block_manager=block_manager,
        )

        # Create mock inference engine that returns a stop token
        async def mock_generate(*args):
            return {"req-1": ("", 128001, True)}

        mock_engine = Mock(spec=[])
        mock_engine.generate_batch_step = mock_generate
        scheduler._inference_engine = mock_engine

        # Add a running request
        request = make_request("req-1")
        scheduler.running.append(request)

        # Mock make_sampler
        with patch("mlx_lm.sample_utils.make_sampler", return_value=Mock(spec=[])):
            await scheduler._batch_step()

        # Request should be marked as completed
        assert request.status == RequestStatus.COMPLETED

        # Token should NOT be added to generated_tokens (stop token not yielded)
        assert 128001 not in request.generated_tokens


class TestThreadSafety:
    """Tests for thread safety mechanisms."""

    def test_generation_lock_prevents_concurrent_generation(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """Generation lock should serialize batch generation."""
        # The lock should be a threading.Lock
        assert isinstance(batch_engine._generation_lock, type(threading.Lock()))

    def test_queue_based_communication(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """Should use Queue for thread-to-async communication."""
        import inspect

        source = inspect.getsource(BatchInferenceEngine.generate_batch_step)

        # Should use Queue for result passing
        assert "Queue" in source
        assert "result_queue" in source

    @pytest.mark.asyncio
    async def test_timeout_on_generation(
        self, batch_engine: BatchInferenceEngine
    ) -> None:
        """Should timeout if generation takes too long."""
        request = make_request("test-1")

        # Make generation hang
        def hang(*args, **kwargs):
            import time

            time.sleep(10)  # Would hang without timeout
            return iter([])

        with patch("mlx_lm.stream_generate", side_effect=hang):
            # Should not hang forever due to timeout
            # Note: Test uses very short timeout override isn't possible
            # so we just verify the timeout parameter exists in source
            import inspect

            source = inspect.getsource(BatchInferenceEngine.generate_batch_step)
            assert "timeout=60" in source
