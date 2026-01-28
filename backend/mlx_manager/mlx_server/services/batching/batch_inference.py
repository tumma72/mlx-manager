"""Batch inference engine for MLX model generation.

This module implements the BatchInferenceEngine for multi-request token
generation within the continuous batching scheduler.

CRITICAL: Uses Queue-based threading pattern for MLX Metal affinity.
MLX Metal operations have thread affinity requirements - all generation
must happen in a dedicated thread that owns the Metal context.

NOTE: mlx-lm doesn't support true batched generation yet (Issue #548).
For now, we generate sequentially within a single thread to maintain
Metal context. Future optimization: native batch KV cache when mlx-lm
supports it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.adapters.base import ModelAdapter
    from mlx_manager.mlx_server.services.batching.prefix_cache import PrefixCache
    from mlx_manager.mlx_server.services.batching.request import BatchRequest

logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """Batch token generation engine with MLX Metal thread affinity.

    Generates one token for each request in a batch while maintaining
    isolated state per request. Uses a dedicated thread for MLX generation
    to respect Metal GPU thread affinity requirements.

    Attributes:
        model: The MLX model for generation
        tokenizer: HuggingFace tokenizer for encoding/decoding
        adapter: Model adapter for stop tokens
        prefix_cache: Optional prefix cache for KV block sharing
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        adapter: ModelAdapter,
        prefix_cache: PrefixCache | None = None,
    ) -> None:
        """Initialize the batch inference engine.

        Args:
            model: MLX model instance
            tokenizer: HuggingFace tokenizer
            adapter: Model adapter for stop token detection
            prefix_cache: Optional prefix cache for KV block reuse
        """
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self._prefix_cache = prefix_cache

        # Extract stop token IDs from adapter
        self._stop_token_ids: set[int] = set(adapter.get_stop_tokens(tokenizer))

        # Thread safety for generation (only one batch can generate at a time)
        self._generation_lock = threading.Lock()

        logger.info(
            f"BatchInferenceEngine initialized: "
            f"stop_tokens={self._stop_token_ids}, "
            f"prefix_cache={'enabled' if prefix_cache else 'disabled'}"
        )

    def _prepare_request_context(self, request: BatchRequest) -> str:
        """Prepare prompt text for a request.

        Decodes the prompt tokens plus any generated tokens back to text
        for feeding into the next generation step.

        Args:
            request: The batch request to prepare context for

        Returns:
            The full prompt text (original prompt + generated so far)
        """
        # Get actual tokenizer (Processor wraps tokenizer, regular tokenizer is itself)
        actual_tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)

        # Combine prompt and generated tokens
        all_tokens = request.prompt_tokens + request.generated_tokens

        # Decode to text
        result: str = actual_tokenizer.decode(all_tokens, skip_special_tokens=False)
        return result

    def generate_tokens_for_batch(
        self,
        requests: list[BatchRequest],
        sampler: Any,
    ) -> dict[str, tuple[str, int, bool]]:
        """Generate one token for each request in batch.

        CRITICAL: This runs in a dedicated thread to maintain MLX Metal
        context. All generation operations must happen in this thread.

        NOTE: Currently generates sequentially since mlx-lm doesn't support
        true batched generation (Issue #548). The single-thread approach
        still maintains Metal context consistency.

        Args:
            requests: List of BatchRequest objects to generate for
            sampler: mlx-lm sampler function (from make_sampler)

        Returns:
            Dictionary mapping request_id -> (token_text, token_id, is_stop)
        """
        from mlx_lm import stream_generate

        results: dict[str, tuple[str, int, bool]] = {}

        with self._generation_lock:
            for request in requests:
                try:
                    # Prepare prompt text
                    prompt = self._prepare_request_context(request)

                    # Generate just one token
                    # stream_generate yields tokens - we just take the first one
                    for response in stream_generate(
                        self.model,
                        self.tokenizer,
                        prompt,
                        max_tokens=1,  # Only generate 1 token
                        sampler=sampler,
                    ):
                        raw_token_id = getattr(response, "token", None)
                        token_text: str = getattr(response, "text", str(response))
                        token_id: int = raw_token_id if raw_token_id is not None else 0

                        # Check for stop token
                        is_stop = raw_token_id is not None and raw_token_id in self._stop_token_ids

                        results[request.request_id] = (token_text, token_id, is_stop)
                        break  # Only take first token

                    # If no tokens generated (e.g., empty prompt), mark as stop
                    if request.request_id not in results:
                        results[request.request_id] = ("", 0, True)
                        logger.warning(f"No token generated for request {request.request_id}")

                except Exception as e:
                    logger.error(f"Generation error for request {request.request_id}: {e}")
                    # Mark as stop on error
                    results[request.request_id] = ("", 0, True)

        return results

    async def generate_batch_step(
        self,
        requests: list[BatchRequest],
        sampler: Any,
    ) -> dict[str, tuple[str, int, bool]]:
        """Generate one token for each request in batch (async wrapper).

        Uses Queue-based threading pattern to maintain MLX Metal context
        while providing async interface for the scheduler.

        CRITICAL: Must use dedicated Thread + Queue pattern.
        Using run_in_executor with next(generator) does NOT work
        because MLX Metal operations have thread affinity requirements.

        Args:
            requests: List of BatchRequest objects to generate for
            sampler: mlx-lm sampler function (from make_sampler)

        Returns:
            Dictionary mapping request_id -> (token_text, token_id, is_stop)

        Raises:
            Exception: If generation fails in the worker thread
        """
        # Queue for passing results from generation thread to async context
        result_queue: Queue[dict[str, tuple[str, int, bool]] | Exception] = Queue()

        def run_generation() -> None:
            """Run batch generation in dedicated thread (owns Metal context)."""
            try:
                results = self.generate_tokens_for_batch(requests, sampler)
                result_queue.put(results)
            except Exception as e:
                result_queue.put(e)

        # Start generation in dedicated thread (daemon=True so it doesn't block exit)
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # Wait for result asynchronously using run_in_executor for queue.get
        loop = asyncio.get_running_loop()

        try:
            # Poll queue with timeout to avoid blocking event loop forever
            result = await loop.run_in_executor(None, lambda: result_queue.get(timeout=60))
        except Empty:
            raise TimeoutError("Batch generation timed out after 60 seconds")

        # Wait for thread to finish
        gen_thread.join(timeout=1.0)

        # Check for exception from generation thread
        if isinstance(result, Exception):
            raise result

        return result

    def get_stop_token_ids(self) -> set[int]:
        """Get the set of stop token IDs.

        Returns:
            Set of token IDs that signal generation should stop
        """
        return self._stop_token_ids.copy()
