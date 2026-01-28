"""Embeddings generation service using mlx-embeddings.

CRITICAL: This module uses the same queue-based threading pattern as inference.py
to respect MLX Metal thread affinity requirements.
"""

import asyncio
import logging
import threading
from queue import Queue

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

logger = logging.getLogger(__name__)


async def generate_embeddings(
    model_id: str,
    texts: list[str],
) -> tuple[list[list[float]], int]:
    """Generate embeddings for a list of texts.

    Args:
        model_id: HuggingFace model ID (must be an embeddings model)
        texts: List of strings to embed

    Returns:
        Tuple of (list of embedding vectors, total token count)

    Raises:
        RuntimeError: If model loading or generation fails
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool

    # Get model from pool
    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    model = loaded.model
    tokenizer = loaded.tokenizer

    logger.info(f"Generating embeddings: model={model_id}, texts={len(texts)}")

    # LogFire span
    span_context = None
    if LOGFIRE_AVAILABLE:
        span_context = logfire.span(
            "embeddings",
            model=model_id,
            num_texts=len(texts),
        )
        span_context.__enter__()

    try:
        # Queue for passing result from generation thread
        result_queue: Queue[tuple[list[list[float]], int] | Exception] = Queue()

        def run_embeddings() -> None:
            """Run embedding generation in dedicated thread (owns Metal context)."""
            try:
                import mlx.core as mx

                # Tokenize batch
                # mlx-embeddings tokenizer uses batch_encode_plus
                inputs = tokenizer.batch_encode_plus(
                    texts,
                    return_tensors="mlx",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                # Count tokens per input (not padded batch)
                total_tokens = 0
                for text in texts:
                    tokens = tokenizer.encode(text, truncation=True, max_length=512)
                    total_tokens += len(tokens)

                # Forward pass
                outputs = model(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )

                # text_embeds are ALREADY L2-normalized (mean pooled + normalized)
                embeddings = outputs.text_embeds

                # Convert to Python lists
                # NOTE: mx.eval() is MLX framework's tensor evaluation (NOT Python eval())
                # It ensures computation is complete before converting to Python lists
                mx.eval(embeddings)
                embeddings_list = embeddings.tolist()

                result_queue.put((embeddings_list, total_tokens))

            except Exception as e:
                result_queue.put(e)

        # Start generation thread
        gen_thread = threading.Thread(target=run_embeddings, daemon=True)
        gen_thread.start()

        # Wait for result (with 5 minute timeout)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: result_queue.get(timeout=300))

        gen_thread.join(timeout=1.0)

        # Check for exception
        if isinstance(result, Exception):
            raise RuntimeError(f"Embeddings generation failed: {result}") from result

        embeddings_list, total_tokens = result

        logger.info(
            f"Embeddings complete: model={model_id}, "
            f"vectors={len(embeddings_list)}, tokens={total_tokens}"
        )

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "embeddings_finished",
                model=model_id,
                num_embeddings=len(embeddings_list),
                total_tokens=total_tokens,
            )

        return embeddings_list, total_tokens

    finally:
        if span_context:
            span_context.__exit__(None, None, None)

        # Clear cache after embeddings
        from mlx_manager.mlx_server.utils.memory import clear_cache

        clear_cache()
