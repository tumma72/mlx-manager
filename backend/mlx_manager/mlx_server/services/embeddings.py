"""Embeddings generation service.

Delegates core generation logic to the model adapter.
This service handles pool access, logging, and observability.
"""

from loguru import logger

from mlx_manager.mlx_server.models.ir import EmbeddingResult

try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


async def generate_embeddings(
    model_id: str,
    texts: list[str],
) -> EmbeddingResult:
    """Generate embeddings for a list of texts."""
    from mlx_manager.mlx_server.models.pool import get_model_pool

    pool = get_model_pool()
    loaded = await pool.get_model(model_id)
    assert loaded.adapter is not None, f"No adapter for model {model_id}"

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
        result: EmbeddingResult = await loaded.adapter.generate_embeddings(loaded.model, texts)

        logger.info(
            f"Embeddings complete: model={model_id}, "
            f"vectors={len(result.embeddings)}, tokens={result.total_tokens}"
        )

        if LOGFIRE_AVAILABLE:
            logfire.info(
                "embeddings_finished",
                model=model_id,
                num_embeddings=len(result.embeddings),
                total_tokens=result.total_tokens,
            )

        return result

    finally:
        if span_context:
            span_context.__exit__(None, None, None)
