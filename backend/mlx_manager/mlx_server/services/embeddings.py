"""Embeddings generation service using mlx-embeddings.

CRITICAL: This module uses run_on_metal_thread utility to respect
MLX Metal thread affinity requirements.
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
    """Generate embeddings for a list of texts.

    Args:
        model_id: HuggingFace model ID (must be an embeddings model)
        texts: List of strings to embed

    Returns:
        EmbeddingResult with embedding vectors, dimensions, and token count

    Raises:
        RuntimeError: If model loading or generation fails
    """
    from mlx_manager.mlx_server.models.pool import get_model_pool
    from mlx_manager.mlx_server.utils.metal import run_on_metal_thread

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

        def run_embeddings() -> tuple[list[list[float]], int]:
            """Run embedding generation in dedicated thread (owns Metal context)."""
            import mlx.core as mx

            # Tokenize batch
            # Use inner tokenizer's __call__ for batch encoding.
            # TokenizerWrapper from mlx-embeddings is not callable and
            # batch_encode_plus was removed in transformers v5.
            inner_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
            encoded = inner_tokenizer(
                texts,
                return_tensors=None,
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: mx.array(v) for k, v in encoded.items()}

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

            return (embeddings_list, total_tokens)

        embeddings_list, total_tokens = await run_on_metal_thread(
            run_embeddings, error_context="Embeddings generation failed"
        )

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

        return EmbeddingResult(
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            total_tokens=total_tokens,
            finish_reason="stop",
        )

    finally:
        if span_context:
            span_context.__exit__(None, None, None)
