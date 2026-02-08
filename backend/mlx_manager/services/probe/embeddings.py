"""Embeddings model probe strategy.

Tests vector dimensions, L2 normalization, max sequence length,
and validates similarity ordering with known test pairs.
"""

from __future__ import annotations

import math
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from loguru import logger

from mlx_manager.mlx_server.models.types import ModelType

from .steps import ProbeResult, ProbeStep

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel


class EmbeddingsProbe:
    """Probe strategy for embedding models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.EMBEDDINGS

    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        result.model_type = ModelType.EMBEDDINGS

        # Step 1: Generate a test embedding and measure dimensions
        yield ProbeStep(step="test_encode", status="running")
        try:
            embedding = await _encode_text(loaded, "hello")
            if embedding is not None:
                result.embedding_dimensions = len(embedding)
                yield ProbeStep(
                    step="test_encode",
                    status="completed",
                    capability="embedding_dimensions",
                    value=len(embedding),
                )
            else:
                yield ProbeStep(
                    step="test_encode",
                    status="failed",
                    error="Encoding returned empty result",
                )
        except Exception as e:
            logger.warning(f"Encode test failed for {model_id}: {e}")
            yield ProbeStep(step="test_encode", status="failed", error=str(e))
            return  # Can't continue if encoding fails

        # Step 2: Check L2 normalization
        yield ProbeStep(step="check_normalization", status="running")
        try:
            if embedding is not None:
                norm = math.sqrt(sum(x * x for x in embedding))
                is_normalized = abs(norm - 1.0) < 0.01
                result.is_normalized = is_normalized
                yield ProbeStep(
                    step="check_normalization",
                    status="completed",
                    capability="is_normalized",
                    value=is_normalized,
                )
        except Exception as e:
            logger.warning(f"Normalization check failed for {model_id}: {e}")
            yield ProbeStep(step="check_normalization", status="failed", error=str(e))

        # Step 3: Read max sequence length from config
        yield ProbeStep(step="check_max_length", status="running")
        try:
            max_len = _get_max_sequence_length(model_id)
            result.max_sequence_length = max_len
            yield ProbeStep(
                step="check_max_length",
                status="completed",
                capability="max_sequence_length",
                value=max_len,
            )
        except Exception as e:
            logger.warning(f"Max length check failed for {model_id}: {e}")
            yield ProbeStep(step="check_max_length", status="failed", error=str(e))

        # Step 4: Validate similarity ordering
        yield ProbeStep(step="test_similarity", status="running")
        try:
            similarity_ok = await _test_similarity_ordering(loaded)
            yield ProbeStep(
                step="test_similarity",
                status="completed" if similarity_ok else "failed",
                capability="similarity_valid",
                value=similarity_ok,
                error=None if similarity_ok else "Similarity ordering incorrect",
            )
        except Exception as e:
            logger.warning(f"Similarity test failed for {model_id}: {e}")
            yield ProbeStep(step="test_similarity", status="failed", error=str(e))


async def _encode_text(loaded: LoadedModel, text: str) -> list[float] | None:
    """Encode a single text string and return the embedding vector."""
    from mlx_manager.mlx_server.services.embeddings import generate_embeddings

    embeddings, _ = await generate_embeddings(loaded.model_id, [text])
    if embeddings and len(embeddings) > 0:
        return embeddings[0]
    return None


def _get_max_sequence_length(model_id: str) -> int | None:
    """Read max sequence length from model config."""
    from mlx_manager.utils.model_detection import read_model_config

    config = read_model_config(model_id)
    if not config:
        return None

    return config.get(
        "max_position_embeddings",
        config.get("max_seq_length", config.get("max_sequence_length")),
    )


async def _test_similarity_ordering(loaded: LoadedModel) -> bool:
    """Validate that semantically similar texts have higher cosine similarity.

    Tests: sim("cat", "kitten") > sim("cat", "airplane")
    """
    from mlx_manager.mlx_server.services.embeddings import generate_embeddings

    texts = ["cat", "kitten", "airplane"]
    embeddings, _ = await generate_embeddings(loaded.model_id, texts)

    if len(embeddings) != 3:
        return False

    def cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    sim_related = cosine_sim(embeddings[0], embeddings[1])  # cat vs kitten
    sim_unrelated = cosine_sim(embeddings[0], embeddings[2])  # cat vs airplane

    logger.debug(f"Similarity test: cat-kitten={sim_related:.3f}, cat-airplane={sim_unrelated:.3f}")

    return sim_related > sim_unrelated
