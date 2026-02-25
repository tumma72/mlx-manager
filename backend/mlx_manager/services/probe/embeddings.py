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

from .base import BaseProbe, get_model_config_value
from .steps import ProbeResult, ProbeStep, probe_step

if TYPE_CHECKING:
    from mlx_manager.mlx_server.models.pool import LoadedModel


class EmbeddingsProbe(BaseProbe):
    """Probe strategy for embedding models."""

    @property
    def model_type(self) -> ModelType:
        return ModelType.EMBEDDINGS

    async def probe(
        self, model_id: str, loaded: LoadedModel, result: ProbeResult
    ) -> AsyncGenerator[ProbeStep, None]:
        result.model_type = ModelType.EMBEDDINGS

        # Step 1: Generate a test embedding and measure dimensions
        async with probe_step("test_encode", "embedding_dimensions") as ctx:
            yield ctx.running
            embedding = await _encode_text(loaded, "hello")
            if embedding is not None:
                result.embedding_dimensions = len(embedding)
                ctx.value = len(embedding)
            else:
                ctx.fail("Encoding returned empty result")
        yield ctx.result
        if ctx._failed:
            return  # Can't continue if encoding fails

        # Step 2: Check L2 normalization
        async with probe_step("check_normalization", "is_normalized") as ctx:
            yield ctx.running
            if embedding is not None:
                norm = math.sqrt(sum(x * x for x in embedding))
                is_normalized = abs(norm - 1.0) < 0.01
                result.is_normalized = is_normalized
                ctx.value = is_normalized
        yield ctx.result

        # Step 3: Read max sequence length from config
        async with probe_step("check_max_length", "max_sequence_length") as ctx:
            yield ctx.running
            max_len = _get_max_sequence_length(model_id)
            result.max_sequence_length = max_len
            ctx.value = max_len
        yield ctx.result

        # Step 4: Validate similarity ordering
        async with probe_step("test_similarity", "similarity_valid") as ctx:
            yield ctx.running
            similarity_ok = await _test_similarity_ordering(loaded)
            ctx.value = similarity_ok
            if not similarity_ok:
                ctx.fail("Similarity ordering incorrect")
        yield ctx.result


async def _encode_text(loaded: LoadedModel, text: str) -> list[float] | None:
    """Encode a single text string and return the embedding vector."""
    adapter = loaded.adapter
    if adapter is None:
        msg = "No adapter available for embeddings"
        raise RuntimeError(msg)

    result = await adapter.generate_embeddings(loaded.model, [text])
    if result.embeddings and len(result.embeddings) > 0:
        return result.embeddings[0]
    return None


def _get_max_sequence_length(model_id: str) -> int | None:
    """Read max sequence length from model config."""
    from typing import cast

    result = get_model_config_value(
        model_id,
        "max_position_embeddings",
        "max_seq_length",
        "max_sequence_length",
    )
    return cast(int | None, result)


async def _test_similarity_ordering(loaded: LoadedModel) -> bool:
    """Validate that semantically similar texts have higher cosine similarity.

    Tests: sim("cat", "kitten") > sim("cat", "airplane")
    """
    adapter = loaded.adapter
    if adapter is None:
        return False

    texts = ["cat", "kitten", "airplane"]
    result = await adapter.generate_embeddings(loaded.model, texts)
    embeddings = result.embeddings

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
