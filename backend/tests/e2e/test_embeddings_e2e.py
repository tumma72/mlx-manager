"""End-to-end tests for embeddings model inference.

Tests validate the complete pipeline:
1. HTTP request to /v1/embeddings
2. Model type detection (EMBEDDINGS)
3. Model loading via mlx-embeddings
4. Vector generation and L2 normalization
5. OpenAI-compatible response format

Run:
  pytest -m e2e_embeddings -v
"""

import math

import pytest


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_embedding_request(
    model: str,
    input_text: str | list[str],
) -> dict:
    """Build an OpenAI-compatible embedding request."""
    return {
        "model": model,
        "input": input_text,
    }


# --------------------------------------------------
# Single embedding tests
# --------------------------------------------------


@pytest.mark.e2e
@pytest.mark.e2e_embeddings
class TestSingleEmbedding:
    """Test single text embedding generation."""

    async def test_single_embedding_response(self, app_client, embeddings_model):
        """Single text input should return a valid embedding vector."""
        request = build_embedding_request(embeddings_model, "Hello world")
        response = await app_client.post("/v1/embeddings", json=request)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert data["object"] == "list"
        assert data["model"] == embeddings_model
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["index"] == 0

        # Validate embedding vector
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0, "Embedding vector should not be empty"
        assert all(isinstance(v, float) for v in embedding), "All values should be floats"

    async def test_embedding_dimensionality(self, app_client, embeddings_model):
        """all-MiniLM-L6-v2 should produce 384-dimensional embeddings."""
        request = build_embedding_request(embeddings_model, "Test dimensionality")
        response = await app_client.post("/v1/embeddings", json=request)

        assert response.status_code == 200
        embedding = response.json()["data"][0]["embedding"]
        assert len(embedding) == 384, (
            f"Expected 384 dimensions for MiniLM, got {len(embedding)}"
        )

    async def test_embedding_is_normalized(self, app_client, embeddings_model):
        """Embeddings should be L2-normalized (unit vectors)."""
        request = build_embedding_request(embeddings_model, "Check normalization")
        response = await app_client.post("/v1/embeddings", json=request)

        assert response.status_code == 200
        embedding = response.json()["data"][0]["embedding"]

        # L2 norm should be ~1.0
        l2_norm = math.sqrt(sum(v * v for v in embedding))
        assert abs(l2_norm - 1.0) < 0.01, f"Expected L2 norm ~1.0, got {l2_norm}"

    async def test_usage_stats(self, app_client, embeddings_model):
        """Response should include usage statistics."""
        request = build_embedding_request(embeddings_model, "Usage test")
        response = await app_client.post("/v1/embeddings", json=request)

        assert response.status_code == 200
        usage = response.json()["usage"]
        assert "prompt_tokens" in usage
        assert "total_tokens" in usage
        assert usage["prompt_tokens"] > 0
        assert usage["total_tokens"] > 0


# --------------------------------------------------
# Batch embedding tests
# --------------------------------------------------


@pytest.mark.e2e
@pytest.mark.e2e_embeddings
class TestBatchEmbeddings:
    """Test batch embedding generation."""

    async def test_batch_returns_multiple(self, app_client, embeddings_model):
        """Batch input should return one embedding per text."""
        texts = [
            "The quick brown fox",
            "jumped over the lazy dog",
            "A completely different sentence about space",
        ]
        request = build_embedding_request(embeddings_model, texts)
        response = await app_client.post("/v1/embeddings", json=request)

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3

        # Each should have correct index
        for i, item in enumerate(data["data"]):
            assert item["index"] == i
            assert len(item["embedding"]) == 384

    async def test_batch_consistent_dimensionality(self, app_client, embeddings_model):
        """All embeddings in a batch should have the same dimensionality."""
        texts = ["Text one", "Text two", "Text three", "Text four"]
        request = build_embedding_request(embeddings_model, texts)
        response = await app_client.post("/v1/embeddings", json=request)

        assert response.status_code == 200
        dimensions = [len(item["embedding"]) for item in response.json()["data"]]
        assert len(set(dimensions)) == 1, f"Inconsistent dimensions: {dimensions}"


# --------------------------------------------------
# Semantic similarity tests
# --------------------------------------------------


@pytest.mark.e2e
@pytest.mark.e2e_embeddings
class TestSemanticSimilarity:
    """Test that embeddings capture semantic similarity."""

    async def test_similar_texts_higher_similarity(self, app_client, embeddings_model):
        """Similar texts should have higher cosine similarity than dissimilar ones."""
        texts = [
            "The cat sat on the mat",  # [0] - about a cat
            "A kitten was resting on a rug",  # [1] - similar to [0]
            "Python is a programming language",  # [2] - completely different
        ]
        request = build_embedding_request(embeddings_model, texts)
        response = await app_client.post("/v1/embeddings", json=request)

        assert response.status_code == 200
        embeddings = [item["embedding"] for item in response.json()["data"]]

        sim_similar = cosine_similarity(embeddings[0], embeddings[1])
        sim_different = cosine_similarity(embeddings[0], embeddings[2])

        assert sim_similar > sim_different, (
            f"Similar texts similarity ({sim_similar:.4f}) should be > "
            f"dissimilar texts ({sim_different:.4f})"
        )

    async def test_identical_texts_high_similarity(self, app_client, embeddings_model):
        """Identical texts should have cosine similarity ~1.0."""
        texts = [
            "This is a test sentence",
            "This is a test sentence",
        ]
        request = build_embedding_request(embeddings_model, texts)
        response = await app_client.post("/v1/embeddings", json=request)

        assert response.status_code == 200
        embeddings = [item["embedding"] for item in response.json()["data"]]

        sim = cosine_similarity(embeddings[0], embeddings[1])
        assert sim > 0.99, f"Identical texts should have similarity ~1.0, got {sim:.4f}"


# --------------------------------------------------
# Error handling tests
# --------------------------------------------------


@pytest.mark.e2e
@pytest.mark.e2e_embeddings
class TestEmbeddingsErrorHandling:
    """Test error handling for embeddings endpoint."""

    async def test_text_model_rejected(self, app_client):
        """Text generation model should be rejected for embeddings."""
        request = build_embedding_request(
            "mlx-community/Qwen3-0.6B-4bit-DWQ",
            "This should fail",
        )
        response = await app_client.post("/v1/embeddings", json=request)
        assert response.status_code == 400

    async def test_empty_input_rejected(self, app_client, embeddings_model):
        """Empty input should be rejected."""
        request = build_embedding_request(embeddings_model, [])
        response = await app_client.post("/v1/embeddings", json=request)
        assert response.status_code == 400
