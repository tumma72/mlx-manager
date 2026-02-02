"""Tests for prefix caching with KV block sharing.

Tests cover:
- Hash computation (determinism, chaining, position sensitivity)
- Prefix lookup (cache hits, partial matches, misses)
- Caching behavior (deduplication, block marking)
- Reference counting (increment, decrement, accumulation)
- Eviction (unused blocks, in-use preservation)
- Per-model isolation
"""

from mlx_manager.mlx_server.services.batching.block import BLOCK_SIZE
from mlx_manager.mlx_server.services.batching.block_manager import PagedBlockManager
from mlx_manager.mlx_server.services.batching.prefix_cache import (
    PrefixCache,
    compute_block_hash,
)


class TestComputeBlockHash:
    """Test hash computation for prefix matching."""

    def test_same_tokens_same_hash(self) -> None:
        """Same tokens should produce same hash."""
        tokens = [1, 2, 3, 4, 5]
        h1 = compute_block_hash(tokens, 0, 5)
        h2 = compute_block_hash(tokens, 0, 5)
        assert h1 == h2

    def test_different_tokens_different_hash(self) -> None:
        """Different tokens should produce different hash."""
        h1 = compute_block_hash([1, 2, 3], 0, 3)
        h2 = compute_block_hash([4, 5, 6], 0, 3)
        assert h1 != h2

    def test_chained_hashes_differ(self) -> None:
        """Same tokens with different prev_hash should differ."""
        tokens = [4, 5, 6]
        h1 = compute_block_hash(tokens, 0, 3, prev_hash=0)
        h2 = compute_block_hash(tokens, 0, 3, prev_hash=123)
        assert h1 != h2

    def test_position_matters_via_chaining(self) -> None:
        """Same tokens at different positions differ due to hash chaining."""
        # Simulate block 0 vs block 1 with same content
        # Block 0: tokens [1,2,3] with no previous hash
        # Block 1: tokens [1,2,3] with previous hash from different block 0
        tokens_block0 = [7, 8, 9]  # Different first block
        tokens_block1 = [1, 2, 3]  # Same content for second block

        # Compute hash for block 0 position
        h_at_block0 = compute_block_hash(tokens_block1, 0, 3, prev_hash=0)

        # Compute hash for block 1 position (with prev_hash from block 0)
        prev = compute_block_hash(tokens_block0, 0, 3, prev_hash=0)
        h_at_block1 = compute_block_hash(tokens_block1, 0, 3, prev_hash=prev)

        # Same content at different positions should have different hashes
        assert h_at_block0 != h_at_block1

    def test_partial_block_at_end(self) -> None:
        """Hash computation handles partial blocks at end of sequence."""
        tokens = [1, 2, 3, 4, 5]  # 5 tokens, block_size=3
        # Block starting at 3 only has tokens [4, 5]
        h = compute_block_hash(tokens, 3, 3)
        assert isinstance(h, int)

    def test_empty_block(self) -> None:
        """Hash computation handles empty token list."""
        h = compute_block_hash([], 0, 3)
        assert isinstance(h, int)


class TestPrefixLookup:
    """Test prefix cache lookup behavior."""

    def test_empty_cache_returns_empty(self) -> None:
        """Empty cache should return empty list and 0 tokens."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE * 3))
        blocks, matched = pc.lookup_prefix(tokens)

        assert blocks == []
        assert matched == 0

    def test_lookup_after_cache_returns_blocks(self) -> None:
        """Lookup should return cached blocks after caching."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        # Cache 3 blocks worth of tokens
        tokens = list(range(BLOCK_SIZE * 3))
        block_ids = [pm.allocate() for _ in range(3)]
        pc.cache_prefix(tokens, block_ids)

        # Lookup with same tokens
        found_blocks, matched = pc.lookup_prefix(tokens)

        assert len(found_blocks) == 3
        assert matched == BLOCK_SIZE * 3
        # Should return the same block IDs
        assert found_blocks == block_ids

    def test_partial_match_returns_matched_prefix(self) -> None:
        """Partial match should return only matched prefix blocks."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        # Cache 2 blocks
        tokens = list(range(BLOCK_SIZE * 2))
        block_ids = [pm.allocate() for _ in range(2)]
        pc.cache_prefix(tokens, block_ids)

        # Lookup with 3 blocks where first 2 match
        lookup_tokens = list(range(BLOCK_SIZE * 2)) + list(range(1000, 1000 + BLOCK_SIZE))
        found_blocks, matched = pc.lookup_prefix(lookup_tokens)

        assert len(found_blocks) == 2
        assert matched == BLOCK_SIZE * 2

    def test_no_match_returns_empty(self) -> None:
        """Different tokens should not match cached prefix."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        # Cache some tokens
        tokens1 = list(range(BLOCK_SIZE))
        block_ids = [pm.allocate()]
        pc.cache_prefix(tokens1, block_ids)

        # Lookup with different tokens
        tokens2 = list(range(100, 100 + BLOCK_SIZE))
        found_blocks, matched = pc.lookup_prefix(tokens2)

        assert found_blocks == []
        assert matched == 0

    def test_lookup_empty_tokens_returns_empty(self) -> None:
        """Lookup with empty tokens should return empty."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        blocks, matched = pc.lookup_prefix([])
        assert blocks == []
        assert matched == 0

    def test_lookup_partial_block_not_cached(self) -> None:
        """Partial blocks (< BLOCK_SIZE tokens) should not be cached/matched."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        # Only BLOCK_SIZE - 1 tokens (incomplete block)
        tokens = list(range(BLOCK_SIZE - 1))
        block_ids = [pm.allocate()]
        pc.cache_prefix(tokens, block_ids)

        # Stats should show nothing cached (incomplete block)
        stats = pc.get_cache_stats()
        assert stats["cached_blocks"] == 0


class TestCaching:
    """Test prefix caching behavior."""

    def test_cache_stores_blocks_correctly(self) -> None:
        """Caching should store block mappings correctly."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE * 2))
        block_ids = [pm.allocate() for _ in range(2)]
        pc.cache_prefix(tokens, block_ids)

        stats = pc.get_cache_stats()
        assert stats["cached_blocks"] == 2
        assert stats["unique_prefixes"] == 2  # Each block has unique hash chain

    def test_blocks_marked_as_prefix_cached(self) -> None:
        """Cached blocks should be marked in block manager."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()
        pc.cache_prefix(tokens, [block_id])

        block = pm.get_block(block_id)
        assert block.is_prefix_cached is True
        assert block.prefix_hash is not None

    def test_same_prefix_not_duplicated(self) -> None:
        """Caching same prefix twice should not duplicate."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()

        # Cache same prefix twice
        pc.cache_prefix(tokens, [block_id])
        pc.cache_prefix(tokens, [block_id])

        stats = pc.get_cache_stats()
        assert stats["cached_blocks"] == 1

    def test_cache_empty_tokens_no_effect(self) -> None:
        """Caching empty tokens should have no effect."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        pc.cache_prefix([], [1, 2, 3])

        stats = pc.get_cache_stats()
        assert stats["cached_blocks"] == 0

    def test_cache_empty_blocks_no_effect(self) -> None:
        """Caching with empty block list should have no effect."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        pc.cache_prefix(tokens, [])

        stats = pc.get_cache_stats()
        assert stats["cached_blocks"] == 0


class TestReferenceCount:
    """Test reference counting for prefix blocks."""

    def test_lookup_increments_ref_count(self) -> None:
        """Lookup should increment ref_count on matched blocks."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()

        pc.cache_prefix(tokens, [block_id])
        after_cache_ref = pm.get_block(block_id).ref_count

        # Lookup should increment
        pc.lookup_prefix(tokens)
        after_lookup_ref = pm.get_block(block_id).ref_count

        assert after_lookup_ref > after_cache_ref

    def test_release_decrements_ref_count(self) -> None:
        """Release should decrement ref_count."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()
        pc.cache_prefix(tokens, [block_id])

        # Lookup to increment ref_count
        pc.lookup_prefix(tokens)
        ref_after_lookup = pm.get_block(block_id).ref_count

        # Release
        pc.release_prefix([block_id])
        ref_after_release = pm.get_block(block_id).ref_count

        assert ref_after_release == ref_after_lookup - 1

    def test_multiple_lookups_accumulate_ref_count(self) -> None:
        """Multiple lookups should accumulate ref_count."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()
        pc.cache_prefix(tokens, [block_id])

        ref_after_cache = pm.get_block(block_id).ref_count

        # Multiple lookups
        pc.lookup_prefix(tokens)
        pc.lookup_prefix(tokens)
        pc.lookup_prefix(tokens)

        ref_after_lookups = pm.get_block(block_id).ref_count
        assert ref_after_lookups == ref_after_cache + 3

    def test_release_does_not_go_negative(self) -> None:
        """Release should not make ref_count negative."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()
        pc.cache_prefix(tokens, [block_id])

        # Release more times than lookups
        pc.release_prefix([block_id])
        pc.release_prefix([block_id])
        pc.release_prefix([block_id])
        pc.release_prefix([block_id])
        pc.release_prefix([block_id])

        ref = pm.get_block(block_id).ref_count
        assert ref >= 0


class TestEviction:
    """Test prefix cache eviction behavior."""

    def test_evict_unused_removes_ref_count_zero(self) -> None:
        """Evict should remove blocks with ref_count=0."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()
        pc.cache_prefix(tokens, [block_id])

        # Set ref_count to 0
        pm.get_block(block_id).ref_count = 0

        stats_before = pc.get_cache_stats()
        evicted = pc.evict_unused()
        stats_after = pc.get_cache_stats()

        assert evicted == 1
        assert stats_after["cached_blocks"] < stats_before["cached_blocks"]

    def test_evict_preserves_in_use_blocks(self) -> None:
        """Evict should not remove blocks with ref_count > 0."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()
        pc.cache_prefix(tokens, [block_id])

        # Ensure ref_count > 0
        pm.get_block(block_id).ref_count = 3

        evicted = pc.evict_unused()

        assert evicted == 0
        stats = pc.get_cache_stats()
        assert stats["cached_blocks"] == 1

    def test_evicted_blocks_unmarked(self) -> None:
        """Evicted blocks should be unmarked in block manager."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()
        pc.cache_prefix(tokens, [block_id])

        # Set ref_count to 0 for eviction
        pm.get_block(block_id).ref_count = 0

        pc.evict_unused()

        block = pm.get_block(block_id)
        assert block.is_prefix_cached is False
        assert block.prefix_hash is None


class TestClear:
    """Test prefix cache clear behavior."""

    def test_clear_removes_all_entries(self) -> None:
        """Clear should remove all cache entries."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        # Cache multiple prefixes
        for i in range(3):
            tokens = list(range(i * 100, i * 100 + BLOCK_SIZE))
            block_id = pm.allocate()
            pc.cache_prefix(tokens, [block_id])

        stats_before = pc.get_cache_stats()
        assert stats_before["cached_blocks"] == 3

        pc.clear()

        stats_after = pc.get_cache_stats()
        assert stats_after["cached_blocks"] == 0
        assert stats_after["unique_prefixes"] == 0

    def test_clear_unmarks_blocks(self) -> None:
        """Clear should unmark all blocks in block manager."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()
        pc.cache_prefix(tokens, [block_id])

        assert pm.get_block(block_id).is_prefix_cached is True

        pc.clear()

        assert pm.get_block(block_id).is_prefix_cached is False


class TestPerModelIsolation:
    """Test that prefix caches are isolated per model."""

    def test_different_models_different_caches(self) -> None:
        """Two PrefixCache instances should not share state."""
        pm = PagedBlockManager(num_blocks=20)

        pc1 = PrefixCache(model_id="model-a", block_manager=pm)
        pc2 = PrefixCache(model_id="model-b", block_manager=pm)

        # Cache same tokens in different caches
        tokens = list(range(BLOCK_SIZE))
        block1 = pm.allocate()
        block2 = pm.allocate()

        pc1.cache_prefix(tokens, [block1])
        pc2.cache_prefix(tokens, [block2])

        # Each cache should have its own entry
        assert pc1.get_cache_stats()["cached_blocks"] == 1
        assert pc2.get_cache_stats()["cached_blocks"] == 1

        # Lookup in pc1 should not find pc2's blocks
        found1, _ = pc1.lookup_prefix(tokens)
        found2, _ = pc2.lookup_prefix(tokens)

        assert found1 == [block1]
        assert found2 == [block2]
        assert block1 != block2

    def test_clear_one_cache_does_not_affect_other(self) -> None:
        """Clearing one cache should not affect another."""
        pm = PagedBlockManager(num_blocks=20)

        pc1 = PrefixCache(model_id="model-a", block_manager=pm)
        pc2 = PrefixCache(model_id="model-b", block_manager=pm)

        # Cache in both
        tokens = list(range(BLOCK_SIZE))
        block1 = pm.allocate()
        block2 = pm.allocate()

        pc1.cache_prefix(tokens, [block1])
        pc2.cache_prefix(tokens, [block2])

        # Clear pc1
        pc1.clear()

        # pc1 should be empty, pc2 should still have entry
        assert pc1.get_cache_stats()["cached_blocks"] == 0
        assert pc2.get_cache_stats()["cached_blocks"] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_release_invalid_block_no_error(self) -> None:
        """Releasing invalid block ID should not raise error."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        # Should not raise
        pc.release_prefix([999, 1000, 1001])

    def test_lookup_updates_last_used(self) -> None:
        """Lookup should update last_used timestamp via touch."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        tokens = list(range(BLOCK_SIZE))
        block_id = pm.allocate()
        pc.cache_prefix(tokens, [block_id])

        # Set old timestamp
        pm.get_block(block_id).last_used = 100.0

        # Lookup should update timestamp
        pc.lookup_prefix(tokens)

        assert pm.get_block(block_id).last_used > 100.0

    def test_cache_more_blocks_than_tokens(self) -> None:
        """Caching more blocks than complete tokens should only cache complete blocks."""
        pm = PagedBlockManager(num_blocks=10)
        pc = PrefixCache(model_id="test", block_manager=pm)

        # Only 1.5 blocks worth of tokens
        tokens = list(range(BLOCK_SIZE + BLOCK_SIZE // 2))
        block_ids = [pm.allocate() for _ in range(3)]

        pc.cache_prefix(tokens, block_ids)

        # Only first complete block should be cached
        stats = pc.get_cache_stats()
        assert stats["cached_blocks"] == 1
