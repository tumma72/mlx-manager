"""Tests for KV cache block management.

Tests cover:
- KVBlock and BlockTable dataclasses
- PagedBlockManager allocation and release
- LRU eviction behavior
- Prefix cache integration
- Edge cases (empty pool, double release, etc.)
"""

import time

import pytest

from mlx_manager.mlx_server.services.batching.block import (
    BLOCK_SIZE,
    BlockTable,
    KVBlock,
)
from mlx_manager.mlx_server.services.batching.block_manager import PagedBlockManager


class TestKVBlock:
    """Test KVBlock dataclass."""

    def test_defaults(self) -> None:
        """Test that KVBlock has correct default values."""
        block = KVBlock(block_id=0)
        assert block.block_id == 0
        assert block.ref_count == 0
        assert block.is_prefix_cached is False
        assert block.prefix_hash is None
        # last_used should be approximately now
        assert time.time() - block.last_used < 1.0

    def test_custom_values(self) -> None:
        """Test KVBlock with custom values."""
        block = KVBlock(
            block_id=42,
            ref_count=3,
            last_used=100.0,
            is_prefix_cached=True,
            prefix_hash=12345,
        )
        assert block.block_id == 42
        assert block.ref_count == 3
        assert block.last_used == 100.0
        assert block.is_prefix_cached is True
        assert block.prefix_hash == 12345


class TestBlockTable:
    """Test BlockTable dataclass and methods."""

    def test_defaults(self) -> None:
        """Test BlockTable has correct default values."""
        table = BlockTable(request_id="test-request")
        assert table.request_id == "test-request"
        assert table.logical_to_physical == {}
        assert table.num_tokens == 0

    def test_current_logical_index(self) -> None:
        """Test logical index calculation from token count."""
        table = BlockTable(request_id="test")

        # At 0 tokens, logical index is 0
        table.num_tokens = 0
        assert table.current_logical_index() == 0

        # At 31 tokens, still in block 0
        table.num_tokens = 31
        assert table.current_logical_index() == 0

        # At 32 tokens, in block 1
        table.num_tokens = 32
        assert table.current_logical_index() == 1

        # At 64 tokens, in block 2
        table.num_tokens = 64
        assert table.current_logical_index() == 2

        # At 100 tokens, in block 3 (100 // 32 = 3)
        table.num_tokens = 100
        assert table.current_logical_index() == 3

    def test_needs_new_block_at_boundaries(self) -> None:
        """Test needs_new_block returns true at block boundaries."""
        table = BlockTable(request_id="test")

        # At 0 tokens, needs new block (starting)
        table.num_tokens = 0
        assert table.needs_new_block() is True

        # At 1 token, doesn't need new block
        table.num_tokens = 1
        assert table.needs_new_block() is False

        # At 31 tokens, doesn't need new block
        table.num_tokens = 31
        assert table.needs_new_block() is False

        # At 32 tokens, needs new block (boundary)
        table.num_tokens = 32
        assert table.needs_new_block() is True

        # At 33 tokens, doesn't need new block
        table.num_tokens = 33
        assert table.needs_new_block() is False

        # At 64 tokens, needs new block (boundary)
        table.num_tokens = 64
        assert table.needs_new_block() is True

    def test_add_token(self) -> None:
        """Test adding tokens updates mapping correctly."""
        table = BlockTable(request_id="test")

        # Add first token with physical block 5
        assert table.needs_new_block() is True
        table.add_token(physical_block_id=5)
        assert table.num_tokens == 1
        assert table.logical_to_physical == {0: 5}

        # Add more tokens in same block
        for _ in range(31):  # Fill rest of block 0
            table.add_token(physical_block_id=5)
        assert table.num_tokens == 32
        assert table.logical_to_physical == {0: 5}

        # Now at boundary, add token with new block
        assert table.needs_new_block() is True
        table.add_token(physical_block_id=10)
        assert table.num_tokens == 33
        assert table.logical_to_physical == {0: 5, 1: 10}

    def test_get_physical_blocks(self) -> None:
        """Test getting physical blocks in logical order."""
        table = BlockTable(request_id="test")

        # Empty table
        assert table.get_physical_blocks() == []

        # Add mapping out of order
        table.logical_to_physical = {0: 7, 2: 3, 1: 9}
        assert table.get_physical_blocks() == [7, 9, 3]


class TestPagedBlockManager:
    """Test PagedBlockManager allocation and eviction."""

    def test_initialization(self) -> None:
        """Test manager initializes with all blocks free."""
        pm = PagedBlockManager(num_blocks=10)
        assert pm.num_blocks == 10
        assert pm.get_free_count() == 10
        assert len(pm.blocks) == 10

    def test_allocate_decrements_free_count(self) -> None:
        """Test allocation reduces free block count."""
        pm = PagedBlockManager(num_blocks=5)
        assert pm.get_free_count() == 5

        block_id = pm.allocate()
        assert pm.get_free_count() == 4
        assert pm.blocks[block_id].ref_count == 1

    def test_release_increments_free_count(self) -> None:
        """Test release returns block to free pool."""
        pm = PagedBlockManager(num_blocks=5)

        block_id = pm.allocate()
        assert pm.get_free_count() == 4

        pm.release(block_id)
        assert pm.get_free_count() == 5
        assert pm.blocks[block_id].ref_count == 0

    def test_allocate_fails_when_empty(self) -> None:
        """Test allocation raises MemoryError when pool is exhausted."""
        pm = PagedBlockManager(num_blocks=2)

        pm.allocate()
        pm.allocate()
        assert pm.get_free_count() == 0

        with pytest.raises(MemoryError, match="No free KV cache blocks"):
            pm.allocate()

    def test_allocate_release_allocate_cycle(self) -> None:
        """Test allocate all, release one, allocate again."""
        pm = PagedBlockManager(num_blocks=3)

        # Allocate all
        ids = [pm.allocate() for _ in range(3)]
        assert pm.get_free_count() == 0

        # Release middle one
        pm.release(ids[1])
        assert pm.get_free_count() == 1

        # Allocate again should work
        new_id = pm.allocate()
        assert new_id == ids[1]  # Should get the released block
        assert pm.get_free_count() == 0

    def test_double_release_no_duplicate(self) -> None:
        """Test double release doesn't duplicate in free list."""
        pm = PagedBlockManager(num_blocks=3)

        block_id = pm.allocate()
        assert pm.get_free_count() == 2

        pm.release(block_id)
        assert pm.get_free_count() == 3

        # Double release should not increase count
        pm.release(block_id)
        assert pm.get_free_count() == 3

    def test_get_block(self) -> None:
        """Test getting block by ID."""
        pm = PagedBlockManager(num_blocks=5)

        block = pm.get_block(2)
        assert block.block_id == 2

        with pytest.raises(KeyError):
            pm.get_block(999)

    def test_touch_updates_last_used(self) -> None:
        """Test touch updates last_used timestamp."""
        pm = PagedBlockManager(num_blocks=5)
        block_id = pm.allocate()

        old_time = pm.blocks[block_id].last_used
        time.sleep(0.01)
        pm.touch(block_id)
        new_time = pm.blocks[block_id].last_used

        assert new_time > old_time

    def test_get_stats(self) -> None:
        """Test stats reporting."""
        pm = PagedBlockManager(num_blocks=10)

        stats = pm.get_stats()
        assert stats["total"] == 10
        assert stats["free"] == 10
        assert stats["used"] == 0
        assert stats["block_size"] == BLOCK_SIZE

        pm.allocate()
        pm.allocate()

        stats = pm.get_stats()
        assert stats["free"] == 8
        assert stats["used"] == 2


class TestLRUEviction:
    """Test LRU eviction behavior.

    Eviction targets blocks that:
    - Have ref_count=0
    - Are NOT already in the free list (typically prefix cached blocks)

    Normal released blocks go directly to free list and don't need eviction.
    Prefix cached blocks with ref_count=0 stay allocated and need eviction.
    """

    def test_evict_returns_count(self) -> None:
        """Test eviction returns count of evicted blocks."""
        pm = PagedBlockManager(num_blocks=5)

        # Allocate 3 blocks and mark as prefix cached (so they don't go to free list on release)
        ids = [pm.allocate() for _ in range(3)]
        for bid in ids:
            pm.mark_as_prefix_cached(bid, prefix_hash=bid * 100)
            pm.blocks[bid].ref_count = 0  # Simulate no active users

        # Free count should still be 2 (only 2 never allocated)
        assert pm.get_free_count() == 2

        # Evict 2
        evicted = pm.evict_lru_blocks(count=2)
        assert evicted == 2
        assert pm.get_free_count() == 4

    def test_evict_only_ref_count_zero(self) -> None:
        """Test eviction only affects ref_count=0 blocks."""
        pm = PagedBlockManager(num_blocks=5)

        # Allocate 3 blocks and mark as prefix cached
        id1 = pm.allocate()
        id2 = pm.allocate()
        id3 = pm.allocate()

        for bid in [id1, id2, id3]:
            pm.mark_as_prefix_cached(bid, prefix_hash=bid * 100)

        # Only set id2 to ref_count=0
        pm.blocks[id2].ref_count = 0
        # id1 and id3 still have ref_count=1

        # Try to evict 3 - should only evict 1 (id2)
        evicted = pm.evict_lru_blocks(count=3)
        assert evicted == 1

        # id1 and id3 still have ref_count=1
        assert pm.blocks[id1].ref_count == 1
        assert pm.blocks[id3].ref_count == 1

    def test_evict_prefers_older_blocks(self) -> None:
        """Test eviction prefers older (by last_used) blocks."""
        pm = PagedBlockManager(num_blocks=5)

        # Allocate 3 blocks with different timestamps
        id1 = pm.allocate()
        pm.mark_as_prefix_cached(id1, prefix_hash=100)
        pm.blocks[id1].ref_count = 0
        pm.blocks[id1].last_used = 100.0  # Oldest

        id2 = pm.allocate()
        pm.mark_as_prefix_cached(id2, prefix_hash=200)
        pm.blocks[id2].ref_count = 0
        pm.blocks[id2].last_used = 200.0  # Middle

        id3 = pm.allocate()
        pm.mark_as_prefix_cached(id3, prefix_hash=300)
        pm.blocks[id3].ref_count = 0
        pm.blocks[id3].last_used = 300.0  # Newest

        # Evict 1 - should be id1 (oldest)
        evicted = pm.evict_lru_blocks(count=1)
        assert evicted == 1

        # id1 should now be in free list, id2 and id3 should not be
        assert id1 in pm.free_blocks
        assert id2 not in pm.free_blocks
        assert id3 not in pm.free_blocks

    def test_evict_no_candidates(self) -> None:
        """Test eviction returns 0 when no blocks can be evicted."""
        pm = PagedBlockManager(num_blocks=5)

        # Allocate all blocks with ref_count > 0
        for _ in range(5):
            pm.allocate()

        # No blocks have ref_count=0
        evicted = pm.evict_lru_blocks(count=3)
        assert evicted == 0


class TestPrefixCache:
    """Test prefix cache support in block manager."""

    def test_mark_prefix_cached(self) -> None:
        """Test marking a block as prefix cached."""
        pm = PagedBlockManager(num_blocks=5)

        block_id = pm.allocate()
        pm.mark_as_prefix_cached(block_id, prefix_hash=12345)

        block = pm.blocks[block_id]
        assert block.is_prefix_cached is True
        assert block.prefix_hash == 12345

    def test_unmark_prefix_cached(self) -> None:
        """Test unmarking a prefix cached block."""
        pm = PagedBlockManager(num_blocks=5)

        block_id = pm.allocate()
        pm.mark_as_prefix_cached(block_id, prefix_hash=12345)
        pm.unmark_prefix_cached(block_id)

        block = pm.blocks[block_id]
        assert block.is_prefix_cached is False
        assert block.prefix_hash is None

    def test_prefix_cached_block_not_freed_on_release(self) -> None:
        """Test prefix cached blocks stay allocated after release."""
        pm = PagedBlockManager(num_blocks=5)

        block_id = pm.allocate()
        assert pm.get_free_count() == 4

        pm.mark_as_prefix_cached(block_id, prefix_hash=12345)
        pm.release(block_id)

        # Should NOT return to free pool
        assert pm.get_free_count() == 4
        assert block_id not in pm.free_blocks

    def test_prefix_cached_can_be_evicted(self) -> None:
        """Test prefix cached blocks with ref_count=0 CAN be evicted."""
        pm = PagedBlockManager(num_blocks=5)

        # Allocate and mark as prefix cached
        block_id = pm.allocate()
        pm.mark_as_prefix_cached(block_id, prefix_hash=12345)

        # Set ref_count to 0 manually (simulating no active users)
        pm.blocks[block_id].ref_count = 0
        pm.blocks[block_id].last_used = 100.0  # Old timestamp

        # Should be evictable
        evicted = pm.evict_lru_blocks(count=1)
        assert evicted == 1
        assert block_id in pm.free_blocks

    def test_prefix_cache_cleared_on_eviction(self) -> None:
        """Test eviction clears prefix cache flags."""
        pm = PagedBlockManager(num_blocks=5)

        block_id = pm.allocate()
        pm.mark_as_prefix_cached(block_id, prefix_hash=12345)
        pm.blocks[block_id].ref_count = 0

        # Evict
        pm.evict_lru_blocks(count=1)

        # Flags should be cleared
        block = pm.blocks[block_id]
        assert block.is_prefix_cached is False
        assert block.prefix_hash is None
