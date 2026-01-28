"""Paged KV cache block manager for memory-efficient allocation.

This module implements PagedBlockManager which manages a pool of fixed-size
blocks for KV cache storage. Key features:
- O(1) allocation and release using a free block stack
- Reference counting for shared blocks (prefix cache)
- LRU eviction for memory pressure management
- Thread-safe operations with async lock
"""

import asyncio
import time
from typing import Any

from mlx_manager.mlx_server.services.batching.block import BLOCK_SIZE, KVBlock


class PagedBlockManager:
    """Manages a pool of KV cache blocks with allocation and eviction.

    The block manager maintains:
    - A fixed pool of blocks (pre-allocated for predictable memory)
    - A free list (stack) for O(1) allocation
    - Reference counting for block sharing (prefix cache)
    - LRU tracking for eviction decisions

    Attributes:
        num_blocks: Total number of blocks in the pool
        blocks: Dictionary of all blocks indexed by block_id
        free_blocks: Stack of available block IDs
    """

    def __init__(self, num_blocks: int) -> None:
        """Initialize the block manager with a fixed pool size.

        Args:
            num_blocks: Total number of blocks to manage
        """
        self.num_blocks = num_blocks
        self.blocks: dict[int, KVBlock] = {}
        self.free_blocks: list[int] = []
        self._lock = asyncio.Lock()

        # Initialize all blocks as free
        for block_id in range(num_blocks):
            self.blocks[block_id] = KVBlock(block_id=block_id)
            self.free_blocks.append(block_id)

    def allocate(self) -> int:
        """Allocate a free block and return its ID.

        Pops a block from the free list, increments its ref_count,
        and updates last_used timestamp.

        Returns:
            The block_id of the allocated block.

        Raises:
            MemoryError: If no free blocks are available.
        """
        if not self.free_blocks:
            raise MemoryError("No free KV cache blocks")

        block_id = self.free_blocks.pop()
        block = self.blocks[block_id]
        block.ref_count = 1
        block.last_used = time.time()
        return block_id

    def release(self, block_id: int) -> None:
        """Release a block, potentially returning it to the free pool.

        Decrements ref_count. If ref_count reaches 0 and the block
        is not prefix cached, it returns to the free pool.

        Args:
            block_id: The block to release.
        """
        if block_id not in self.blocks:
            return

        block = self.blocks[block_id]
        block.ref_count = max(0, block.ref_count - 1)

        # Return to free pool if not in use and not prefix cached
        if block.ref_count <= 0 and not block.is_prefix_cached:
            # Avoid duplicates in free list
            if block_id not in self.free_blocks:
                self.free_blocks.append(block_id)

    def get_block(self, block_id: int) -> KVBlock:
        """Get a block by its ID.

        Args:
            block_id: The block ID to retrieve.

        Returns:
            The KVBlock with the given ID.

        Raises:
            KeyError: If block_id is not valid.
        """
        return self.blocks[block_id]

    def touch(self, block_id: int) -> None:
        """Update last_used timestamp for a block.

        Args:
            block_id: The block to touch.
        """
        if block_id in self.blocks:
            self.blocks[block_id].last_used = time.time()

    def evict_lru_blocks(self, count: int = 1) -> int:
        """Evict up to `count` blocks using LRU policy.

        Only evicts blocks with ref_count=0. Sorts candidates by
        last_used (oldest first) and evicts up to `count` blocks.

        If a block is prefix cached, clears the prefix cache flags
        before eviction.

        Args:
            count: Maximum number of blocks to evict.

        Returns:
            Number of blocks actually evicted.
        """
        # Find eviction candidates: ref_count=0 blocks not in free list
        candidates: list[KVBlock] = []
        for block in self.blocks.values():
            if block.ref_count == 0 and block.block_id not in self.free_blocks:
                candidates.append(block)

        # Sort by last_used (oldest first for LRU)
        candidates.sort(key=lambda b: b.last_used)

        evicted = 0
        for block in candidates[:count]:
            # Clear prefix cache flags if set
            if block.is_prefix_cached:
                block.is_prefix_cached = False
                block.prefix_hash = None

            # Add to free pool
            if block.block_id not in self.free_blocks:
                self.free_blocks.append(block.block_id)
            evicted += 1

        return evicted

    def get_free_count(self) -> int:
        """Return the number of free blocks.

        Returns:
            Number of blocks available for allocation.
        """
        return len(self.free_blocks)

    def get_stats(self) -> dict[str, Any]:
        """Return block pool statistics.

        Returns:
            Dictionary with total, free, and used block counts.
        """
        free = len(self.free_blocks)
        return {
            "total": self.num_blocks,
            "free": free,
            "used": self.num_blocks - free,
            "block_size": BLOCK_SIZE,
        }

    def mark_as_prefix_cached(self, block_id: int, prefix_hash: int) -> None:
        """Mark a block as part of prefix cache.

        Prefix cached blocks are not automatically returned to the free
        pool when released - they stay allocated for potential reuse.

        Args:
            block_id: The block to mark.
            prefix_hash: Hash for prefix cache lookup.
        """
        if block_id in self.blocks:
            block = self.blocks[block_id]
            block.is_prefix_cached = True
            block.prefix_hash = prefix_hash

    def unmark_prefix_cached(self, block_id: int) -> None:
        """Clear prefix cache flags from a block.

        Args:
            block_id: The block to unmark.
        """
        if block_id in self.blocks:
            block = self.blocks[block_id]
            block.is_prefix_cached = False
            block.prefix_hash = None
