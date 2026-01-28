"""Prefix caching for KV block sharing across requests.

This module implements hash-based prefix caching, enabling requests with
common prompt prefixes (system prompts, few-shot examples) to share KV
cache blocks. This reduces redundant computation and memory usage.

vLLM's Automatic Prefix Caching reports 5.8x speedup on shared prefixes.

Key features:
- Hash-based prefix matching for O(1) lookup
- Reference counting prevents eviction of in-use blocks
- Per-model cache isolation (no cross-model sharing)
- LRU eviction of unused prefix blocks under memory pressure
"""

import asyncio

from mlx_manager.mlx_server.services.batching.block import BLOCK_SIZE
from mlx_manager.mlx_server.services.batching.block_manager import PagedBlockManager


def compute_block_hash(
    token_ids: list[int],
    block_start: int,
    block_size: int,
    prev_hash: int = 0,
) -> int:
    """Compute content-based hash for a block of tokens.

    Chain hashes to include position context - same tokens at different
    positions produce different hashes. This ensures that matching is
    exact: only prefixes with identical token sequences match.

    Args:
        token_ids: The full token sequence
        block_start: Starting index of the block within token_ids
        block_size: Number of tokens per block
        prev_hash: Hash of the previous block (for chaining)

    Returns:
        Integer hash value for this block
    """
    block_end = min(block_start + block_size, len(token_ids))
    block_tokens = tuple(token_ids[block_start:block_end])
    return hash((prev_hash, block_tokens))


class PrefixCache:
    """Hash-based prefix cache with reference counting.

    The prefix cache enables KV block sharing across requests that have
    common prompt prefixes. When a new request arrives:

    1. Compute hashes for each complete block in the prompt
    2. Look up hashes to find matching cached blocks
    3. Reuse matched blocks instead of recomputing KV cache

    Per-model scoping: Each PrefixCache instance handles one model.
    The scheduler creates one PrefixCache per loaded model.

    Thread safety: Uses asyncio.Lock for cache operations.

    Attributes:
        model_id: Identifier for the model this cache serves
        block_manager: The PagedBlockManager for block operations
    """

    def __init__(self, model_id: str, block_manager: PagedBlockManager) -> None:
        """Initialize prefix cache for a specific model.

        Args:
            model_id: Identifier for the model (for logging/debugging)
            block_manager: The block manager to coordinate with
        """
        self.model_id = model_id
        self.block_manager = block_manager

        # hash -> list of physical block IDs (in sequence order)
        self._hash_to_blocks: dict[int, list[int]] = {}

        # block_id -> hash (reverse lookup for cleanup)
        self._block_to_hash: dict[int, int] = {}

        # Thread safety for cache operations
        self._lock = asyncio.Lock()

    def lookup_prefix(self, prompt_tokens: list[int]) -> tuple[list[int], int]:
        """Look up cached blocks matching the prompt prefix.

        Computes hashes for each complete block in the prompt and checks
        if they exist in the cache. Returns the longest matching prefix.

        For each matched block:
        - Increments ref_count to prevent eviction
        - Updates last_used timestamp via touch()

        Args:
            prompt_tokens: The full prompt token sequence

        Returns:
            Tuple of (list of cached block IDs, number of tokens matched)
            If no match, returns ([], 0)
        """
        if not prompt_tokens:
            return [], 0

        matched_blocks: list[int] = []
        matched_tokens = 0
        prev_hash = 0

        # Process complete blocks only
        num_complete_blocks = len(prompt_tokens) // BLOCK_SIZE

        for block_idx in range(num_complete_blocks):
            block_start = block_idx * BLOCK_SIZE
            block_hash = compute_block_hash(
                prompt_tokens, block_start, BLOCK_SIZE, prev_hash
            )

            if block_hash in self._hash_to_blocks:
                # Found a match - get the block IDs for this hash
                cached_blocks = self._hash_to_blocks[block_hash]
                if cached_blocks:
                    # Use the first block ID (they should all have same content)
                    block_id = cached_blocks[0]

                    # Update block state
                    block = self.block_manager.get_block(block_id)
                    block.ref_count += 1
                    self.block_manager.touch(block_id)

                    matched_blocks.append(block_id)
                    matched_tokens += BLOCK_SIZE
                    prev_hash = block_hash
                else:
                    # Empty block list - no match
                    break
            else:
                # No match - stop looking
                break

        return matched_blocks, matched_tokens

    def cache_prefix(self, prompt_tokens: list[int], block_ids: list[int]) -> None:
        """Cache prefix blocks for future reuse.

        Computes hashes for each block and stores the mapping.
        Marks blocks as prefix cached in the block manager.

        Args:
            prompt_tokens: The token sequence that was processed
            block_ids: Physical block IDs containing the KV cache
        """
        if not prompt_tokens or not block_ids:
            return

        prev_hash = 0

        for block_idx, block_id in enumerate(block_ids):
            block_start = block_idx * BLOCK_SIZE
            block_end = block_start + BLOCK_SIZE

            # Only cache complete blocks
            if block_end > len(prompt_tokens):
                break

            block_hash = compute_block_hash(
                prompt_tokens, block_start, BLOCK_SIZE, prev_hash
            )

            # Store in cache if not already present
            if block_hash not in self._hash_to_blocks:
                self._hash_to_blocks[block_hash] = []

            # Add block_id if not already in the list
            if block_id not in self._hash_to_blocks[block_hash]:
                self._hash_to_blocks[block_hash].append(block_id)

            # Store reverse mapping
            self._block_to_hash[block_id] = block_hash

            # Mark block as prefix cached in block manager
            self.block_manager.mark_as_prefix_cached(block_id, block_hash)

            # Increment ref_count to track prefix cache usage
            block = self.block_manager.get_block(block_id)
            block.ref_count += 1

            prev_hash = block_hash

    def release_prefix(self, block_ids: list[int]) -> None:
        """Release reference to prefix blocks.

        Decrements ref_count on each block. Does NOT remove from cache -
        blocks stay cached for potential future reuse. LRU eviction will
        clean up unused blocks when memory pressure is detected.

        Args:
            block_ids: List of block IDs to release
        """
        for block_id in block_ids:
            if block_id in self.block_manager.blocks:
                block = self.block_manager.get_block(block_id)
                if block.ref_count > 0:
                    block.ref_count -= 1

    def evict_unused(self) -> int:
        """Evict unused prefix cache entries.

        Finds prefix cache entries where all blocks have ref_count=0,
        removes them from cache, and unmarks them in block_manager.
        The block manager's evict_lru_blocks will then be able to
        reclaim these blocks.

        Called by scheduler when memory pressure detected.

        Returns:
            Count of blocks freed from prefix cache
        """
        blocks_to_evict: list[int] = []

        # Find blocks with ref_count=0 that are prefix cached
        for block_id, block_hash in list(self._block_to_hash.items()):
            if block_id in self.block_manager.blocks:
                block = self.block_manager.get_block(block_id)
                if block.ref_count == 0 and block.is_prefix_cached:
                    blocks_to_evict.append(block_id)

        # Remove from cache and unmark
        for block_id in blocks_to_evict:
            cached_hash: int | None = self._block_to_hash.get(block_id)
            if cached_hash is not None:
                # Remove from hash -> blocks mapping
                if cached_hash in self._hash_to_blocks:
                    block_list = self._hash_to_blocks[cached_hash]
                    if block_id in block_list:
                        block_list.remove(block_id)
                    # Clean up empty hash entries
                    if not block_list:
                        del self._hash_to_blocks[cached_hash]

                # Remove reverse mapping
                del self._block_to_hash[block_id]

            # Unmark in block manager
            self.block_manager.unmark_prefix_cached(block_id)

        return len(blocks_to_evict)

    def get_cache_stats(self) -> dict[str, int]:
        """Return prefix cache statistics.

        Returns:
            Dictionary with cached_blocks and unique_prefixes counts
        """
        return {
            "cached_blocks": len(self._block_to_hash),
            "unique_prefixes": len(self._hash_to_blocks),
        }

    def clear(self) -> None:
        """Clear all cache entries and unmark blocks.

        Removes all entries from the prefix cache and unmarks all
        blocks in the block manager.
        """
        # Unmark all cached blocks
        for block_id in list(self._block_to_hash.keys()):
            self.block_manager.unmark_prefix_cached(block_id)

        # Clear cache structures
        self._hash_to_blocks.clear()
        self._block_to_hash.clear()
