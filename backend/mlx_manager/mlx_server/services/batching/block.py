"""KV cache block primitives for paged memory management.

This module defines the foundational data structures for paged KV cache:
- KVBlock: A fixed-size block that can hold BLOCK_SIZE tokens of KV cache data
- BlockTable: Per-request mapping from logical to physical block indices

The paged approach allows non-contiguous memory allocation, reducing
memory waste from 60-80% (traditional pre-allocation) to under 4%.
"""

import time
from dataclasses import dataclass, field

# Tokens per block - balance between flexibility and efficiency
# 32 tokens provides ~4% internal fragmentation vs ~60-80% with pre-allocation
BLOCK_SIZE = 32


@dataclass
class KVBlock:
    """A fixed-size block for KV cache storage.

    Attributes:
        block_id: Unique identifier for this block
        ref_count: Number of requests currently using this block
        last_used: Timestamp of last access (for LRU eviction)
        is_prefix_cached: Whether this block is part of prefix cache
        prefix_hash: Hash for prefix cache lookup (if prefix_cached)
    """

    block_id: int
    ref_count: int = 0
    last_used: float = field(default_factory=time.time)
    is_prefix_cached: bool = False
    prefix_hash: int | None = None


@dataclass
class BlockTable:
    """Per-request mapping from logical to physical blocks.

    A BlockTable tracks which physical blocks hold a request's KV cache data.
    As tokens are generated, the table grows by mapping new logical indices
    to physical block IDs allocated by the block manager.

    Attributes:
        request_id: Unique identifier for the request
        logical_to_physical: Mapping from logical block index to physical block ID
        num_tokens: Total tokens processed for this request
    """

    request_id: str
    logical_to_physical: dict[int, int] = field(default_factory=dict)
    num_tokens: int = 0

    def current_logical_index(self) -> int:
        """Return the current logical block index based on token count.

        Returns:
            The logical index of the block containing the current token position.
        """
        return self.num_tokens // BLOCK_SIZE

    def needs_new_block(self) -> bool:
        """Check if a new block is needed (at block boundary).

        Returns:
            True if we're at a block boundary and need a new block.
            This is true at 0, 32, 64, etc. tokens.
        """
        return self.num_tokens % BLOCK_SIZE == 0

    def add_token(self, physical_block_id: int) -> None:
        """Record a new token being added.

        If we're at a block boundary (needs_new_block() was true),
        this records the mapping to the newly allocated physical block.

        Args:
            physical_block_id: The physical block ID for the current logical index
        """
        logical_idx = self.current_logical_index()
        if logical_idx not in self.logical_to_physical:
            self.logical_to_physical[logical_idx] = physical_block_id
        self.num_tokens += 1

    def get_physical_blocks(self) -> list[int]:
        """Return list of physical block IDs in logical order.

        Returns:
            Ordered list of physical block IDs, from logical index 0 onwards.
        """
        if not self.logical_to_physical:
            return []
        max_idx = max(self.logical_to_physical.keys())
        return [
            self.logical_to_physical[i]
            for i in range(max_idx + 1)
            if i in self.logical_to_physical
        ]
