"""Radix Tree index for efficient prefix matching."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, NamedTuple
from enum import Enum
import time


class StorageTier(Enum):
    """Storage tier for KV cache."""
    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"


@dataclass
class SegmentMeta:
    """Metadata for a cached segment."""

    segment_hash: str
    start_idx: int
    end_idx: int
    num_tokens: int
    storage_tier: StorageTier
    storage_key: str
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0

    def update_access(self):
        """Update access statistics."""
        self.last_access = time.time()
        self.access_count += 1


class MatchResult(NamedTuple):
    """Result of prefix matching."""
    matched_tokens: int              # Number of matched tokens
    matched_segments: List[SegmentMeta]  # List of matched segment metadata
    remaining_tokens: int            # Number of unmatched tokens


class RadixNode:
    """Node in the Radix Tree."""

    def __init__(self):
        # segment_hash -> child_node
        self.children: Dict[str, "RadixNode"] = {}
        # Metadata for this node's segment
        self.meta: Optional[SegmentMeta] = None
        # Reference count for garbage collection
        self.ref_count: int = 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def __repr__(self) -> str:
        meta_info = f"meta={self.meta.segment_hash[:8]}..." if self.meta else "no_meta"
        return f"RadixNode({meta_info}, children={len(self.children)})"


class SegmentRadixTree:
    """
    Radix Tree for efficient prefix matching of token sequences.

    Unlike hash-based indexing, Radix Tree supports:
    - O(n) prefix matching where n is the sequence length
    - Natural support for variable-length segments
    - Efficient memory sharing for common prefixes
    """

    def __init__(self):
        self.root = RadixNode()
        self._size = 0

    def insert(
        self,
        segments: List["Segment"],
        storage_tier: StorageTier,
    ) -> List[SegmentMeta]:
        """
        Insert a sequence of segments.

        Args:
            segments: List of Segment objects
            storage_tier: Initial storage tier for the KV cache

        Returns:
            List of created SegmentMeta objects
        """
        node = self.root
        metas = []

        for seg in segments:
            seg_hash = seg.hash_value

            # Create child if not exists
            if seg_hash not in node.children:
                node.children[seg_hash] = RadixNode()
                self._size += 1

            child = node.children[seg_hash]

            # Create or update metadata
            if child.meta is None:
                meta = SegmentMeta(
                    segment_hash=seg_hash,
                    start_idx=seg.start_idx,
                    end_idx=seg.end_idx,
                    num_tokens=len(seg),
                    storage_tier=storage_tier,
                    storage_key=f"kv_{seg_hash}",
                )
                child.meta = meta
            else:
                meta = child.meta
                meta.update_access()

            child.ref_count += 1
            metas.append(meta)
            node = child

        return metas

    def match_prefix(self, segments: List["Segment"]) -> MatchResult:
        """
        Match the longest prefix of segments.

        Args:
            segments: List of Segment objects to match

        Returns:
            MatchResult with matched segments and statistics
        """
        node = self.root
        matched_metas = []
        total_matched_tokens = 0

        for seg in segments:
            seg_hash = seg.hash_value

            if seg_hash not in node.children:
                break

            child = node.children[seg_hash]

            if child.meta is None:
                break

            # Update access statistics
            child.meta.update_access()

            matched_metas.append(child.meta)
            total_matched_tokens += child.meta.num_tokens
            node = child

        total_tokens = sum(len(seg) for seg in segments)
        remaining = total_tokens - total_matched_tokens

        return MatchResult(
            matched_tokens=total_matched_tokens,
            matched_segments=matched_metas,
            remaining_tokens=remaining,
        )

    def lookup(self, segment_hash: str) -> Optional[SegmentMeta]:
        """
        Look up a segment by its hash.

        Note: This is O(n) where n is tree size. Use match_prefix for efficient lookups.
        """
        def _search(node: RadixNode) -> Optional[SegmentMeta]:
            if segment_hash in node.children:
                child = node.children[segment_hash]
                return child.meta
            for child in node.children.values():
                result = _search(child)
                if result:
                    return result
            return None

        return _search(self.root)

    def update_storage_tier(self, segment_hash: str, new_tier: StorageTier) -> bool:
        """Update the storage tier of a segment."""
        meta = self.lookup(segment_hash)
        if meta:
            meta.storage_tier = new_tier
            return True
        return False

    def remove(self, segment_hash: str) -> bool:
        """
        Remove a segment from the tree.

        Only removes if reference count reaches 0.
        """
        def _remove(node: RadixNode, target_hash: str) -> bool:
            if target_hash in node.children:
                child = node.children[target_hash]
                child.ref_count -= 1

                if child.ref_count <= 0:
                    del node.children[target_hash]
                    self._size -= 1
                    return True
                return False

            for child in node.children.values():
                if _remove(child, target_hash):
                    return True
            return False

        return _remove(self.root, segment_hash)

    def get_all_metas(self) -> List[SegmentMeta]:
        """Get metadata for all segments."""
        metas = []

        def _collect(node: RadixNode):
            if node.meta:
                metas.append(node.meta)
            for child in node.children.values():
                _collect(child)

        _collect(self.root)
        return metas

    def get_lru_candidates(self, n: int) -> List[SegmentMeta]:
        """Get n least recently used segments."""
        metas = self.get_all_metas()
        metas.sort(key=lambda m: m.last_access)
        return metas[:n]

    def get_lfu_candidates(self, n: int) -> List[SegmentMeta]:
        """Get n least frequently used segments."""
        metas = self.get_all_metas()
        metas.sort(key=lambda m: m.access_count)
        return metas[:n]

    def __len__(self) -> int:
        return self._size

    def stats(self) -> Dict:
        """Get statistics about the tree."""
        metas = self.get_all_metas()
        tier_counts = {}
        total_tokens = 0

        for meta in metas:
            tier = meta.storage_tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            total_tokens += meta.num_tokens

        return {
            "total_segments": len(metas),
            "total_nodes": self._size,
            "total_tokens": total_tokens,
            "tier_distribution": tier_counts,
        }

    def clear(self):
        """Clear the entire tree."""
        self.root = RadixNode()
        self._size = 0
