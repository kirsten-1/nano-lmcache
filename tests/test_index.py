"""Tests for index module."""

import pytest
import time
from nano_lmcache.index import (
    SegmentRadixTree,
    StorageTier,
    SegmentMeta,
    MatchResult,
    RadixNode,
)
from nano_lmcache.segment import Segment


def make_segment(tokens, start=0, hash_suffix=""):
    """Helper to create a Segment."""
    return Segment(
        tokens=tokens,
        start_idx=start,
        end_idx=start + len(tokens),
        hash_value=f"hash_{hash_suffix or ''.join(map(str, tokens))}",
    )


class TestRadixNode:
    """Tests for RadixNode."""

    def test_is_leaf(self):
        """Test is_leaf method."""
        node = RadixNode()
        assert node.is_leaf()

        node.children["child1"] = RadixNode()
        assert not node.is_leaf()


class TestSegmentMeta:
    """Tests for SegmentMeta."""

    def test_update_access(self):
        """Test access statistics update."""
        meta = SegmentMeta(
            segment_hash="test",
            start_idx=0,
            end_idx=10,
            num_tokens=10,
            storage_tier=StorageTier.CPU,
            storage_key="kv_test",
        )

        old_access = meta.last_access
        old_count = meta.access_count

        time.sleep(0.01)
        meta.update_access()

        assert meta.last_access > old_access
        assert meta.access_count == old_count + 1


class TestSegmentRadixTree:
    """Tests for SegmentRadixTree."""

    def test_empty_tree(self):
        """Test empty tree."""
        tree = SegmentRadixTree()
        assert len(tree) == 0
        assert tree.stats()["total_segments"] == 0

    def test_insert_single(self):
        """Test inserting a single segment."""
        tree = SegmentRadixTree()
        seg = make_segment([1, 2, 3])

        metas = tree.insert([seg], StorageTier.CPU)

        assert len(metas) == 1
        assert metas[0].num_tokens == 3
        assert metas[0].storage_tier == StorageTier.CPU
        assert len(tree) == 1

    def test_insert_sequence(self):
        """Test inserting a sequence of segments."""
        tree = SegmentRadixTree()
        segments = [
            make_segment([1, 2], start=0, hash_suffix="a"),
            make_segment([3, 4], start=2, hash_suffix="b"),
            make_segment([5, 6], start=4, hash_suffix="c"),
        ]

        metas = tree.insert(segments, StorageTier.CPU)

        assert len(metas) == 3
        assert len(tree) == 3

    def test_match_prefix_exact(self):
        """Test exact prefix matching."""
        tree = SegmentRadixTree()
        segments = [
            make_segment([1, 2], start=0, hash_suffix="a"),
            make_segment([3, 4], start=2, hash_suffix="b"),
        ]
        tree.insert(segments, StorageTier.CPU)

        # Exact match
        result = tree.match_prefix(segments)

        assert result.matched_tokens == 4
        assert len(result.matched_segments) == 2
        assert result.remaining_tokens == 0

    def test_match_prefix_partial(self):
        """Test partial prefix matching."""
        tree = SegmentRadixTree()
        segments = [
            make_segment([1, 2], start=0, hash_suffix="a"),
            make_segment([3, 4], start=2, hash_suffix="b"),
        ]
        tree.insert(segments, StorageTier.CPU)

        # Partial match (first segment only)
        query = [
            make_segment([1, 2], start=0, hash_suffix="a"),
            make_segment([5, 6], start=2, hash_suffix="c"),  # Different
        ]
        result = tree.match_prefix(query)

        assert result.matched_tokens == 2
        assert len(result.matched_segments) == 1

    def test_match_prefix_none(self):
        """Test no match."""
        tree = SegmentRadixTree()
        segments = [make_segment([1, 2], hash_suffix="a")]
        tree.insert(segments, StorageTier.CPU)

        # No match
        query = [make_segment([9, 9], hash_suffix="x")]
        result = tree.match_prefix(query)

        assert result.matched_tokens == 0
        assert len(result.matched_segments) == 0

    def test_update_storage_tier(self):
        """Test updating storage tier."""
        tree = SegmentRadixTree()
        seg = make_segment([1, 2, 3], hash_suffix="test")
        metas = tree.insert([seg], StorageTier.CPU)

        # Update tier
        success = tree.update_storage_tier("hash_test", StorageTier.DISK)
        assert success

        # Verify
        meta = tree.lookup("hash_test")
        assert meta.storage_tier == StorageTier.DISK

    def test_remove(self):
        """Test removing a segment."""
        tree = SegmentRadixTree()
        seg = make_segment([1, 2, 3], hash_suffix="test")
        tree.insert([seg], StorageTier.CPU)

        assert len(tree) == 1

        # Remove
        success = tree.remove("hash_test")
        assert success
        assert len(tree) == 0

    def test_get_lru_candidates(self):
        """Test getting LRU candidates."""
        tree = SegmentRadixTree()

        # Insert segments with time gaps
        seg1 = make_segment([1], hash_suffix="old")
        tree.insert([seg1], StorageTier.CPU)
        time.sleep(0.01)

        seg2 = make_segment([2], hash_suffix="new")
        tree.insert([seg2], StorageTier.CPU)

        # Get LRU (oldest)
        candidates = tree.get_lru_candidates(1)

        assert len(candidates) == 1
        assert candidates[0].segment_hash == "hash_old"

    def test_get_lfu_candidates(self):
        """Test getting LFU candidates."""
        tree = SegmentRadixTree()

        seg1 = make_segment([1], hash_suffix="low")
        seg2 = make_segment([2], hash_suffix="high")

        tree.insert([seg1], StorageTier.CPU)
        tree.insert([seg2], StorageTier.CPU)

        # Access seg2 multiple times
        for _ in range(5):
            tree.match_prefix([seg2])

        # Get LFU (least accessed)
        candidates = tree.get_lfu_candidates(1)

        assert len(candidates) == 1
        assert candidates[0].segment_hash == "hash_low"

    def test_clear(self):
        """Test clearing the tree."""
        tree = SegmentRadixTree()

        for i in range(5):
            seg = make_segment([i], hash_suffix=str(i))
            tree.insert([seg], StorageTier.CPU)

        assert len(tree) == 5

        tree.clear()

        assert len(tree) == 0
        assert tree.get_all_metas() == []

    def test_stats(self):
        """Test stats method."""
        tree = SegmentRadixTree()

        tree.insert([make_segment([1, 2], hash_suffix="cpu")], StorageTier.CPU)
        tree.insert([make_segment([3, 4], hash_suffix="disk")], StorageTier.DISK)

        stats = tree.stats()

        assert stats["total_segments"] == 2
        assert stats["total_tokens"] == 4
        assert stats["tier_distribution"]["cpu"] == 1
        assert stats["tier_distribution"]["disk"] == 1
