"""Tests for segment module."""

import pytest
from nano_lmcache.segment import Segment, SegmentSplitter, RollingHasher


class TestSegmentSplitter:
    """Tests for SegmentSplitter."""

    def test_empty_input(self):
        """Test with empty token list."""
        splitter = SegmentSplitter()
        segments = splitter.split([])
        assert segments == []

    def test_short_sequence(self):
        """Test with sequence shorter than min_length."""
        splitter = SegmentSplitter(min_length=32, max_length=512)
        tokens = list(range(20))
        segments = splitter.split(tokens)

        assert len(segments) == 1
        assert segments[0].tokens == tokens
        assert segments[0].start_idx == 0
        assert segments[0].end_idx == 20

    def test_long_sequence(self):
        """Test with sequence longer than max_length."""
        splitter = SegmentSplitter(max_length=100)
        tokens = list(range(250))
        segments = splitter.split(tokens)

        # Should be split into multiple segments
        assert len(segments) >= 2

        # Each segment should be <= max_length
        for seg in segments:
            assert len(seg) <= 100

        # All tokens should be covered
        all_tokens = []
        for seg in segments:
            all_tokens.extend(seg.tokens)
        assert all_tokens == tokens

    def test_boundary_tokens(self):
        """Test splitting on boundary tokens."""
        # Token 999 is a boundary
        splitter = SegmentSplitter(
            min_length=1,
            max_length=100,
            boundary_tokens=[999],
        )

        tokens = [1, 2, 3, 999, 4, 5, 6, 999, 7, 8]
        segments = splitter.split(tokens)

        # Should split at boundaries
        assert len(segments) >= 2

    def test_segment_hashes_unique(self):
        """Test that different segments have different hashes."""
        splitter = SegmentSplitter()

        tokens1 = [1, 2, 3, 4, 5]
        tokens2 = [1, 2, 3, 4, 6]  # Different last token

        seg1 = splitter.split(tokens1)[0]
        seg2 = splitter.split(tokens2)[0]

        assert seg1.hash_value != seg2.hash_value

    def test_segment_hashes_consistent(self):
        """Test that same tokens produce same hash."""
        splitter = SegmentSplitter()

        tokens = [1, 2, 3, 4, 5]

        seg1 = splitter.split(tokens)[0]
        seg2 = splitter.split(tokens)[0]

        assert seg1.hash_value == seg2.hash_value


class TestRollingHasher:
    """Tests for RollingHasher."""

    def test_basic_hash(self):
        """Test basic hashing."""
        hasher = RollingHasher()
        h = hasher.hash_segment([1, 2, 3])
        assert isinstance(h, str)
        assert len(h) == 16

    def test_position_sensitive(self):
        """Test that hash is position-sensitive."""
        hasher = RollingHasher()

        # Hash segment A then B
        h1_a = hasher.hash_segment([1, 2, 3])
        h1_b = hasher.hash_segment([4, 5, 6])

        hasher.reset()

        # Hash segment B then A (reversed order)
        h2_b = hasher.hash_segment([4, 5, 6])
        h2_a = hasher.hash_segment([1, 2, 3])

        # Same content but different position -> different hash
        assert h1_b != h2_b

    def test_hash_sequence(self):
        """Test hashing a sequence of segments."""
        hasher = RollingHasher()

        segments = [[1, 2], [3, 4], [5, 6]]
        hashes = hasher.hash_sequence(segments)

        assert len(hashes) == 3
        assert all(len(h) == 16 for h in hashes)

    def test_reset(self):
        """Test resetting the hasher."""
        hasher = RollingHasher()

        h1 = hasher.hash_segment([1, 2, 3])
        hasher.reset()
        h2 = hasher.hash_segment([1, 2, 3])

        # After reset, same input should give same hash
        assert h1 == h2


class TestSegment:
    """Tests for Segment dataclass."""

    def test_len(self):
        """Test __len__ method."""
        seg = Segment(
            tokens=[1, 2, 3, 4, 5],
            start_idx=0,
            end_idx=5,
            hash_value="abc123",
        )
        assert len(seg) == 5

    def test_repr(self):
        """Test __repr__ method."""
        seg = Segment(
            tokens=[1, 2, 3],
            start_idx=10,
            end_idx=13,
            hash_value="abcdef1234567890",
        )
        repr_str = repr(seg)
        assert "tokens=3" in repr_str
        assert "10:13" in repr_str
