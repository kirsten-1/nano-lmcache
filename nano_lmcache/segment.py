"""Segment splitting for token sequences."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Segment:
    """Represents a segment of tokens."""

    tokens: List[int]
    start_idx: int
    end_idx: int
    hash_value: str

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"Segment(tokens={len(self.tokens)}, range=[{self.start_idx}:{self.end_idx}], hash={self.hash_value[:8]}...)"


class SegmentSplitter:
    """
    Split token sequences into segments.

    Unlike LMCache's fixed-size chunking, we use variable-length segments
    based on semantic boundaries (sentence endings, special tokens, etc.)

    IMPORTANT: Uses position-dependent hashing via RollingHasher.
    The hash of each segment depends on ALL previous segments in the sequence,
    ensuring the same tokens at different positions produce different hashes.
    """

    def __init__(
        self,
        max_length: int = 512,
        min_length: int = 32,
        boundary_tokens: Optional[List[int]] = None,
    ):
        """
        Initialize the splitter.

        Args:
            max_length: Maximum tokens per segment
            min_length: Minimum tokens per segment (will merge small segments)
            boundary_tokens: Token IDs that indicate segment boundaries
        """
        self.max_length = max_length
        self.min_length = min_length
        self.boundary_tokens = set(boundary_tokens or [])
        self._hasher = RollingHasher()

    def split(self, tokens: List[int]) -> List[Segment]:
        """
        Split token sequence into segments.

        Strategy:
        1. Find natural boundaries (boundary_tokens)
        2. If segment > max_length, force split
        3. If segment < min_length, merge with next

        IMPORTANT: Hash is position-dependent. The same tokens at different
        positions in the sequence will produce different hashes.

        Args:
            tokens: List of token IDs

        Returns:
            List of Segment objects
        """
        if not tokens:
            return []

        # Reset hasher for new sequence - hash depends on position in THIS sequence
        self._hasher.reset()

        # Find split points
        split_points = self._find_split_points(tokens)

        # Create segments from split points (using rolling hash)
        segments = self._create_segments(tokens, split_points)

        return segments

    def _find_split_points(self, tokens: List[int]) -> List[int]:
        """Find all potential split points."""
        points = [0]  # Start is always a split point

        for i, token in enumerate(tokens):
            if token in self.boundary_tokens:
                # Split after boundary token
                points.append(i + 1)

        # End is always a split point
        if points[-1] != len(tokens):
            points.append(len(tokens))

        return sorted(set(points))

    def _create_segments(
        self, tokens: List[int], split_points: List[int]
    ) -> List[Segment]:
        """Create segments from split points, handling min/max constraints."""
        segments = []
        i = 0

        while i < len(split_points) - 1:
            start = split_points[i]
            end = split_points[i + 1]

            # Handle segments that are too long
            while end - start > self.max_length:
                seg_end = start + self.max_length
                seg_tokens = tokens[start:seg_end]
                segments.append(self._make_segment(seg_tokens, start, seg_end))
                start = seg_end

            # Handle segments that are too short
            remaining = end - start
            if remaining < self.min_length and i + 2 < len(split_points):
                # Try to merge with next segment
                i += 1
                end = split_points[i + 1]

            # Handle potentially still too long after merge
            while end - start > self.max_length:
                seg_end = start + self.max_length
                seg_tokens = tokens[start:seg_end]
                segments.append(self._make_segment(seg_tokens, start, seg_end))
                start = seg_end

            # Create final segment for this range
            if start < end:
                seg_tokens = tokens[start:end]
                segments.append(self._make_segment(seg_tokens, start, end))

            i += 1

        return segments

    def _make_segment(self, tokens: List[int], start: int, end: int) -> Segment:
        """Create a Segment object with position-dependent hash."""
        # Use rolling hasher - hash depends on all previous segments
        hash_value = self._hasher.hash_segment(tokens)
        return Segment(
            tokens=tokens,
            start_idx=start,
            end_idx=end,
            hash_value=hash_value,
        )


class RollingHasher:
    """
    Rolling hash for position-dependent segment hashing.

    The hash of a segment depends on all previous segments,
    ensuring the same tokens at different positions have different hashes.
    """

    def __init__(self, base: int = 31, mod: int = 10**9 + 7):
        self.base = base
        self.mod = mod
        self.prev_hash = 0

    def reset(self):
        """Reset the hasher state."""
        self.prev_hash = 0

    def hash_segment(self, tokens: List[int]) -> str:
        """
        Compute position-dependent hash for a segment.

        The hash incorporates the previous hash, making it position-sensitive.
        """
        h = self.prev_hash
        for token in tokens:
            h = (h * self.base + token) % self.mod

        self.prev_hash = h
        return hex(h)[2:].zfill(16)

    def hash_sequence(self, segments: List[List[int]]) -> List[str]:
        """Hash a sequence of segments."""
        self.reset()
        return [self.hash_segment(seg) for seg in segments]
