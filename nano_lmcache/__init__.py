"""
nano-lmcache: A minimal, educational implementation of LMCache

Core components:
- NanoLMCache: Main cache engine
- Config: Configuration dataclass
- Segment: Token sequence segmentation
- RadixTree: Efficient prefix matching index
- TieredStorage: Multi-tier storage management (CPU/Disk)
"""

from .config import (
    NanoLMCacheConfig,
    StorageConfig,
    SegmentConfig,
    EvictionPolicy,
)
from .engine import NanoLMCache
from .segment import Segment, SegmentSplitter
from .index import SegmentRadixTree, StorageTier, MatchResult

__version__ = "0.1.0"
__all__ = [
    "NanoLMCache",
    "NanoLMCacheConfig",
    "StorageConfig",
    "SegmentConfig",
    "EvictionPolicy",
    "Segment",
    "SegmentSplitter",
    "SegmentRadixTree",
    "StorageTier",
    "MatchResult",
]
