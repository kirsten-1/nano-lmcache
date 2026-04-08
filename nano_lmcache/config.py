"""Configuration for nano-lmcache."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import os


class EvictionPolicy(Enum):
    """Cache eviction policy."""
    LRU = "lru"   # Least Recently Used
    LFU = "lfu"   # Least Frequently Used


@dataclass
class StorageConfig:
    """Storage layer configuration."""

    # CPU storage
    cpu_size_gb: float = 4.0
    use_pinned_memory: bool = True

    # Disk storage
    disk_cache_dir: str = field(
        default_factory=lambda: os.path.join(os.path.expanduser("~"), ".nano_lmcache")
    )
    disk_size_gb: float = 50.0

    # GPU storage (optional, for future use)
    gpu_size_gb: float = 0.0  # 0 means disabled
    gpu_device: str = "cuda:0"


@dataclass
class SegmentConfig:
    """Segment splitting configuration."""

    max_segment_length: int = 512   # Maximum tokens per segment
    min_segment_length: int = 32    # Minimum tokens per segment

    # Semantic splitting options
    split_on_newline: bool = True
    split_on_sentence: bool = True

    # Special tokens that indicate segment boundaries
    # These should be set based on your tokenizer
    boundary_tokens: List[int] = field(default_factory=list)


@dataclass
class NanoLMCacheConfig:
    """Main configuration for NanoLMCache."""

    storage: StorageConfig = field(default_factory=StorageConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)

    # Eviction policy
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU

    # Async operations
    enable_async_write: bool = True
    async_write_workers: int = 2

    # Prefetch
    enable_prefetch: bool = True
    prefetch_count: int = 3

    # Model info (for KV cache shape calculation)
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "float16"

    # Debug
    enable_logging: bool = True
    log_level: str = "INFO"

    def kv_cache_bytes_per_token(self) -> int:
        """Calculate bytes per token for KV cache."""
        dtype_bytes = {"float16": 2, "float32": 4, "bfloat16": 2}
        bytes_per_element = dtype_bytes.get(self.dtype, 2)
        # 2 for K and V, num_layers, num_kv_heads, head_dim
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * bytes_per_element

    def max_cached_tokens(self) -> int:
        """Estimate max tokens that can be cached."""
        total_bytes = (
            self.storage.cpu_size_gb * 1e9 +
            self.storage.disk_size_gb * 1e9
        )
        return int(total_bytes / self.kv_cache_bytes_per_token())
