"""Main NanoLMCache engine."""

from typing import List, Optional, Tuple, Dict, Any
import torch
import threading
from queue import Queue
import logging

from .config import NanoLMCacheConfig
from .segment import Segment, SegmentSplitter
from .index import SegmentRadixTree, StorageTier, MatchResult
from .storage.manager import TieredStorageManager


logger = logging.getLogger(__name__)


class NanoLMCache:
    """
    NanoLMCache: A minimal KV cache management engine.

    Features:
    - Variable-length segment-based caching
    - Radix tree for efficient prefix matching
    - Tiered storage (CPU -> Disk)
    - Async write support
    - LRU eviction
    """

    def __init__(self, config: Optional[NanoLMCacheConfig] = None):
        """
        Initialize NanoLMCache.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or NanoLMCacheConfig()

        # Setup logging
        if self.config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # Initialize components
        self.splitter = SegmentSplitter(
            max_length=self.config.segment.max_segment_length,
            min_length=self.config.segment.min_segment_length,
            boundary_tokens=self.config.segment.boundary_tokens,
        )

        self.index = SegmentRadixTree()

        self.storage = TieredStorageManager(
            cpu_size_gb=self.config.storage.cpu_size_gb,
            disk_cache_dir=self.config.storage.disk_cache_dir,
            disk_size_gb=self.config.storage.disk_size_gb,
            use_pinned=self.config.storage.use_pinned_memory,
        )

        # Async write support
        self._async_enabled = self.config.enable_async_write
        self._write_queue: Queue = Queue()
        self._write_thread: Optional[threading.Thread] = None

        if self._async_enabled:
            self._start_async_writer()

        logger.info(f"NanoLMCache initialized with config: {self.config}")

    def _start_async_writer(self):
        """Start the async write worker thread."""
        self._write_thread = threading.Thread(
            target=self._async_write_worker,
            daemon=True,
        )
        self._write_thread.start()

    def _async_write_worker(self):
        """Background worker for async writes."""
        while True:
            try:
                item = self._write_queue.get()
                if item is None:  # Shutdown signal
                    break

                key, tensor, segment = item
                success, tier = self.storage.put(key, tensor)

                if success:
                    self.index.insert([segment], tier)
                    logger.debug(f"Async write: {key} -> {tier.value}")
                else:
                    logger.warning(f"Async write failed: {key}")

            except Exception as e:
                logger.error(f"Async write error: {e}")
            finally:
                self._write_queue.task_done()

    def store(
        self,
        tokens: List[int],
        kv_cache: torch.Tensor,
        async_write: bool = True,
    ) -> List[str]:
        """
        Store KV cache for a token sequence.

        Args:
            tokens: List of token IDs
            kv_cache: KV cache tensor
                Shape: [num_layers, 2, seq_len, num_heads, head_dim]
                The dim=1 contains K and V respectively
            async_write: Whether to write asynchronously

        Returns:
            List of storage keys for the stored segments
        """
        if not tokens:
            return []

        # Split into segments
        segments = self.splitter.split(tokens)

        if not segments:
            return []

        # Slice KV cache for each segment
        kv_slices = self._slice_kv_cache(kv_cache, segments)

        # Store each segment
        keys = []
        for seg, kv_slice in zip(segments, kv_slices):
            key = f"kv_{seg.hash_value}"

            if async_write and self._async_enabled:
                # Queue for async write
                cpu_copy = kv_slice.cpu().clone()
                self._write_queue.put((key, cpu_copy, seg))
            else:
                # Sync write
                success, tier = self.storage.put(key, kv_slice)
                if success:
                    self.index.insert([seg], tier)
                    logger.debug(f"Stored: {key} -> {tier.value}")

            keys.append(key)

        logger.info(f"Store: {len(tokens)} tokens -> {len(keys)} segments")
        return keys

    def retrieve(
        self,
        tokens: List[int],
        target_device: str = "cpu",
    ) -> Tuple[Optional[torch.Tensor], int]:
        """
        Retrieve KV cache for a token sequence.

        Args:
            tokens: List of token IDs
            target_device: Device to load tensors to

        Returns:
            (kv_cache, matched_token_count)
            Returns (None, 0) if no match found
        """
        if not tokens:
            return None, 0

        # Split into segments
        segments = self.splitter.split(tokens)

        if not segments:
            return None, 0

        # Match prefix
        match_result = self.index.match_prefix(segments)

        if match_result.matched_tokens == 0:
            logger.debug("Retrieve: No match found")
            return None, 0

        # Load matched KV caches
        kv_slices = []
        for meta in match_result.matched_segments:
            kv = self.storage.get(meta.storage_key, target_device)

            if kv is None:
                # Data lost, stop here
                logger.warning(f"Data lost for key: {meta.storage_key}")
                break

            kv_slices.append(kv)

        if not kv_slices:
            return None, 0

        # Concatenate KV caches along sequence dimension
        # Assuming shape: [num_layers, 2, seq_len, num_heads, head_dim]
        kv_cache = torch.cat(kv_slices, dim=2)

        actual_matched = sum(
            meta.num_tokens
            for meta in match_result.matched_segments[: len(kv_slices)]
        )

        logger.info(
            f"Retrieve: {actual_matched}/{len(tokens)} tokens matched "
            f"({len(kv_slices)} segments)"
        )

        return kv_cache, actual_matched

    def prefetch(
        self,
        tokens: List[int],
        target_tier: StorageTier = StorageTier.CPU,
    ) -> int:
        """
        Prefetch KV cache to a faster tier.

        Args:
            tokens: Token sequence to prefetch
            target_tier: Target storage tier

        Returns:
            Number of segments prefetched
        """
        segments = self.splitter.split(tokens)
        match_result = self.index.match_prefix(segments)

        prefetched = 0
        for meta in match_result.matched_segments:
            if meta.storage_tier != target_tier:
                if self.storage.promote(meta.storage_key, target_tier):
                    meta.storage_tier = target_tier
                    prefetched += 1

        logger.info(f"Prefetch: {prefetched} segments promoted to {target_tier.value}")
        return prefetched

    def _slice_kv_cache(
        self,
        kv_cache: torch.Tensor,
        segments: List[Segment],
    ) -> List[torch.Tensor]:
        """
        Slice KV cache tensor according to segments.

        Args:
            kv_cache: Full KV cache tensor
                Shape: [num_layers, 2, seq_len, num_heads, head_dim]
            segments: List of segments

        Returns:
            List of KV cache slices
        """
        slices = []
        for seg in segments:
            # Slice along sequence dimension (dim=2)
            kv_slice = kv_cache[:, :, seg.start_idx : seg.end_idx, :, :].clone()
            slices.append(kv_slice)
        return slices

    def evict(self, n: int = 1) -> int:
        """
        Manually evict n segments.

        Args:
            n: Number of segments to evict

        Returns:
            Number of segments actually evicted
        """
        evicted = 0

        # Get LRU candidates
        candidates = self.index.get_lru_candidates(n)

        for meta in candidates:
            if self.storage.delete(meta.storage_key):
                self.index.remove(meta.segment_hash)
                evicted += 1

        logger.info(f"Evicted {evicted} segments")
        return evicted

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "index": self.index.stats(),
            "storage": self.storage.stats(),
            "config": {
                "max_segment_length": self.config.segment.max_segment_length,
                "cpu_size_gb": self.config.storage.cpu_size_gb,
                "disk_size_gb": self.config.storage.disk_size_gb,
            },
        }

    def clear(self):
        """Clear all cached data."""
        self.storage.clear()
        self.index.clear()
        logger.info("Cache cleared")

    def shutdown(self):
        """Shutdown the cache engine."""
        if self._async_enabled and self._write_thread:
            # Wait for pending writes
            self._write_queue.join()
            # Send shutdown signal
            self._write_queue.put(None)
            self._write_thread.join(timeout=5.0)

        logger.info("NanoLMCache shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"NanoLMCache("
            f"segments={stats['index']['total_segments']}, "
            f"tokens={stats['index']['total_tokens']}, "
            f"cpu_util={stats['storage'].get('cpu', {}).get('utilization', 0):.1%})"
        )
