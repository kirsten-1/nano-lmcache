"""
vLLM Integration for nano-lmcache.

This module provides integration between nano-lmcache and vLLM's KV cache system.

Usage:
    from nano_lmcache.integration import VLLMConnector

    # Initialize
    connector = VLLMConnector(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype=torch.float16,
    )

    # Store KV cache from vLLM
    connector.store(token_ids, kv_cache)

    # Retrieve and check for prefix match
    cached_kv, matched_len = connector.retrieve(token_ids)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import torch
import logging

from ..config import NanoLMCacheConfig, StorageConfig, SegmentConfig
from ..engine import NanoLMCache
from ..index import StorageTier


logger = logging.getLogger(__name__)


@dataclass
class NanoLMCacheVLLMConfig:
    """Configuration for vLLM integration."""

    # Model configuration
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "float16"

    # Storage configuration
    gpu_size_gb: float = 2.0      # GPU cache size
    cpu_size_gb: float = 8.0      # CPU cache size
    disk_size_gb: float = 50.0    # Disk cache size
    disk_cache_dir: str = "/tmp/nano_lmcache_vllm"

    # Segment configuration
    chunk_size: int = 256         # Tokens per chunk (for compatibility)
    max_segment_length: int = 512
    min_segment_length: int = 64

    # Features
    enable_prefix_caching: bool = True
    enable_gpu_cache: bool = True
    enable_async_write: bool = True
    auto_promote_on_hit: bool = True

    # vLLM specific
    block_size: int = 16          # vLLM block size
    gpu_device: str = "cuda:0"


@dataclass
class VLLMIntegrationStatus:
    """Describe which connector paths are actually implemented."""

    direct_tensor_store_supported: bool = True
    direct_tensor_retrieve_supported: bool = True
    vllm_block_conversion_supported: bool = False
    notes: str = (
        "Direct tensor KV cache storage is supported. "
        "Native vLLM block conversion still requires version-specific runtime hooks."
    )


class VLLMConnector:
    """
    Connector between vLLM and nano-lmcache.

    This class handles:
    1. Converting vLLM's block-based KV cache to tensor format
    2. Storing/retrieving KV cache with prefix matching
    3. Managing GPU/CPU/Disk tiers
    """

    def __init__(
        self,
        config: Optional[NanoLMCacheVLLMConfig] = None,
        **kwargs,
    ):
        """
        Initialize the connector.

        Args:
            config: VLLMConnector configuration
            **kwargs: Override config fields
        """
        self.config = config or NanoLMCacheVLLMConfig()

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Create NanoLMCache config
        cache_config = NanoLMCacheConfig(
            storage=StorageConfig(
                gpu_size_gb=self.config.gpu_size_gb if self.config.enable_gpu_cache else 0,
                cpu_size_gb=self.config.cpu_size_gb,
                disk_size_gb=self.config.disk_size_gb,
                disk_cache_dir=self.config.disk_cache_dir,
                gpu_device=self.config.gpu_device,
            ),
            segment=SegmentConfig(
                max_segment_length=self.config.max_segment_length,
                min_segment_length=self.config.min_segment_length,
            ),
            num_layers=self.config.num_layers,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            dtype=self.config.dtype,
            enable_async_write=self.config.enable_async_write,
        )

        # Initialize cache engine
        self.cache = NanoLMCache(cache_config)
        self._status = VLLMIntegrationStatus()

        # Statistics
        self._stats = {
            "store_count": 0,
            "retrieve_count": 0,
            "hit_count": 0,
            "miss_count": 0,
            "total_tokens_stored": 0,
            "total_tokens_hit": 0,
        }

        logger.info(f"VLLMConnector initialized with config: {self.config}")

    def store(
        self,
        token_ids: List[int],
        kv_cache: torch.Tensor,
        async_write: bool = True,
    ) -> List[str]:
        """
        Store KV cache for a token sequence.

        Args:
            token_ids: List of token IDs
            kv_cache: KV cache tensor from vLLM
                Expected shape: [num_layers, 2, seq_len, num_kv_heads, head_dim]
            async_write: Whether to write asynchronously

        Returns:
            List of storage keys
        """
        if not self.config.enable_prefix_caching:
            return []

        # Validate shape
        expected_shape = (
            self.config.num_layers,
            2,
            len(token_ids),
            self.config.num_kv_heads,
            self.config.head_dim,
        )

        if kv_cache.shape != expected_shape:
            logger.warning(
                f"KV cache shape mismatch. Expected {expected_shape}, got {kv_cache.shape}"
            )
            # Try to handle common shape variations
            kv_cache = self._normalize_kv_shape(kv_cache, len(token_ids))

        keys = self.cache.store(token_ids, kv_cache, async_write=async_write)

        self._stats["store_count"] += 1
        self._stats["total_tokens_stored"] += len(token_ids)

        return keys

    def integration_status(self) -> Dict[str, Any]:
        """Return which integration paths are implemented."""
        return {
            "direct_tensor_store_supported": self._status.direct_tensor_store_supported,
            "direct_tensor_retrieve_supported": self._status.direct_tensor_retrieve_supported,
            "vllm_block_conversion_supported": self._status.vllm_block_conversion_supported,
            "notes": self._status.notes,
        }

    def retrieve(
        self,
        token_ids: List[int],
        target_device: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], int]:
        """
        Retrieve KV cache for a token sequence.

        Args:
            token_ids: List of token IDs
            target_device: Device to load tensor to (default: gpu_device)

        Returns:
            (kv_cache, matched_token_count)
            Returns (None, 0) if no match found
        """
        if not self.config.enable_prefix_caching:
            return None, 0

        if target_device is None:
            target_device = self.config.gpu_device

        kv_cache, matched_len = self.cache.retrieve(token_ids, target_device)

        self._stats["retrieve_count"] += 1

        if matched_len > 0:
            self._stats["hit_count"] += 1
            self._stats["total_tokens_hit"] += matched_len
            logger.debug(f"Cache hit: {matched_len}/{len(token_ids)} tokens")
        else:
            self._stats["miss_count"] += 1
            logger.debug("Cache miss")

        return kv_cache, matched_len

    def prefetch(
        self,
        token_ids: List[int],
        target_tier: StorageTier = StorageTier.GPU,
    ) -> int:
        """
        Prefetch KV cache to a faster tier.

        Args:
            token_ids: Token sequence to prefetch
            target_tier: Target storage tier

        Returns:
            Number of segments prefetched
        """
        return self.cache.prefetch(token_ids, target_tier)

    def store_from_vllm_blocks(
        self,
        token_ids: List[int],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> List[str]:
        """
        Store KV cache from vLLM's block format.

        vLLM stores KV cache in blocks with shape:
        - key_cache: [num_blocks, num_kv_heads, head_dim // x, block_size, x]
        - value_cache: [num_blocks, num_kv_heads, head_dim, block_size]

        This method converts to our format:
        - [num_layers, 2, seq_len, num_kv_heads, head_dim]

        Args:
            token_ids: Token IDs
            key_cache: vLLM key cache blocks
            value_cache: vLLM value cache blocks
            slot_mapping: Mapping from positions to slots

        Returns:
            List of storage keys
        """
        # Convert vLLM block format to our tensor format
        kv_cache = self._convert_vllm_blocks_to_tensor(
            key_cache, value_cache, slot_mapping, len(token_ids)
        )

        return self.store(token_ids, kv_cache)

    def retrieve_to_vllm_blocks(
        self,
        token_ids: List[int],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> int:
        """
        Retrieve KV cache and write to vLLM's block format.

        Args:
            token_ids: Token IDs to retrieve
            key_cache: vLLM key cache blocks (will be modified in-place)
            value_cache: vLLM value cache blocks (will be modified in-place)
            slot_mapping: Mapping from positions to slots

        Returns:
            Number of matched tokens (0 if miss)
        """
        kv_cache, matched_len = self.retrieve(token_ids, target_device=key_cache.device)

        if kv_cache is None or matched_len == 0:
            return 0

        # Convert our tensor format back to vLLM blocks
        self._write_tensor_to_vllm_blocks(
            kv_cache, key_cache, value_cache, slot_mapping, matched_len
        )

        return matched_len

    def _normalize_kv_shape(
        self,
        kv_cache: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Normalize KV cache to expected shape."""
        # Handle common variations
        # Expected: [num_layers, 2, seq_len, num_kv_heads, head_dim]

        if kv_cache.dim() == 4:
            # Might be [num_layers, seq_len, num_kv_heads, head_dim * 2]
            # or [2, num_layers, seq_len, num_heads * head_dim]
            logger.warning("4D tensor detected, attempting reshape")

        return kv_cache

    def _convert_vllm_blocks_to_tensor(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Convert vLLM block format to our tensor format.

        This conversion is version-specific and is not implemented yet.
        """
        raise NotImplementedError(
            "vLLM block-to-tensor conversion is not implemented yet. "
            "Use direct tensor store/retrieve APIs or provide version-specific "
            "runtime hooks for vLLM's KV cache layout."
        )

    def _write_tensor_to_vllm_blocks(
        self,
        kv_cache: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_len: int,
    ):
        """
        Write our tensor format to vLLM blocks.

        This conversion is version-specific and is not implemented yet.
        """
        raise NotImplementedError(
            "Writing tensor KV cache back into vLLM block storage is not implemented yet. "
            "A version-specific adapter is required for the active vLLM runtime."
        )

    def stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        cache_stats = self.cache.stats()

        hit_rate = (
            self._stats["hit_count"] / self._stats["retrieve_count"]
            if self._stats["retrieve_count"] > 0
            else 0.0
        )

        token_hit_rate = (
            self._stats["total_tokens_hit"] / self._stats["total_tokens_stored"]
            if self._stats["total_tokens_stored"] > 0
            else 0.0
        )

        return {
            "connector": {
                **self._stats,
                "hit_rate": hit_rate,
                "token_hit_rate": token_hit_rate,
            },
            "integration_status": self.integration_status(),
            "cache": cache_stats,
        }

    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self._stats = {
            "store_count": 0,
            "retrieve_count": 0,
            "hit_count": 0,
            "miss_count": 0,
            "total_tokens_stored": 0,
            "total_tokens_hit": 0,
        }

    def shutdown(self):
        """Shutdown the connector."""
        self.cache.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


class VLLMCacheWrapper:
    """
    Wrapper that can be used as a drop-in replacement for vLLM's cache.

    This is an experimental feature for deeper vLLM integration.
    """

    def __init__(
        self,
        original_cache,
        connector: VLLMConnector,
    ):
        """
        Wrap vLLM's cache with nano-lmcache.

        Args:
            original_cache: The original vLLM cache object
            connector: NanoLMCache connector
        """
        self._original = original_cache
        self._connector = connector

    def __getattr__(self, name):
        """Forward attribute access to original cache."""
        return getattr(self._original, name)
