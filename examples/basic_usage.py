"""
Basic usage example for NanoLMCache.

This example demonstrates:
1. Creating a cache instance
2. Storing KV cache
3. Retrieving with prefix matching
4. Cache statistics
"""

import torch
from nano_lmcache import NanoLMCache, NanoLMCacheConfig, StorageConfig, SegmentConfig


def create_mock_kv_cache(
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    seq_len: int = 512,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Create a mock KV cache tensor."""
    # Shape: [num_layers, 2, seq_len, num_heads, head_dim]
    return torch.randn(num_layers, 2, seq_len, num_heads, head_dim, dtype=dtype)


def main():
    print("=" * 60)
    print("NanoLMCache Basic Usage Example")
    print("=" * 60)

    # 1. Create configuration
    config = NanoLMCacheConfig(
        storage=StorageConfig(
            cpu_size_gb=1.0,
            disk_size_gb=2.0,
            disk_cache_dir="/tmp/nano_lmcache_example",
        ),
        segment=SegmentConfig(
            max_segment_length=256,
            min_segment_length=32,
        ),
        enable_async_write=False,  # Sync for demo clarity
        enable_logging=True,
        log_level="INFO",
    )

    # 2. Create cache instance
    with NanoLMCache(config) as cache:
        print(f"\nInitialized: {cache}")

        # 3. Create mock data
        seq_len = 512
        tokens = list(range(seq_len))  # Mock token IDs
        kv_cache = create_mock_kv_cache(seq_len=seq_len)

        print(f"\nMock data:")
        print(f"  Tokens: {len(tokens)}")
        print(f"  KV shape: {kv_cache.shape}")
        print(f"  KV size: {kv_cache.numel() * kv_cache.element_size() / 1e6:.2f} MB")

        # 4. Store KV cache
        print("\n--- Storing KV cache ---")
        keys = cache.store(tokens, kv_cache, async_write=False)
        print(f"Stored {len(keys)} segments")

        # 5. Check stats
        stats = cache.stats()
        print(f"\nCache stats after store:")
        print(f"  Total segments: {stats['index']['total_segments']}")
        print(f"  Total tokens: {stats['index']['total_tokens']}")
        print(f"  CPU utilization: {stats['storage']['cpu']['utilization']:.1%}")

        # 6. Retrieve with exact match
        print("\n--- Retrieving (exact match) ---")
        retrieved, matched = cache.retrieve(tokens)
        print(f"Matched {matched}/{len(tokens)} tokens")
        if retrieved is not None:
            print(f"Retrieved shape: {retrieved.shape}")

        # 7. Retrieve with prefix match
        print("\n--- Retrieving (prefix match) ---")
        # Use first 300 tokens + some new tokens
        partial_tokens = tokens[:300] + [9999, 9998, 9997]
        retrieved, matched = cache.retrieve(partial_tokens)
        print(f"Matched {matched}/{len(partial_tokens)} tokens (prefix)")

        # 8. Retrieve with no match
        print("\n--- Retrieving (no match) ---")
        new_tokens = [9000 + i for i in range(100)]  # Completely new tokens
        retrieved, matched = cache.retrieve(new_tokens)
        print(f"Matched {matched}/{len(new_tokens)} tokens")

        # 9. Final stats
        print("\n--- Final Statistics ---")
        stats = cache.stats()
        print(f"Index stats: {stats['index']}")
        print(f"Storage stats:")
        for tier, tier_stats in stats["storage"].items():
            print(f"  {tier}: {tier_stats}")

        # 10. Clear cache
        print("\n--- Clearing cache ---")
        cache.clear()
        print(f"After clear: {cache}")


if __name__ == "__main__":
    main()
